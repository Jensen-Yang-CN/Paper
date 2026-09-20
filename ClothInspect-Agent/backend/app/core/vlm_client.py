# -*- coding: utf-8 -*-
"""
多模态大模型复核模块（verify_with_vlm）
======================================
论文 3.5 节：VLM 负责**语义判断**（是不是瑕疵 / 什么类别 / 要不要信），
并给出结构化 JSON 输出；它不负责像素级精确定位（那是 YOLO 的职责）。

本模块提供两种复核后端：
* QwenVLClient      —— 真实调用 qwen3-vl-plus（DashScope OpenAI 兼容模式）；
* HeuristicVerifier —— 离线启发式复核，无 API Key 时兜底，
                       仅用于演示系统流程，**不代表论文结果**。

两者输出统一为 VLMVerdict，便于上层融合逻辑无差别处理。
"""
from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import requests

from .. import config
from .detector import Detection

# 论文第一周迭代确定的"质检提示词 v2"（难样本一致率 90.2%）
PROMPT_VERIFY = """你是服装与面料质检流水线的 AI 质检员。这张图是从质检图裁剪出的局部区域，请判断该区域是否存在瑕疵（hole破洞/stain污渍/wrinkle褶皱，质检场景中褶皱属于瑕疵）。有则 is_defect=true，完全没有才 false。只输出 JSON：{"is_defect": true或false, "category": "hole|stain|wrinkle|none", "confidence": 0到1, "reason": "一句话"}"""

VALID_CATEGORIES = ("hole", "stain", "wrinkle")


@dataclass
class VLMVerdict:
    """VLM 结构化复核结论。"""
    is_defect: bool | None
    category: str            # hole | stain | wrinkle | none
    confidence: float
    reason: str
    mode: str                # real | heuristic
    latency_ms: int
    raw: str = ""            # 原始返回文本（便于排查）
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ 工具
def encode_jpeg_b64(img: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("图像编码失败")
    return base64.b64encode(buf.tobytes()).decode()


def parse_json_loose(text: str) -> dict | None:
    """从模型输出中稳健地抽取 JSON（容忍 ```json 包裹与前后缀说明）。"""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        # 兜底：修掉尾随逗号
        frag = re.sub(r",\s*([}\]])", r"\1", t[s:e + 1])
        try:
            return json.loads(frag)
        except Exception:
            return None


def _normalize_category(cat) -> str:
    c = str(cat or "").strip().lower()
    for v in VALID_CATEGORIES:
        if v in c:
            return v
    if "破洞" in c or "洞" in c:
        return "hole"
    if "污" in c or "渍" in c:
        return "stain"
    if "皱" in c or "褶" in c:
        return "wrinkle"
    return "none"


# ------------------------------------------------------------------ 真实 VLM
class QwenVLClient:
    """qwen3-vl-plus 客户端（DashScope 兼容 OpenAI 协议）。"""

    mode = "real"

    def __init__(self, api_key: str = "", model: str = "", api_url: str = ""):
        self.api_key = api_key or config.VLM_API_KEY
        self.model = model or config.VLM_MODEL
        self.api_url = api_url or config.VLM_API_URL

    def verify(self, crop_img: np.ndarray, scale_prompt: str = "",
               det: Detection | None = None) -> VLMVerdict:
        """统一接口：det 参数仅为与离线后端保持签名一致，真实 VLM 不使用它。"""
        t0 = time.perf_counter()
        try:
            b64 = encode_jpeg_b64(crop_img)
        except Exception as e:
            return VLMVerdict(None, "none", 0.0, "", self.mode, 0, error=str(e))

        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": (scale_prompt + "\n" + PROMPT_VERIFY).strip()},
                ],
            }],
            "temperature": config.VLM_TEMPERATURE,
        }
        try:
            r = requests.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=config.VLM_TIMEOUT,
            )
        except Exception as e:
            return VLMVerdict(None, "none", 0.0, "", self.mode,
                              int((time.perf_counter() - t0) * 1000),
                              error=f"网络异常: {str(e)[:120]}")

        latency = int((time.perf_counter() - t0) * 1000)
        if r.status_code != 200:
            return VLMVerdict(None, "none", 0.0, "", self.mode, latency,
                              raw=r.text[:400],
                              error=f"HTTP {r.status_code}")

        try:
            content = r.json()["choices"][0]["message"]["content"]
        except Exception:
            return VLMVerdict(None, "none", 0.0, "", self.mode, latency,
                              raw=r.text[:400], error="返回结构异常")

        pred = parse_json_loose(content)
        if pred is None:
            return VLMVerdict(None, "none", 0.0, "", self.mode, latency,
                              raw=content[:400], error="JSON 解析失败")

        is_defect = pred.get("is_defect")
        if isinstance(is_defect, str):
            is_defect = is_defect.strip().lower() in ("true", "yes", "1", "是")
        try:
            conf = float(pred.get("confidence") or 0.0)
        except Exception:
            conf = 0.0
        return VLMVerdict(
            is_defect=bool(is_defect) if is_defect is not None else None,
            category=_normalize_category(pred.get("category")),
            confidence=max(0.0, min(1.0, conf)),
            reason=str(pred.get("reason") or "")[:200],
            mode=self.mode,
            latency_ms=latency,
            raw=content[:400],
        )


# ------------------------------------------------------------------ 离线兜底
class HeuristicVerifier:
    """离线启发式复核（无 API Key 时兜底）。

    原理：用经典图像统计量近似"这块局部区域像不像真实瑕疵"——
      * 拉普拉斯方差（清晰度/纹理强度）
      * 灰度标准差（局部对比度）
      * 与整图的亮度偏离度
      * YOLO 自身置信度

    ⚠️ 它**不是大模型**，只用于让系统在完全离线时仍可演示完整流程。
    界面上会明确标注"离线启发式复核"，所有相关结论都不代表论文结果。
    """

    mode = "heuristic"

    def __init__(self, det: Detection | None = None, whole_img: np.ndarray | None = None):
        self.det = det

    def verify(self, crop_img: np.ndarray, scale_prompt: str = "",
               det: Detection | None = None) -> VLMVerdict:
        t0 = time.perf_counter()
        d = det or self.det
        if crop_img is None or crop_img.size == 0:
            return VLMVerdict(None, "none", 0.0, "裁剪区域为空", self.mode, 0,
                              error="empty crop")

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        std = float(np.std(gray))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        conf = float(d.conf) if d else 0.5

        # 打分：低局部对比度 + 低 YOLO 置信度 -> 更可能是织物纹理/假缺陷
        score = 0.0
        score += 0.45 * (1.0 - min(1.0, std / 40.0))
        score += 0.35 * (1.0 - min(1.0, lap_var / 300.0))
        score += 0.20 * (1.0 - min(1.0, conf / 0.6))
        is_defect = score >= 0.45
        category = d.cls_name if d else "none"
        if not is_defect:
            category = "none"

        reason = (f"离线启发式：局部对比度 σ={std:.1f}、清晰度={lap_var:.0f}、"
                  f"YOLO置信度={conf:.2f} → {'判为瑕疵' if is_defect else '判为非瑕疵'}")
        return VLMVerdict(
            is_defect=is_defect,
            category=category if is_defect else "none",
            confidence=round(min(0.95, max(0.05, score)), 3),
            reason=reason,
            mode=self.mode,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )


# ------------------------------------------------------------------ 工厂
def build_verifier():
    """按配置返回实际生效的复核后端。"""
    if config.vlm_effective_mode() == "real":
        return QwenVLClient()
    return HeuristicVerifier()


def verifier_info() -> dict:
    """供 /api/health 与前端状态条展示。"""
    effective = config.vlm_effective_mode()
    return {
        "configured_mode": config.VLM_MODE,
        "effective_mode": effective,
        "is_real": effective == "real",
        "model": config.VLM_MODEL if effective == "real" else "heuristic-local",
        "api_key_present": bool(config.VLM_API_KEY),
        "api_url": config.VLM_API_URL,
        "note": (
            f"真实调用 {config.VLM_MODEL}"
            if effective == "real"
            else "未配置 DASHSCOPE_API_KEY，已降级为离线启发式复核（仅演示流程，不代表论文结果）"
        ),
    }
