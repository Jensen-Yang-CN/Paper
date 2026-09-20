# -*- coding: utf-8 -*-
"""
Agent 工具注册表（Function Calling 风格）
=========================================
论文第八节要求的最小 Agent 工具集：

    1. detect_defect     —— YOLOv11 视觉检测
    2. crop_region       —— 局部区域裁剪（扩边 + 放大）
    3. verify_with_vlm   —— 多模态大模型复核
    4. enhance_region    —— 局部区域增强（可选）

设计：每个工具声明 name / description / parameters(JSON Schema) / fn。
流水线通过 `registry.call(name, args, runtime=...)` 统一调度：
* args     —— 可 JSON 序列化的显式参数，会被完整记录进**推理轨迹**；
* runtime  —— 大对象（图像数组、模型句柄）等不便记录的上下文，只传入 fn 不落轨迹。

调用过程被记录为推理轨迹（ToolCall），前端据此渲染"Agent 工具调用时间线"，
同时也是软著的功能证据；`/api/tools` 可直接导出 Function Calling 声明。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import config
from . import cropper
from .detector import Detection, YoloDetector

# 非工具型流水线步骤的中文显示名（用于推理轨迹渲染）
PIPELINE_STEP_LABELS = {
    "route_decision": "置信度路由",
    "fuse_results": "结果融合",
}


@dataclass
class ToolCall:
    """一次工具调用的完整记录（推理轨迹的一个节点）。"""
    step: int
    tool: str
    label: str                     # 中文显示名
    args: dict[str, Any]
    summary: str                   # 一句话结论
    latency_ms: int
    status: str = "ok"             # ok | skipped | error
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step": self.step, "tool": self.tool, "label": self.label,
            "args": self.args, "summary": self.summary,
            "latency_ms": self.latency_ms, "status": self.status,
            "detail": self.detail,
        }


class ToolRegistry:
    """工具注册表：声明 JSON Schema 并统一调度执行。"""

    def __init__(self) -> None:
        self._tools: dict[str, dict] = {}
        self._seq = 0

    # ------------------------------------------------------------ 注册
    def register(self, name: str, label: str, description: str,
                 parameters: dict, fn: Callable) -> None:
        self._tools[name] = {
            "name": name, "label": label, "description": description,
            "parameters": parameters, "fn": fn,
        }

    def schemas(self) -> list[dict]:
        """导出 OpenAI Function Calling 风格的工具声明。"""
        return [
            {"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["parameters"]}}
            for t in self._tools.values()
        ]

    def catalog(self) -> list[dict]:
        """供前端"关于 / 工具"页展示。"""
        return [
            {"name": t["name"], "label": t["label"],
             "description": t["description"], "parameters": t["parameters"]}
            for t in self._tools.values()
        ]

    def has(self, name: str) -> bool:
        return name in self._tools

    # ------------------------------------------------------------ 调用
    def call(self, name: str, args: dict | None = None,
             runtime: dict | None = None, summary: str = "",
             status: str = "ok", detail: dict | None = None) -> tuple[Any, ToolCall]:
        """执行工具并生成轨迹节点。"""
        if name not in self._tools:
            raise KeyError(f"未注册的工具: {name}")
        tool = self._tools[name]
        args = dict(args or {})
        runtime = dict(runtime or {})
        t0 = time.perf_counter()
        result = tool["fn"](**args, **runtime)
        latency = int((time.perf_counter() - t0) * 1000)
        return result, self.record(name, args, summary, latency, status, detail)

    def record(self, name: str, args: dict, summary: str, latency_ms: int = 0,
               status: str = "ok", detail: dict | None = None) -> ToolCall:
        """只记录轨迹节点（用于路由决策、融合汇总等非工具步骤）。"""
        label = (self._tools.get(name, {}).get("label")
                 or PIPELINE_STEP_LABELS.get(name, name))
        self._seq += 1
        return ToolCall(step=self._seq, tool=name, label=label, args=args,
                        summary=summary, latency_ms=latency_ms,
                        status=status, detail=detail or {})

    def reset(self) -> None:
        self._seq = 0


# ---------------------------------------------------------------- 工具实现
def _tool_detect_defect(image, conf: float, image_size: int | None = None,
                        detector: YoloDetector | None = None):
    """工具 1：YOLOv11 检测。返回 (检测框列表, 耗时秒)。"""
    det = detector or YoloDetector.instance()
    return det.predict(image, conf=conf)


def _tool_crop_region(image, box: list[float], margin: float | None = None,
                      enhance: bool = False):
    """工具 2：局部裁剪（可选同步增强）。"""
    m = config.CROP_MARGIN if margin is None else margin
    res = cropper.crop_region(image, box, margin=m)
    if enhance:
        res.image = cropper.enhance_region(res.image)
    return res


def _tool_enhance_region(crop, clip_limit: float = 2.0):
    """工具 4：局部增强。"""
    return cropper.enhance_region(crop, clip_limit=clip_limit)


def _tool_verify_with_vlm(verifier, crop, scale_prompt: str,
                          det: Detection | None = None,
                          crop_shape: list | None = None,
                          yolo_cls: str | None = None,
                          yolo_conf: float | None = None):
    """工具 3：多模态复核（统一签名，兼容 real / heuristic 两种后端）。

    crop_shape / yolo_cls / yolo_conf 仅用于推理轨迹的展示，不参与计算。
    """
    return verifier.verify(crop, scale_prompt, det=det)


_TOOL_DETECT_PARAMS = {
    "type": "object",
    "properties": {
        "conf": {"type": "number", "description": "推理置信度下限，低于该值的候选框不输出"},
        "image_size": {"type": "integer", "description": "推理输入尺寸，默认 640"},
    },
    "required": ["conf"],
}

_TOOL_CROP_PARAMS = {
    "type": "object",
    "properties": {
        "box": {"type": "array", "items": {"type": "number"},
                "description": "检测框像素坐标 [x1, y1, x2, y2]"},
        "margin": {"type": "number", "description": "外扩比例，默认 0.30（向外扩 30% 框宽/框高）"},
        "enhance": {"type": "boolean", "description": "是否同步做对比度增强，默认 false"},
    },
    "required": ["box"],
}

_TOOL_VLM_PARAMS = {
    "type": "object",
    "properties": {
        "scale_prompt": {"type": "string", "description": "尺度提示文本（告知原图尺寸与候选框占比）"},
        "crop_shape": {"type": "array", "items": {"type": "integer"},
                       "description": "裁剪图尺寸 [高, 宽]"},
        "yolo_conf": {"type": "number", "description": "该候选框的 YOLO 原始置信度"},
        "yolo_cls": {"type": "string", "description": "该候选框的 YOLO 原始类别"},
    },
    "required": ["scale_prompt"],
}

_TOOL_ENHANCE_PARAMS = {
    "type": "object",
    "properties": {
        "clip_limit": {"type": "number", "description": "CLAHE 对比度限幅，默认 2.0"},
    },
    "required": [],
}


def build_registry() -> ToolRegistry:
    """构建论文要求的 4 个 Agent 工具。"""
    reg = ToolRegistry()

    reg.register(
        name="detect_defect", label="视觉检测",
        description="调用冻结的 YOLOv11 模型对整图做一次前向推理，返回所有候选缺陷框的"
                    "类别、像素坐标与置信度。定位精确，但低置信度样本与褶皱类容易漏检。",
        parameters=_TOOL_DETECT_PARAMS,
        fn=_tool_detect_defect,
    )

    reg.register(
        name="crop_region", label="局部裁剪",
        description="以检测框为中心外扩指定比例裁出局部区域，区域过小时自动放大，"
                    "并生成尺度提示（告知大模型原始尺度）——为复核提供更聚焦的视觉证据。",
        parameters=_TOOL_CROP_PARAMS,
        fn=_tool_crop_region,
    )

    reg.register(
        name="verify_with_vlm", label="多模态复核",
        description="把裁剪出的局部区域交给多模态大模型（qwen3-vl-plus）做语义复核，"
                    "返回结构化结论：是否为瑕疵 / 类别 / 置信度 / 理由。"
                    "大模型擅长语义判断，但定位粗，不替代 YOLO 的精确框。",
        parameters=_TOOL_VLM_PARAMS,
        fn=_tool_verify_with_vlm,
    )

    reg.register(
        name="enhance_region", label="区域增强",
        description="对局部区域做 CLAHE 对比度增强与轻微锐化，用于低对比度微痕的辅助辨识"
                    "（论文中的可选步骤，默认关闭）。",
        parameters=_TOOL_ENHANCE_PARAMS,
        fn=_tool_enhance_region,
    )
    return reg
