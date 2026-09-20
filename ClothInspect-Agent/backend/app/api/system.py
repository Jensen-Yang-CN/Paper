# -*- coding: utf-8 -*-
"""系统信息接口：健康检查、配置、内置样例图库、Agent 工具清单。"""
from __future__ import annotations

import random

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import config
from ..core.agent_tools import build_registry
from ..core.detector import YoloDetector
from ..core.vlm_client import verifier_info
from .. import db

router = APIRouter(prefix="/api", tags=["system"])

# ---------------------------------------------------------------- 样例图库缓存
_SAMPLE_CACHE: dict = {"signature": None, "items": []}


def _read_label_counts(stem: str) -> dict[str, int]:
    """读取重标标签文件，统计各类别框数（样例卡片的"参考答案"）。"""
    counts: dict[str, int] = {}
    if config.SAMPLE_DIR is None:
        return counts
    lbl = config.SAMPLE_DIR.parent / "labels" / f"{stem}.txt"
    if not lbl.exists():
        return counts
    try:
        for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split()
            if len(p) >= 5:
                try:
                    name = config.CLASS_NAMES[int(float(p[0]))]
                except Exception:
                    continue
                counts[name] = counts.get(name, 0) + 1
    except Exception:
        pass
    return counts


def build_sample_library(force: bool = False, limit: int = 18) -> list[dict]:
    """构建"新用户友好"的样例库：覆盖三类缺陷 + 含无缺陷的负样本。"""
    if config.SAMPLE_DIR is None:
        return []
    files = sorted(p for p in config.SAMPLE_DIR.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    signature = (str(config.SAMPLE_DIR), len(files))
    if not force and _SAMPLE_CACHE["signature"] == signature:
        return _SAMPLE_CACHE["items"]

    rng = random.Random(20260922)   # 固定种子，保证每次刷新样例一致
    buckets: dict[str, list[dict]] = {"hole": [], "stain": [], "wrinkle": [], "clean": []}
    for p in files:
        counts = _read_label_counts(p.stem)
        if not counts:
            buckets["clean"].append({"file": p.name, "counts": {}, "n_defect": 0})
            continue
        for cls in counts:
            if cls in buckets:
                buckets[cls].append({"file": p.name, "counts": counts,
                                     "n_defect": sum(counts.values())})

    picked: list[dict] = []
    seen: set[str] = set()
    # 每类各取若干（有缺陷样本优先），再补几张无缺陷样本
    quota = {"hole": 4, "stain": 4, "wrinkle": 5, "clean": 5}
    for cls, q in quota.items():
        pool = [x for x in buckets[cls] if x["file"] not in seen]
        rng.shuffle(pool)
        for item in pool[:q]:
            seen.add(item["file"])
            item = dict(item)
            item["label"] = "无缺陷（负样本）" if cls == "clean" else f"含{config.CLASS_LABELS_ZH[cls]}"
            item["primary_class"] = cls
            picked.append(item)

    items = []
    for it in picked:
        items.append({
            **it,
            "url": f"/api/samples/image/{it['file']}",
            "label_zh": "  ".join(f"{config.CLASS_LABELS_ZH[k]}×{v}"
                                  for k, v in it["counts"].items()) or "无标注缺陷",
        })
    _SAMPLE_CACHE.update(signature=signature, items=items)
    return items


# ---------------------------------------------------------------- 接口
@router.get("/health")
def health():
    """系统健康状态：检测器、大模型复核后端、记录库。"""
    stats = db.statistics()
    return {
        "status": "ok",
        "system": {
            "name": config.SYSTEM_NAME,
            "name_zh": config.SYSTEM_NAME_ZH,
            "version": config.SYSTEM_VERSION,
            "classes": [{"key": k, "label": config.CLASS_LABELS_ZH[k]}
                        for k in config.CLASS_NAMES],
        },
        "detector": YoloDetector.status(),
        "vlm": verifier_info(),
        "samples": {
            "available": config.SAMPLE_DIR is not None,
            "dir": str(config.SAMPLE_DIR) if config.SAMPLE_DIR else None,
            "count": len(build_sample_library()),
        },
        "records": stats,
    }


@router.get("/config")
def get_config():
    """默认参数与推荐操作点（前端"一键推荐值"的数据来源）。"""
    return {
        "defaults": {
            "threshold": config.DEFAULT_ROUTER_THRESHOLD,
            "infer_conf": config.DEFAULT_INFER_CONF,
            "baseline_conf": config.BASELINE_CONF,
            "enable_vlm": True,
            "enable_enhance": False,
        },
        "recommended_points": [
            {"threshold": 0.5, "label": "低预算", "note": "0.48 次复核/张，F1 0.576"},
            {"threshold": 0.6, "label": "推荐（Pareto 膝点）", "note": "0.75 次复核/张，F1 0.599"},
            {"threshold": 0.7, "label": "高精度", "note": "1.30 次复核/张，F1 0.623"},
        ],
        "ranges": {
            "threshold": {"min": 0.30, "max": 0.90, "step": 0.05},
            "infer_conf": {"min": 0.05, "max": 0.60, "step": 0.05},
            "baseline_conf": {"min": 0.10, "max": 0.90, "step": 0.05},
        },
        "crop": {"margin": config.CROP_MARGIN, "min_size": config.MIN_CROP},
        "class_labels": config.CLASS_LABELS_ZH,
        "class_names": config.CLASS_NAMES,
    }


@router.get("/samples")
def list_samples(limit: int = 18):
    """内置样例图库（含"参考答案"，帮助新用户理解系统在做什么）。"""
    items = build_sample_library(limit=limit)
    return {"total": len(items), "items": items[:limit] if limit else items}


@router.get("/samples/image/{name}")
def sample_image(name: str):
    """按文件名返回样例原图。"""
    if config.SAMPLE_DIR is None:
        raise HTTPException(status_code=404, detail="未配置样例图库")
    p = (config.SAMPLE_DIR / name).resolve()
    try:
        p.relative_to(config.SAMPLE_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="非法文件名")
    if not p.exists():
        raise HTTPException(status_code=404, detail="样例不存在")
    return FileResponse(str(p))


@router.get("/tools")
def list_tools():
    """Agent 工具清单（Function Calling 声明 + 中文说明）。"""
    reg = build_registry()
    return {
        "count": len(reg.catalog()),
        "tools": reg.catalog(),
        "function_calling_schema": reg.schemas(),
        "pipeline": [
            {"step": 1, "tool": "detect_defect", "title": "视觉检测",
             "desc": "YOLOv11 冻结模型输出候选缺陷框（类别 / 坐标 / 置信度）"},
            {"step": 2, "tool": "route_decision", "title": "置信度路由",
             "desc": "置信度 ≥ 阈值 T 直接接受；< T 判定为疑难样本，转交复核"},
            {"step": 3, "tool": "crop_region", "title": "局部裁剪",
             "desc": "以外扩 30% 裁出局部区域并放大，附带尺度提示"},
            {"step": 4, "tool": "verify_with_vlm", "title": "多模态复核",
             "desc": "qwen3-vl-plus 判定该区域是否为瑕疵、属于哪一类"},
            {"step": 5, "tool": "fuse_results", "title": "结果融合",
             "desc": "VLM 判为瑕疵则保留并修正类别，否则剔除；框沿用 YOLO 精确定位"},
        ],
    }


@router.get("/model/info")
def model_info():
    """模型与阈值信息卡。"""
    return {
        "detector": YoloDetector.status(),
        "vlm": verifier_info(),
        "params": {
            "img_size": config.IMG_SIZE,
            "device": config.DEVICE,
            "iou_threshold": config.IOU_THRESHOLD,
            "crop_margin": config.CROP_MARGIN,
            "min_crop": config.MIN_CROP,
        },
    }
