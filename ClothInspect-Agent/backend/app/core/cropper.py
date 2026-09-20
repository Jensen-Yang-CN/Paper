# -*- coding: utf-8 -*-
"""
局部区域处理模块（crop_region / enhance_region）
================================================
论文 3.4 节：为多模态大模型提供"更清晰、更聚焦的视觉证据"。

两个能力：
1. crop_region   —— 按检测框外扩一定比例裁出局部区域，并对过小区域自动放大；
2. enhance_region —— 可选图像增强（CLAHE 局部对比度 + 轻微锐化），
   用于提升低对比度微痕的可辨识度。

裁剪结果同时返回"尺度提示"文本：告诉 VLM 这块裁剪图来自多大的原图、
候选框占原图面积比例是多少——论文中该提示已被实验证实有效。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .. import config


@dataclass
class CropResult:
    image: np.ndarray           # 裁剪后的 BGR 图像
    box: tuple[int, int, int, int]   # 裁剪区域在原图中的像素范围
    scale_prompt: str           # 注入到 VLM prompt 的尺度提示
    upscaled: bool              # 是否发生了放大


def crop_region(
    img: np.ndarray,
    det_box: list[float],
    margin: float = config.CROP_MARGIN,
    min_size: int = config.MIN_CROP,
) -> CropResult:
    """按检测框裁剪局部区域。

    参数
    ----
    img       : 原图 BGR
    det_box   : [x1, y1, x2, y2] 像素坐标
    margin    : 外扩比例（0.30 表示向外扩 30% 的框宽/框高）
    min_size  : 裁剪区域最小边长；不足则先扩到该尺寸，再整体放大
    """
    h, w = img.shape[:2]
    bw, bh = det_box[2] - det_box[0], det_box[3] - det_box[1]
    pad_x, pad_y = bw * margin, bh * margin

    x1 = max(0, int(det_box[0] - pad_x))
    y1 = max(0, int(det_box[1] - pad_y))
    x2 = min(w, int(det_box[2] + pad_x))
    y2 = min(h, int(det_box[3] + pad_y))

    # 区域过小：以框中心为基准补足到 min_size
    if x2 - x1 < min_size or y2 - y1 < min_size:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = min_size // 2
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)

    crop = img[y1:y2, x1:x2]

    # 裁剪区域本身仍偏小 -> 放大，让 VLM 看清微痕细节
    upscaled = False
    ch, cw = crop.shape[:2]
    if ch and cw and min(ch, cw) < min_size:
        s = min_size / float(min(ch, cw))
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        upscaled = True

    frac = (bw * bh) / float(w * h) if w and h else 0.0
    ch2, cw2 = crop.shape[:2]
    prompt = (
        f"[尺度提示] 这是原始质检图（{w}x{h}px）中一块区域放大后的裁剪图，"
        f"裁剪图本身 {cw2}x{ch2}px，候选缺陷框占原图面积约 {frac * 100:.1f}%。"
        f"瑕疵可能非常小，请据此判断该区域是否存在瑕疵。"
    )
    return CropResult(image=crop, box=(x1, y1, x2, y2), scale_prompt=prompt, upscaled=upscaled)


def enhance_region(crop: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """可选增强：CLAHE 提升局部对比度 + 轻微非锐化掩蔽。

    针对"低对比度微痕"这一难点类别使用，属于论文 3.4 节的可选步骤。
    """
    if crop is None or crop.size == 0:
        return crop
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    return cv2.addWeighted(out, 1.4, blur, -0.4, 0)


def region_contrast(crop: np.ndarray) -> float:
    """局部灰度标准差——论文 Hard Set 定义中的"对比度"客观属性。"""
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))
