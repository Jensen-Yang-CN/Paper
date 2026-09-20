# -*- coding: utf-8 -*-
"""
检测报告导出模块
================
把一次协同推理的结果渲染成一张可直接用于软著材料 / 论文插图 / 交付的
可视化报告图：原图 + 缺陷框 + 每框决策溯源 + 汇总条。

说明：前端页面上的框由浏览器 SVG 实时绘制（中文标签更清晰）；
本模块用于**离线导出**，因此在这里完成图像合成。
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config

# 类别配色（与前端保持一致）
CLASS_COLORS = {
    "hole": (239, 68, 68),      # 红
    "stain": (245, 158, 11),    # 琥珀
    "wrinkle": (99, 102, 241),  # 靛蓝
}
_ACCENT = (59, 130, 246)

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _font(size: int) -> ImageFont.FreeTypeFont:
    if size in _font_cache:
        return _font_cache[size]
    for p in _FONT_CANDIDATES:
        try:
            if Path(p).exists():
                f = ImageFont.truetype(p, size)
                _font_cache[size] = f
                return f
        except Exception:
            continue
    f = ImageFont.load_default()
    _font_cache[size] = f
    return f


def _bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def render_report(image: np.ndarray, result: dict, source_name: str = "") -> bytes:
    """渲染检测报告 PNG，返回字节流。

    排版策略：先量出底部汇总文字换行后的实际行数，再据此确定画布高度，
    最后一次性绘制——避免长文本被裁掉。
    """
    boxes = result.get("final_boxes", [])
    pil = _bgr_to_pil(image).convert("RGB")
    W, H = pil.size
    head_h = 64

    # ---------------- 底部汇总 ----------------
    f = result.get("fusion", {})
    r = result.get("routing", {})
    t = result.get("timing", {})
    per = {}
    for b in boxes:
        per[b["cls_name"]] = per.get(b["cls_name"], 0) + 1

    # 先排版，再按实际行数决定汇总条高度（避免文字被裁切）
    pad_x, max_w = 20, W - 40
    body: list[tuple[str, int, tuple[int, int, int]]] = []

    def add(text: str, size: int, color):
        body.append((text, size, color))

    per_txt = "  ".join(f"{config.CLASS_LABELS_ZH.get(k, k)} {v}" for k, v in per.items()) \
        or "未检出缺陷"
    add(f"最终缺陷框：{len(boxes)} 个（{per_txt}）", 17, (15, 23, 42))
    add(f"路由：候选 {r.get('total', 0)} 框 → 高置信直接接受 {r.get('accepted', 0)} 框，"
        f"转大模型复核 {r.get('verified', 0)} 框", 15, (51, 65, 85))
    add(f"协同增益：大模型剔除误检 {f.get('dropped_by_vlm', 0)} 个，"
        f"类别纠错 {f.get('class_corrected', 0)} 个", 15, (51, 65, 85))
    add(f"耗时：YOLO {t.get('yolo_ms', 0)} ms ＋ 复核 {t.get('vlm_ms', 0)} ms"
        f" ＝ {t.get('total_ms', 0)} ms", 15, (51, 65, 85))
    vlm_info = result.get("vlm_info", {})
    add(f"复核后端：{result.get('vlm', {}).get('model', '')}", 13, (100, 116, 139))

    measure = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    lines: list[tuple[str, int, tuple[int, int, int]]] = []
    for text, size, color in body:
        font = _font(size)
        cur = ""
        for ch in text:
            if measure.textlength(cur + ch, font=font) > max_w and cur:
                lines.append((cur, size, color))
                cur = ch
            else:
                cur += ch
        lines.append((cur, size, color))

    line_h = {17: 26, 15: 23, 13: 20}
    note = f"⚠ {vlm_info.get('note', '')}" if not vlm_info.get("is_real") else ""
    if note:
        note_lines = []
        cur = ""
        for ch in note:
            if measure.textlength(cur + ch, font=_font(12)) > max_w and cur:
                note_lines.append(cur)
                cur = ch
            else:
                cur += ch
        note_lines.append(cur)
    else:
        note_lines = []
    foot_h = 16 + sum(line_h.get(s, 22) for _, s, _ in lines) \
        + (6 + 19 * len(note_lines) if note_lines else 8)

    foot_y = H + head_h
    canvas = Image.new("RGB", (W, H + head_h + foot_h), (248, 250, 252))
    canvas.paste(pil, (0, head_h))
    draw = ImageDraw.Draw(canvas)

    # ---------------- 标题栏 ----------------
    draw.rectangle([0, 0, W, head_h], fill=(15, 23, 42))
    draw.text((20, 12), f"{config.SYSTEM_NAME} {config.SYSTEM_VERSION} · 检测报告",
              font=_font(22), fill=(255, 255, 255))
    sub = (f"{source_name}  |  {W}x{H}  |  阈值 T={result.get('params', {}).get('threshold')}"
           f"  |  复核模式 {result.get('vlm', {}).get('mode')}")
    while measure.textlength(sub, font=_font(13)) > max_w and len(sub) > 8:
        sub = sub[:-2]
    draw.text((22, 40), sub, font=_font(13), fill=(148, 163, 184))

    # ---------------- 检测框 ----------------
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        y1 += head_h
        y2 += head_h
        color = CLASS_COLORS.get(b["cls_name"], _ACCENT)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        tag = f"{config.CLASS_LABELS_ZH.get(b['cls_name'], b['cls_name'])} {b['conf']:.2f}"
        if b.get("class_corrected"):
            tag += f" (纠错:{config.CLASS_LABELS_ZH.get(b['yolo_cls_name'], b['yolo_cls_name'])})"
        tw = draw.textlength(tag, font=_font(15))
        ty = max(head_h + 2, y1 - 22)
        draw.rectangle([x1, ty, x1 + tw + 12, ty + 21], fill=color)
        draw.text((x1 + 6, ty + 2), tag, font=_font(15), fill=(255, 255, 255))

    draw.rectangle([0, foot_y, W, foot_y + foot_h], fill=(255, 255, 255))
    draw.line([0, foot_y, W, foot_y], fill=(226, 232, 240), width=2)

    y = foot_y + 12
    for text, size, color in lines:
        draw.text((pad_x, y), text, font=_font(size), fill=color)
        y += line_h.get(size, 22)
    if note_lines:
        y += 6
        for nl in note_lines:
            draw.text((pad_x, y), nl, font=_font(12), fill=(180, 83, 9))
            y += 19

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()
