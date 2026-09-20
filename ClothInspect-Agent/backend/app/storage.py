# -*- coding: utf-8 -*-
"""
文件存储层
==========
统一管理上传原图、局部裁剪图、导出报告的落盘与 URL 生成。
所有文件名使用 uuid，避免中文/重名问题；读写走 Unicode 安全接口。
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np

from . import config
from .core.detector import imwrite_unicode


def _safe_name(prefix: str, ext: str = ".jpg") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}{ext}"


def save_upload(data: bytes, filename: str = "") -> tuple[Path, str]:
    """保存上传的原图，返回 (绝对路径, 文件名)。"""
    ext = Path(filename).suffix.lower() if filename else ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        ext = ".jpg"
    name = _safe_name("img", ext)
    path = config.UPLOAD_DIR / name
    path.write_bytes(data)
    return path, name


def copy_to_uploads(src: Path) -> tuple[Path, str]:
    """把样例图复制进 uploads（保留原文件，避免污染数据集）。"""
    ext = src.suffix.lower() or ".jpg"
    name = _safe_name("sample", ext)
    dst = config.UPLOAD_DIR / name
    dst.write_bytes(src.read_bytes())
    return dst, name


def save_crop(img: np.ndarray, prefix: str) -> str:
    """保存局部裁剪图，返回可供前端访问的相对 URL。"""
    if img is None or img.size == 0:
        return ""
    name = _safe_name(prefix)
    path = config.CROP_DIR / name
    ok = imwrite_unicode(path, img, quality=92)
    return f"/media/crops/{name}" if ok else ""


def save_report(img: np.ndarray, prefix: str = "report") -> str:
    """保存导出的检测报告图，返回相对 URL。"""
    name = _safe_name(prefix, ".png")
    path = config.REPORT_DIR / name
    ok = imwrite_unicode(path, img)
    return f"/media/reports/{name}" if ok else ""


def media_url(kind: str, name: str) -> str:
    return f"/media/{kind}/{name}"
