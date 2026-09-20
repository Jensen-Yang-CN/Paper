# -*- coding: utf-8 -*-
"""
YOLOv11 视觉检测模块
====================
职责：加载冻结权重的 YOLOv11 模型，对输入图像做一次前向推理，
输出结构化的检测框列表 {类别, bbox(像素), 置信度}。

设计要点：
* 单例 + 线程锁：模型只加载一次，避免并发请求重复占用显存；
* 延迟加载：首次调用时才载入，缩短服务启动时间；
* 多编码兜底：Windows 下中文路径用 cv2.imdecode 读取，避免 imread 失败。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from .. import config


@dataclass
class Detection:
    """单个检测框。"""
    cls_id: int
    cls_name: str
    conf: float
    box: list[float]          # [x1, y1, x2, y2] 像素坐标
    box_norm: list[float]     # [x1, y1, x2, y2] 归一化坐标（前端绘制用）
    area_ratio: float         # 框面积 / 整图面积

    def to_dict(self) -> dict:
        return asdict(self)


class DetectorError(RuntimeError):
    """检测器不可用（权重缺失 / 依赖缺失 / 推理失败）。"""


def imread_unicode(path: str | Path) -> np.ndarray | None:
    """支持中文路径的图像读取。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: str | Path, img: np.ndarray, quality: int = 92) -> bool:
    """支持中文路径的图像写出。"""
    try:
        ext = Path(path).suffix or ".jpg"
        ok, buf = cv2.imencode(ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


class YoloDetector:
    """YOLOv11 冻结检测器（线程安全单例）。"""

    _instance: "YoloDetector | None" = None
    _lock = threading.Lock()

    def __init__(self, weights: Path):
        from ultralytics import YOLO  # 延迟导入，避免无依赖时模块级崩溃

        self.weights = Path(weights)
        self.model = YOLO(str(self.weights))
        self.device = config.DEVICE
        self._infer_lock = threading.Lock()
        self.warmup_done = False

    # ------------------------------------------------------------ 单例入口
    @classmethod
    def instance(cls) -> "YoloDetector":
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is None:
                if config.YOLO_WEIGHTS is None:
                    raise DetectorError(
                        "未找到 YOLOv11 权重文件。请将 Final_Model_V6_best.pt 放到 "
                        f"{config.BACKEND_DIR / 'assets' / 'weights'}，"
                        "或设置环境变量 YOLO_WEIGHTS 指向权重路径。"
                    )
                try:
                    cls._instance = cls(config.YOLO_WEIGHTS)
                except ImportError as e:
                    raise DetectorError(
                        f"缺少推理依赖（{e}）。请安装 ultralytics 与 torch。"
                    ) from e
        return cls._instance

    @classmethod
    def status(cls) -> dict:
        """不触发加载，报告检测器可用性（供 /api/health 使用）。"""
        return {
            "weights_found": config.YOLO_WEIGHTS is not None,
            "weights_path": str(config.YOLO_WEIGHTS) if config.YOLO_WEIGHTS else None,
            "loaded": cls._instance is not None,
            "device": config.DEVICE,
        }

    # ------------------------------------------------------------ 推理
    def warmup(self) -> None:
        """预热：首次推理包含 CUDA 初始化，放在服务启动时做掉。"""
        if self.warmup_done:
            return
        dummy = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
        self.predict(dummy, conf=config.BASELINE_CONF)
        self.warmup_done = True

    def predict(self, img: np.ndarray, conf: float | None = None) -> tuple[list[Detection], float]:
        """对整图做一次检测。

        返回 (检测框列表, 推理耗时秒)。
        """
        conf = config.DEFAULT_INFER_CONF if conf is None else conf
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        with self._infer_lock:
            results = self.model.predict(
                source=img,
                conf=conf,
                iou=0.5,
                imgsz=config.IMG_SIZE,
                device=self.device,
                verbose=False,
            )
        elapsed = time.perf_counter() - t0

        dets: list[Detection] = []
        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                cls_id = int(b.cls[0])
                bx = max(0.0, min(float(w), x1))
                by = max(0.0, min(float(h), y1))
                bx2 = max(0.0, min(float(w), x2))
                by2 = max(0.0, min(float(h), y2))
                area_ratio = ((bx2 - bx) * (by2 - by)) / float(w * h) if w and h else 0.0
                dets.append(Detection(
                    cls_id=cls_id,
                    cls_name=config.CLASS_NAMES[cls_id]
                    if 0 <= cls_id < len(config.CLASS_NAMES) else str(cls_id),
                    conf=round(float(b.conf[0]), 4),
                    box=[round(bx, 2), round(by, 2), round(bx2, 2), round(by2, 2)],
                    box_norm=[
                        round(bx / w, 5), round(by / h, 5),
                        round(bx2 / w, 5), round(by2 / h, 5),
                    ] if w and h else [0, 0, 0, 0],
                    area_ratio=round(area_ratio, 6),
                ))
        dets.sort(key=lambda d: -d.conf)
        return dets, elapsed
