# -*- coding: utf-8 -*-
"""Pydantic 请求 / 响应模型。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DetectParams(BaseModel):
    """检测参数（均有默认值，前端"一键推荐值"即默认值）。"""
    threshold: float = Field(0.60, ge=0.05, le=0.99,
                             description="路由阈值 T：置信度 ≥ T 直接接受，< T 转大模型复核")
    infer_conf: float = Field(0.25, ge=0.01, le=0.99,
                              description="YOLO 推理置信度下限")
    baseline_conf: float = Field(0.40, ge=0.01, le=0.99,
                                 description="仅 YOLO 对照基线阈值（论文口径 0.40）")
    enable_vlm: bool = Field(True, description="是否启用多模态大模型复核")
    enable_enhance: bool = Field(False, description="是否对局部区域做对比度增强")
    save_record: bool = Field(True, description="是否写入检测记录（结果历史）")


class HealthResponse(BaseModel):
    status: str
    system: dict
    detector: dict
    vlm: dict
    records: dict


class ApiError(BaseModel):
    detail: str
    hint: Optional[str] = None
    extra: Optional[dict] = None
