# -*- coding: utf-8 -*-
"""
置信度驱动的动态路由模块
========================
论文核心机制（3.3 节 / 4.2 节方案 A 单阈值）：

        C_yolo >= T  ->  accept   直接接受 YOLO 结果，不调用大模型
        C_yolo <  T  ->  verify   送入局部裁剪 + VLM 复核

T 是唯一的决策信号（论文推荐操作点 T=0.6，Pareto 膝点）。
论文的立论基础 H1 已被实测支撑：置信度 <0.5 的平均错误率约 25%，
>=0.6 约 12% 且随置信度单调下降——因此"按置信度路由"才有依据。
"""
from __future__ import annotations

from dataclasses import dataclass

from .detector import Detection


@dataclass
class RouteDecision:
    """单框路由结果。"""
    action: str          # accept | verify
    reason: str          # 人类可读的理由（前端时间线展示）

    @property
    def needs_vlm(self) -> bool:
        return self.action == "verify"


def route(det: Detection, threshold: float) -> RouteDecision:
    """单阈值路由。"""
    if det.conf >= threshold:
        return RouteDecision(
            action="accept",
            reason=f"置信度 {det.conf:.2f} ≥ 阈值 {threshold:.2f}，直接接受 YOLO 判定",
        )
    return RouteDecision(
        action="verify",
        reason=f"置信度 {det.conf:.2f} < 阈值 {threshold:.2f}，属疑难样本，转交多模态大模型复核",
    )


def route_all(dets: list[Detection], threshold: float) -> list[RouteDecision]:
    return [route(d, threshold) for d in dets]


def route_summary(dets: list[Detection], decisions: list[RouteDecision]) -> dict:
    """路由分布统计，用于前端"这次省了多少次大模型调用"的可视化。"""
    n = len(dets)
    n_verify = sum(1 for d in decisions if d.needs_vlm)
    confs = [d.conf for d in dets] or [0.0]
    return {
        "total": n,
        "accepted": n - n_verify,
        "verified": n_verify,
        "vlm_call_ratio": round(n_verify / n, 4) if n else 0.0,
        "conf_min": round(min(confs), 4),
        "conf_max": round(max(confs), 4),
        "conf_mean": round(sum(confs) / len(confs), 4),
    }
