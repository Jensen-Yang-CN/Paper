# -*- coding: utf-8 -*-
"""
结果融合模块
============
论文 3.6 节 / Research MVP 明确规定的规则融合（第一版不上学习型融合器）：

    高置信框（conf >= T）  ->  直接保留 YOLO 结果（类别也用 YOLO 的）
    低置信框（conf <  T）  ->  由 VLM 的 is_defect 决定去留：
                                 is_defect = true  -> 保留该框，
                                   且**类别以 VLM 为准**（类别纠错）
                                 is_defect = false -> 剔除该框（VLM 剔除误检）
                                 VLM 调用失败      -> 保守保留 YOLO 原判（可配置）

关键点：框的**精确定位始终沿用 YOLO**（VLM 定位粗、不可靠），
融合只改变"这个框要不要留"和"它属于哪一类"。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

from .detector import Detection
from .router import RouteDecision
from .vlm_client import VLMVerdict

# VLM 调用失败时的兜底策略
KEEP_ON_VLM_ERROR = True


@dataclass
class FusedBox:
    """融合后的最终缺陷框（含完整决策溯源，供前端逐框解释）。"""
    cls_id: int
    cls_name: str
    conf: float
    box: list[float]
    box_norm: list[float]
    area_ratio: float
    source: str                  # yolo_accepted | vlm_kept | yolo_fallback
    routed: str                  # accept | verify
    route_reason: str
    yolo_cls_name: str
    vlm: dict | None = None
    class_corrected: bool = False   # VLM 是否纠正了类别

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FusionResult:
    boxes: list[FusedBox] = field(default_factory=list)
    # 本次融合相对"仅 YOLO 低阈值输出"的变化，用于直观展示协同收益
    dropped_by_vlm: int = 0
    class_corrected: int = 0
    vlm_errors: int = 0

    def to_dict(self) -> dict:
        return {
            "boxes": [b.to_dict() for b in self.boxes],
            "dropped_by_vlm": self.dropped_by_vlm,
            "class_corrected": self.class_corrected,
            "vlm_errors": self.vlm_errors,
        }


def fuse(
    dets: list[Detection],
    decisions: list[RouteDecision],
    verdicts: dict[int, VLMVerdict],
) -> FusionResult:
    """按论文规则融合。

    参数
    ----
    dets      : YOLO 低阈值推理得到的全部候选框
    decisions : 与之等长的路由决策
    verdicts  : {框序号: VLMVerdict}，仅低置信框会有条目
    """
    res = FusionResult()
    for i, det in enumerate(dets):
        dec = decisions[i]
        if not dec.needs_vlm:
            res.boxes.append(FusedBox(
                cls_id=det.cls_id, cls_name=det.cls_name, conf=det.conf,
                box=det.box, box_norm=det.box_norm, area_ratio=det.area_ratio,
                source="yolo_accepted", routed="accept",
                route_reason=dec.reason, yolo_cls_name=det.cls_name,
            ))
            continue

        v = verdicts.get(i)
        if v is None or v.is_defect is None:
            # VLM 无有效结论：按兜底策略处理
            res.vlm_errors += 1
            if KEEP_ON_VLM_ERROR:
                res.boxes.append(FusedBox(
                    cls_id=det.cls_id, cls_name=det.cls_name, conf=det.conf,
                    box=det.box, box_norm=det.box_norm, area_ratio=det.area_ratio,
                    source="yolo_fallback", routed="verify",
                    route_reason=dec.reason, yolo_cls_name=det.cls_name,
                    vlm=v.to_dict() if v else None,
                ))
            continue

        if not v.is_defect:
            res.dropped_by_vlm += 1
            continue

        # VLM 确认为瑕疵 -> 保留，类别以 VLM 为准（类别纠错）
        final_name = v.category if v.category in ("hole", "stain", "wrinkle") else det.cls_name
        corrected = final_name != det.cls_name
        if corrected:
            res.class_corrected += 1
        try:
            final_id = ["hole", "stain", "wrinkle"].index(final_name)
        except ValueError:
            final_id = det.cls_id
        res.boxes.append(FusedBox(
            cls_id=final_id, cls_name=final_name, conf=det.conf,
            box=det.box, box_norm=det.box_norm, area_ratio=det.area_ratio,
            source="vlm_kept", routed="verify",
            route_reason=dec.reason, yolo_cls_name=det.cls_name,
            vlm=v.to_dict(), class_corrected=corrected,
        ))
    return res
