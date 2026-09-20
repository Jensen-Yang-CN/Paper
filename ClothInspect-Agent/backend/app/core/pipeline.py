# -*- coding: utf-8 -*-
"""
协同推理流水线（系统核心编排）
==============================
把论文方法串成一次可解释、可复现的推理过程：

    输入图像
      ↓
    ① detect_defect   YOLOv11 视觉检测（冻结权重，conf=推理下限）
      ↓
    ② 置信度路由       C_yolo >= T → 直接接受；C_yolo < T → 转 VLM 复核
      ↓
    ③ crop_region     局部裁剪（扩边 30% + 过小放大）+ 尺度提示
      ↓
    ④ verify_with_vlm 多模态大模型语义复核 → {is_defect, category, confidence, reason}
      ↓
    ⑤ 结果融合        保留/剔除由 VLM 决定，类别以 VLM 为准（类别纠错），框沿用 YOLO
      ↓
    最终缺陷框 + 完整推理轨迹

流水线同时跑一遍"仅 YOLO（论文基线阈值 0.4）"，用于前端并排对比，
让使用者一眼看出协同相对基线的变化。
"""
from __future__ import annotations

import time
import uuid

import numpy as np

from .. import config
from ..storage import save_crop
from . import fusion as fusion_mod
from . import router as router_mod
from .agent_tools import ToolRegistry, build_registry
from .detector import YoloDetector
from .vlm_client import VLMVerdict, build_verifier, verifier_info


class Pipeline:
    """一次检测请求的执行上下文。"""

    def __init__(
        self,
        threshold: float | None = None,
        infer_conf: float | None = None,
        baseline_conf: float | None = None,
        enable_vlm: bool = True,
        enable_enhance: bool = False,
        save_crops: bool = True,
        registry: ToolRegistry | None = None,
        detector: YoloDetector | None = None,
    ) -> None:
        self.threshold = config.DEFAULT_ROUTER_THRESHOLD if threshold is None else float(threshold)
        self.infer_conf = config.DEFAULT_INFER_CONF if infer_conf is None else float(infer_conf)
        self.baseline_conf = config.BASELINE_CONF if baseline_conf is None else float(baseline_conf)
        self.enable_vlm = enable_vlm
        self.enable_enhance = enable_enhance
        self.save_crops = save_crops
        self.registry = registry or build_registry()
        self.detector = detector          # None 时由工具内部取单例
        self.run_id = uuid.uuid4().hex[:16]

    # ------------------------------------------------------------ 主流程
    def run(self, image: np.ndarray) -> dict:
        t_start = time.perf_counter()
        self.registry.reset()
        h, w = image.shape[:2]
        trace = []
        # 各工具所需的运行时上下文（只传递它自己声明的参数）
        detect_runtime: dict = {"image": image}
        if self.detector is not None:
            detect_runtime["detector"] = self.detector
        image_runtime: dict = {"image": image}

        # ---------- ① YOLOv11 视觉检测（系统工作阈值） ----------
        result, call = self.registry.call(
            "detect_defect",
            args={"conf": self.infer_conf, "image_size": config.IMG_SIZE},
            runtime=detect_runtime,
            summary="",
            detail={"n_detections": 0, "role": "working"},
        )
        dets, yolo_ms = result
        call.latency_ms = int(yolo_ms * 1000)
        call.summary = (f"在 conf≥{self.infer_conf:.2f} 下检出 {len(dets)} 个候选缺陷框"
                        f"（耗时 {call.latency_ms} ms）")
        call.detail = {"n_detections": len(dets), "role": "working",
                       "detections": [d.to_dict() for d in dets]}
        trace.append(call)

        # ---------- ①ᐟ 仅 YOLO 基线（论文口径 conf=0.4，用于并排对比） ----------
        if abs(self.baseline_conf - self.infer_conf) > 1e-6:
            bresult, bcall = self.registry.call(
                "detect_defect",
                args={"conf": self.baseline_conf, "image_size": config.IMG_SIZE},
                runtime=detect_runtime,
                summary="",
                detail={"n_detections": 0, "role": "baseline"},
            )
            baseline_dets, base_ms = bresult
            bcall.latency_ms = int(base_ms * 1000)
            bcall.summary = (f"[对照] 仅 YOLO 基线 conf≥{self.baseline_conf:.2f} 检出 "
                             f"{len(baseline_dets)} 个框（耗时 {bcall.latency_ms} ms）")
            bcall.detail = {"n_detections": len(baseline_dets), "role": "baseline",
                            "detections": [d.to_dict() for d in baseline_dets]}
            trace.append(bcall)
        else:
            baseline_dets = list(dets)

        # ---------- ② 置信度路由 ----------
        decisions = router_mod.route_all(dets, self.threshold)
        summary = router_mod.route_summary(dets, decisions)
        trace.append(self.registry.record(
            "route_decision",
            args={"threshold": self.threshold, "n_detections": len(dets)},
            summary=(f"阈值 T={self.threshold:.2f}：{summary['accepted']} 个高置信框直接接受，"
                     f"{summary['verified']} 个疑难框转交多模态大模型复核"),
            detail=summary,
        ))

        # ---------- ③④ 逐框裁剪 + VLM 复核 ----------
        verdicts: dict[int, VLMVerdict] = {}
        crop_records: list[dict] = []
        vlm_ms_total = 0.0
        verifier = build_verifier() if (self.enable_vlm and summary["verified"] > 0) else None

        for i, det in enumerate(dets):
            if not decisions[i].needs_vlm or not self.enable_vlm:
                continue

            crop_res, ccall = self.registry.call(
                "crop_region",
                args={"box": [round(v, 1) for v in det.box],
                      "margin": config.CROP_MARGIN,
                      "enhance": self.enable_enhance},
                runtime=image_runtime,
                summary="",
                detail={},
            )
            ch, cw = crop_res.image.shape[:2]
            ccall.summary = (f"以框 #{i + 1}（{det.cls_name} {det.conf:.2f}）为中心外扩 "
                             f"{int(config.CROP_MARGIN * 100)}% 裁出局部区域 "
                             f"{cw}x{ch}px{'（已放大）' if crop_res.upscaled else ''}，"
                             f"并生成尺度提示")
            ccall.detail = {
                "source_box": det.box, "class": det.cls_name, "conf": det.conf,
                "crop_box": list(crop_res.box), "crop_size": [cw, ch],
                "upscaled": crop_res.upscaled, "scale_prompt": crop_res.scale_prompt,
                "enhanced": self.enable_enhance,
            }
            trace.append(ccall)

            crop_url = save_crop(crop_res.image, f"{self.run_id}_{i:02d}") if self.save_crops else ""

            verdict, vcall = self.registry.call(
                "verify_with_vlm",
                args={"scale_prompt": crop_res.scale_prompt,
                      "crop_shape": [ch, cw],
                      "yolo_cls": det.cls_name, "yolo_conf": det.conf},
                runtime={"verifier": verifier, "crop": crop_res.image, "det": det},
                summary="",
                detail={},
            )
            verdicts[i] = verdict
            vlm_ms_total += verdict.latency_ms

            if verdict.error:
                vcall.status = "error"
                vcall.summary = f"框 #{i + 1} 复核失败（{verdict.error}），按兜底策略保留 YOLO 原判"
            else:
                vcall.summary = (
                    f"框 #{i + 1}（YOLO 判 {det.cls_name}）：大模型判定"
                    f"{'存在' if verdict.is_defect else '不存在'}瑕疵"
                    + (f"，类别 {verdict.category}" if verdict.is_defect else "")
                    + f"（置信度 {verdict.confidence:.2f}，耗时 {verdict.latency_ms} ms）"
                )
            vcall.detail = {
                "verdict": verdict.to_dict(), "mode": verdict.mode,
                "crop_url": crop_url, "source_box": det.box,
                "yolo_cls": det.cls_name, "yolo_conf": det.conf,
            }
            trace.append(vcall)

            crop_records.append({
                "index": i, "url": crop_url, "source_box": det.box,
                "yolo_cls": det.cls_name, "yolo_conf": det.conf,
                "verdict": verdict.to_dict(),
            })

        # ---------- ⑤ 结果融合 ----------
        if self.enable_vlm:
            fused = fusion_mod.fuse(dets, decisions, verdicts)
        else:
            passthrough = [router_mod.RouteDecision("accept", "未启用大模型复核，直接采用 YOLO 结果")
                           for _ in dets]
            fused = fusion_mod.fuse(dets, passthrough, {})

        trace.append(self.registry.record(
            "fuse_results",
            args={"rule": "VLM 判 is_defect 则保留、类别以 VLM 为准；否则剔除；框沿用 YOLO 精确定位"},
            summary=(f"融合完成：保留 {len(fused.boxes)} 个框；"
                     f"大模型剔除误检 {fused.dropped_by_vlm} 个；"
                     f"类别纠错 {fused.class_corrected} 个"
                     + (f"；复核异常 {fused.vlm_errors} 个" if fused.vlm_errors else "")),
            detail={
                "dropped_by_vlm": fused.dropped_by_vlm,
                "class_corrected": fused.class_corrected,
                "vlm_errors": fused.vlm_errors,
                "final_count": len(fused.boxes),
            },
        ))

        total_ms = int((time.perf_counter() - t_start) * 1000)
        vlm_info = verifier_info()

        return {
            "run_id": self.run_id,
            "params": {
                "threshold": self.threshold,
                "infer_conf": self.infer_conf,
                "baseline_conf": self.baseline_conf,
                "enable_vlm": self.enable_vlm,
                "enable_enhance": self.enable_enhance,
                "crop_margin": config.CROP_MARGIN,
                "min_crop": config.MIN_CROP,
                "img_size": config.IMG_SIZE,
            },
            "image": {"width": w, "height": h},
            "detections": [d.to_dict() for d in dets],
            "baseline_detections": [d.to_dict() for d in baseline_dets],
            "routing": summary,
            "decisions": [
                {"index": i, "action": decisions[i].action, "reason": decisions[i].reason,
                 "conf": dets[i].conf, "cls_name": dets[i].cls_name,
                 "box_norm": dets[i].box_norm}
                for i in range(len(dets))
            ],
            "final_boxes": [b.to_dict() for b in fused.boxes],
            "fusion": {
                "dropped_by_vlm": fused.dropped_by_vlm,
                "class_corrected": fused.class_corrected,
                "vlm_errors": fused.vlm_errors,
            },
            "crops": crop_records,
            "vlm": {
                "enabled": self.enable_vlm,
                "calls": len(verdicts),
                "mode": vlm_info["effective_mode"],
                "model": vlm_info["model"],
                "avg_latency_ms": int(vlm_ms_total / len(verdicts)) if verdicts else 0,
                "total_latency_ms": int(vlm_ms_total),
            },
            "timing": {
                "yolo_ms": int(yolo_ms * 1000),
                "vlm_ms": int(vlm_ms_total),
                "total_ms": total_ms,
            },
            "comparison": {
                "yolo_only_conf": self.baseline_conf,
                "yolo_only_count": len(baseline_dets),
                "collab_count": len(fused.boxes),
                "dropped_by_vlm": fused.dropped_by_vlm,
                "class_corrected": fused.class_corrected,
                "kept_by_vlm": sum(1 for b in fused.boxes if b.source == "vlm_kept"),
                "vlm_calls": len(verdicts),
            },
            "trace": [t.to_dict() for t in trace],
            "vlm_info": vlm_info,
            "class_labels": config.CLASS_LABELS_ZH,
            "class_names": config.CLASS_NAMES,
        }
