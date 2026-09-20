# -*- coding: utf-8 -*-
"""
实验数据接口（论文结果页）
==========================
把 paper_experiments 里已定版的结果直接读出来给前端展示，
保证"系统展示的数字"与"论文里的数字"来自同一份数据源，不会漂移。
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import config

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

_P = config.PAPER_ASSETS_DIR

# 论文推荐操作点 T=0.6 的定版数值（Research MVP 报告 表2 / 表4）。
# week4_results_full.json 只保存了 T=0.5 与 T=0.7 两次完整跑批，
# 因此 T=0.6 这一行直接取自定版报告，保证与论文数字完全一致。
OURS_T06 = {
    "key": "D_ours06", "label": "Ours（动态路由，T=0.60）",
    "note": "论文最终方法 · Pareto 膝点 · 推荐操作点",
    "P": 0.627, "R": 0.572, "F1": 0.599,
    "F1_agnostic": 0.845, "imgF1": 0.923,
}
OURS_T06_ROW = {k: OURS_T06[k] for k in ("key", "label", "note", "P", "R", "F1",
                                         "F1_agnostic", "imgF1")}


def _load(name: str) -> dict | None:
    p = _P / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


@router.get("/summary")
def summary():
    """论文核心结果汇总：方法定义、四组对比、阈值扫描、消融、边界说明。"""
    w4 = _load("week4_results_full.json") or {}
    h2 = _load("h2_crop_vs_full.json") or {}
    hard = _load("hard_set_experiment.json") or {}
    h1 = _load("h1_confidence_error.json") or {}
    base = _load("baseline_test_results.json") or {}
    collab = _load("collab_results.json") or {}

    def row(key: str, label: str, note: str = "") -> dict | None:
        d = w4.get(key)
        if not d:
            return None
        s = d.get("strict", {})
        a = d.get("agnostic", {})
        return {
            "key": key, "label": label, "note": note,
            "P": s.get("P"), "R": s.get("R"), "F1": s.get("F1"),
            "F1_agnostic": a.get("F1"), "imgF1": s.get("imgF1"),
        }

    comparison = [r for r in [
        row("A_yolo40", "YOLOv11 基线（conf=0.4）", "论文诚实基线"),
        row("control_yolo25", "低阈值控制组（conf=0.25）", "用于证明增益来自 VLM 而非单纯降阈值"),
        row("B_allverify", "YOLO + 裁剪 + VLM（全复核）", "性能上限，但调用成本最高"),
        dict(OURS_T06_ROW),
        row("D_ours05", "Ours 动态路由 T=0.50", "低预算操作点"),
        row("D_ours07", "Ours 动态路由 T=0.70", "高精度操作点"),
    ] if r]

    return {
        "method": {
            "name": "视觉模型与多模态大模型协同推理",
            "one_line": "让 YOLO 负责快速发现，让多模态大模型负责疑难复核，"
                        "让协同路由机制决定什么时候值得调用大模型。",
            "framework_steps": [
                "YOLOv11（冻结权重）以 conf=0.25 推理，输出候选缺陷框",
                "置信度路由：C ≥ T 直接接受；C < T 判为疑难样本",
                "局部裁剪：外扩 30%、过小放大，并注入尺度提示",
                "多模态复核：qwen3-vl-plus 输出 {is_defect, category, confidence, reason}",
                "结果融合：VLM 判为瑕疵则保留且类别以 VLM 为准，否则剔除；框沿用 YOLO",
            ],
            "key_point": "研究的是推理时协同（模型权重冻结 + 系统层裁判），"
                         "不是用 VLM 复训 YOLO。",
        },
        "dataset": {
            "train": 2155, "val": 271, "test": 300,
            "note": "val/test 已完成 VLM+人工重标；train/val 泄漏已修复（MD5 级查重）",
            "classes": ["hole", "stain", "wrinkle"],
        },
        "baseline": base,
        "collab_run": collab,
        "comparison": comparison,
        "pareto": [
            {"T": 0.3, "calls_per_img": 0.12, "F1": 0.549, "marginal": 0.068},
            {"T": 0.4, "calls_per_img": 0.31, "F1": 0.562, "marginal": 0.082},
            {"T": 0.5, "calls_per_img": 0.48, "F1": 0.576, "marginal": 0.085},
            {"T": 0.6, "calls_per_img": 0.75, "F1": 0.599, "marginal": 0.085,
             "recommended": True},
            {"T": 0.7, "calls_per_img": 1.30, "F1": 0.623, "marginal": 0.044},
        ],
        # 消融：显式给出每个组件是否启用，避免前端按 key 猜
        "ablation": [
            dict(row("A_yolo40", "A：仅 YOLO（基线）", "无局部裁剪 / 无 VLM / 无动态路由") or {},
                 crop=False, vlm=False, router=False),
            dict(row("C_nocrop", "C：整图送 VLM 复核", "去掉局部裁剪，验证 Crop 的作用") or {},
                 crop=False, vlm=True, router=False),
            dict(OURS_T06_ROW, label="D：完整方法（动态路由 T=0.60）",
                 note="论文最终方法", crop=True, vlm=True, router=True),
            dict(row("B_allverify", "B：逐框裁剪 + 全复核", "去掉动态路由，作为性能上限参照") or {},
                 crop=True, vlm=True, router=False),
        ],
        "h2_crop_vs_full": h2,
        "hard_set": hard,
        "h1_confidence_error": h1.get("bins", []),
        "hypotheses": [
            {"id": "H1", "text": "YOLO 的低置信度样本更容易误检/漏检，"
                                 "因此是多模态大模型介入的主要价值区域。",
             "status": "已支撑", "evidence": "置信度 <0.5 平均错误率约 25%，≥0.6 约 12%，0.6→0.9 单调下降"},
            {"id": "H2", "text": "以局部缺陷区域作为额外视觉证据，"
                                 "能提升多模态大模型对细粒度微痕的复核能力。",
             "status": "已支撑",
             "evidence": "同一批 810 框，VLM 看裁剪图 vs 整图：准确率 0.643 vs 0.598、F1 0.755 vs 0.732"},
            {"id": "H3", "text": "相比全部样本都调用大模型，置信度动态路由能在保持性能的同时"
                                 "降低调用率与推理开销。",
             "status": "已支撑（含边界）",
             "evidence": "T=0.6 用全复核 28% 的调用拿到其 35% 的 F1 提升；T=0.7 为 48% / 45%"},
        ],
        "boundary": [
            "全复核（F1 0.755）仍是性能上限，路由的价值是「预算内逼近上限」，"
            "不能写成「超过全复核」。",
            "面积/对比度定义的 Hard Set 与类别混淆，「简单集」召回反而更低，"
            "因此本数据集的有效难度轴是类别（褶皱最难，基线召回仅 0.278）。",
            "系统 Demo 中的单张图片结果仅为流程演示，论文结论以重标 test 集"
            "（300 张）上的批量实验为准。",
        ],
    }


@router.get("/report")
def mvp_report_markdown():
    """Research MVP 报告原文（Markdown）。"""
    p = _P / "Research_MVP_report.md"
    if not p.exists():
        raise HTTPException(status_code=404, detail="未找到 Research MVP 报告")
    return {"markdown": p.read_text(encoding="utf-8")}


@router.get("/figure/{name}")
def figure(name: str):
    """返回论文插图（框架图 / Pareto 曲线 / 置信度-错误率图）。"""
    allowed = {"framework.png", "pareto_curve.png", "h1_confidence_error.png"}
    if name not in allowed:
        raise HTTPException(status_code=404, detail="未收录该插图")
    p = _P / name
    if not p.exists():
        raise HTTPException(status_code=404, detail="插图文件缺失")
    return FileResponse(str(p), media_type="image/png")
