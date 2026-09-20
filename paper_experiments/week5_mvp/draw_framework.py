# -*- coding: utf-8 -*-
# 论文方法框架图（YOLO → 置信度路由 → VLM 复核 → 融合）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week5_mvp"

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")


def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec="#333", lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)


def arrow(x1, y1, x2, y2, text="", style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=16, lw=1.5, color="#333"))
    if text:
        ax.text((x1 + x2) / 2 + 0.12, (y1 + y2) / 2, text, fontsize=9, color="#c0392b")


box(4.3, 7.1, 3.4, 0.7, "Input Image", "#eaf2ff")
arrow(6.0, 7.1, 6.0, 6.5)
box(4.0, 5.6, 4.0, 0.9, "YOLOv11 Detector\n(dets: cls, bbox, conf)", "#dbe9ff")
ax.text(8.3, 6.05, "frozen weights\n(no retraining)", fontsize=8.5, color="#555")
arrow(6.0, 5.6, 6.0, 5.0)
box(4.1, 4.1, 3.8, 0.9, "Confidence-driven Router\n(threshold T)", "#fff3cd")

arrow(4.1, 4.55, 2.2, 4.55, "conf ≥ T")
arrow(7.9, 4.55, 9.8, 4.55, "conf < T")

box(0.3, 4.1, 1.9, 0.9, "Accept directly\n(YOLO result)", "#d5f5e3")
box(9.6, 4.1, 2.1, 0.9, "Crop + scale hint\n→ VLM verify", "#fde2e2")
arrow(10.65, 4.1, 10.65, 3.4)
box(9.3, 2.5, 2.7, 0.9, "VLM verdict\nis_defect / category", "#fde2e2")
arrow(9.3, 2.95, 7.9, 2.95, "")

box(4.1, 2.5, 3.8, 0.9, "Decision Fusion\n(yolo-accept + vlm-correct)", "#e8daef")
arrow(6.0, 2.5, 6.0, 1.9)
box(4.5, 1.1, 3.0, 0.8, "Final Defect Boxes", "#d5f5e3")

# 反馈说明
ax.text(10.6, 4.55, "VLM only called for\nlow-confidence region", fontsize=8.5,
        color="#555", ha="center")
ax.text(6.0, 0.5, "Fusion rule: keep if VLM says defect (category from VLM); drop otherwise",
        ha="center", fontsize=9, color="#333")
plt.tight_layout()
out = BASE + r"\framework.png"
import os
os.makedirs(BASE, exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight")
print("已保存:", out)
