# -*- coding: utf-8 -*-
# 生成论文"错误案例"拼图（2行×3列，含子图标注）
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

SRC = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week2_test_relabel\error_cases")
OUT = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week5_mvp")

LABEL = {
    "missed_hole": "(a) YOLO漏检破洞",
    "missed_stain": "(b) YOLO漏检污渍",
    "missed_wrinkle": "(c) YOLO漏检褶皱",
    "false_positive": "(d) YOLO误检（无缺陷）",
    "cls_conflict": "(e) 类别冲突",
}
PICK = ["missed_hole_01.jpg", "missed_stain_01.jpg", "missed_wrinkle_01.jpg",
        "false_positive_01.jpg", "cls_conflict_01.jpg"]

rows = {r["case_id"]: r for r in csv.DictReader(open(SRC / "error_cases.csv", encoding="utf-8-sig"))}


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()
for ax in axes:
    ax.axis("off")

for i, name in enumerate(PICK):
    p = SRC / name
    if not p.exists():
        continue
    img = imread_u(p)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[i].imshow(img)
    t = rows.get(name, {}).get("type", "")
    axes[i].set_title(LABEL.get(t, t), fontsize=12)

axes[5].axis("off")
axes[5].text(0.02, 0.95, "绿框=真实缺陷(GT)  红框=YOLO检测", fontsize=11, va="top")
axes[5].text(0.02, 0.80, "协同机制据此对低置信度框做保留/剔除与类别纠错",
             fontsize=11, va="top")
axes[5].text(0.02, 0.62, "失败情形：极端模糊、标注边界不清的样本仍会误判",
             fontsize=11, va="top")
plt.tight_layout()
out = OUT / "error_cases_grid.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print("已保存:", out)
