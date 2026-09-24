# -*- coding: utf-8 -*-
# 按《计算机工程与应用》要求重做论文插图：黑白、宽度<8cm、图字小6号宋体(6.5pt)
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams["font.sans-serif"] = ["SimSun", "宋体", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 6.5
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["text.color"] = "black"
matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["xtick.color"] = "black"
matplotlib.rcParams["ytick.color"] = "black"
# 《计算机工程与应用》要求：横、纵坐标的刻度线置于坐标轴内侧
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["xtick.top"] = False
matplotlib.rcParams["ytick.right"] = False

W = 3.15  # 8cm 宽
OUT = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week5_mvp")
W2 = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week4_experiments")
W1 = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week2_test_relabel")


# ---------- 图1 框架图 ----------
def fig_framework():
    fig, ax = plt.subplots(figsize=(W, 2.9))
    ax.set_xlim(-0.35, 10.35); ax.set_ylim(2.55, 10.2); ax.axis("off")

    def box(x, y, w, h, text):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    fc="white", ec="black", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5, linespacing=1.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=6, lw=0.8, color="black",
                                     shrinkA=0, shrinkB=0))

    box(3.5, 9.35, 3.0, 0.72, "输入图像")
    arrow(5.0, 9.35, 5.0, 9.02)
    box(2.8, 8.10, 4.4, 0.92, "YOLOv11 检测（权重冻结）\n输出类别/边框/置信度")
    arrow(5.0, 8.10, 5.0, 7.55)
    box(2.8, 6.72, 4.4, 0.83, "置信度路由（阈值 T）")
    arrow(2.8, 7.14, 1.80, 7.14)
    arrow(7.2, 7.14, 8.10, 7.14)
    ax.text(2.30, 7.32, "C≥T", ha="center", fontsize=6)
    ax.text(7.65, 7.32, "C<T", ha="center", fontsize=6)
    box(0.05, 6.62, 1.75, 1.04, "直接接受\nYOLO 结果")
    box(8.10, 6.62, 1.85, 1.04, "局部裁剪\n+尺度提示")
    arrow(9.02, 6.62, 9.02, 5.85)
    box(8.10, 4.88, 1.85, 0.97, "多模态大模型\n复核")
    arrow(8.10, 5.36, 7.20, 5.36)
    box(2.8, 4.88, 4.4, 0.97, "结果融合\n（保留/剔除+类别纠错）")
    arrow(5.0, 4.88, 5.0, 4.35)
    box(3.5, 3.58, 3.0, 0.77, "最终缺陷框")
    arrow(0.98, 6.62, 0.98, 5.36)
    arrow(0.98, 5.36, 2.8, 5.36)
    plt.savefig(OUT / "fig1_framework_bw.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close()


# ---------- 图2 置信度-错误率 ----------
def fig_h1():
    data = [
        (0.15, 171, 0.287), (0.25, 90, 0.211), (0.35, 57, 0.298), (0.45, 50, 0.380),
        (0.55, 81, 0.185), (0.65, 165, 0.152), (0.75, 154, 0.123), (0.85, 196, 0.082), (0.95, 70, 0.086)]
    fig, axes = plt.subplots(2, 1, figsize=(W, 3.4))
    xs = [d[0] for d in data]; ns = [d[1] for d in data]; es = [d[2] for d in data]
    axes[0].bar(xs, ns, width=0.08, facecolor="0.75", edgecolor="black", lw=0.6)
    axes[0].set_xlabel("检测置信度", fontsize=6.5)
    axes[0].set_ylabel("检测框数量", fontsize=6.5)
    axes[0].set_xlim(0.05, 1.0)
    axes[0].tick_params(labelsize=6)
    axes[1].bar(xs, es, width=0.08, facecolor="white", edgecolor="black", lw=0.6, hatch="///")
    axes[1].plot(xs, es, "o-", color="black", lw=0.8, ms=2.5)
    axes[1].set_xlabel("检测置信度", fontsize=6.5)
    axes[1].set_ylabel("误检率", fontsize=6.5)
    axes[1].set_xlim(0.05, 1.0); axes[1].set_ylim(0, 0.55)
    axes[1].tick_params(labelsize=6)
    for ax, t in zip(axes, ["(a) 置信度分布", "(b) 置信度与误检率"]):
        ax.text(0.02, 0.88, t, transform=ax.transAxes, fontsize=6.5)
    plt.tight_layout()
    plt.savefig(OUT / "fig2_h1_bw.png", dpi=600, bbox_inches="tight")
    plt.close()


# ---------- 图3 Pareto ----------
def fig_pareto():
    sweep = [(0.12, 0.549), (0.31, 0.562), (0.48, 0.576), (0.75, 0.599), (1.30, 0.623)]
    fig, ax = plt.subplots(figsize=(W, 2.6))
    x = [s[0] for s in sweep]; y = [s[1] for s in sweep]
    ax.plot(x, y, "o-", color="black", lw=0.9, ms=3)
    offs = [(4, 3), (4, 3), (4, 3), (4, 3), (-18, 2)]
    for (xi, yi), t, o in zip(sweep, ["0.3", "0.4", "0.5", "0.6", "0.7"], offs):
        ax.annotate("T=" + t, (xi, yi), textcoords="offset points", xytext=o, fontsize=6,
                    bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xlabel("大模型调用次数 / 幅", fontsize=6.5)
    ax.set_ylabel("F1（类别严格）", fontsize=6.5)
    ax.set_xlim(0.0, 1.45); ax.set_ylim(0.543, 0.630)
    ax.tick_params(labelsize=6)
    ax.grid(ls=":", lw=0.4, color="0.6")
    plt.tight_layout()
    plt.savefig(OUT / "fig3_pareto_bw.png", dpi=600, bbox_inches="tight")
    plt.close()


# ---------- 图4 错误案例（2×2，灰度） ----------
def fig_errors():
    SRC = W1 / "error_cases"
    rows = {r["case_id"]: r for r in csv.DictReader(open(SRC / "error_cases.csv", encoding="utf-8-sig"))}
    pick = ["missed_hole_01.jpg", "missed_stain_01.jpg", "missed_wrinkle_01.jpg", "false_positive_01.jpg"]
    lab = {"missed_hole": "(a) 漏检破洞", "missed_stain": "(b) 漏检污渍",
           "missed_wrinkle": "(c) 漏检褶皱", "false_positive": "(d) 误检"}
    fig, axes = plt.subplots(2, 2, figsize=(W, 3.0))
    axes = axes.ravel()
    for i, name in enumerate(pick):
        p = SRC / name
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR) if p.exists() else None
        ax = axes[i]
        if img is None:
            ax.axis("off"); continue
        g = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
        ax.imshow(g)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5)
        ax.set_title(lab.get(rows.get(name, {}).get("type", ""), ""), fontsize=6.5)
    plt.tight_layout()
    plt.savefig(OUT / "fig4_error_cases_bw.png", dpi=600, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    fig_framework(); print("图1 完成")
    fig_h1(); print("图2 完成")
    fig_pareto(); print("图3 完成")
    fig_errors(); print("图4 完成")
