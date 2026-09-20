# -*- coding: utf-8 -*-
"""
H1 证据图：置信度分布 + 置信度-错误率曲线（重标 test 上）
========================================================
对 test 300 张跑 V6（conf=0.1），把所有检测框按置信度分箱，
统计每箱的"错误率"（未匹配到同类别 GT 的检测占比），输出：
  h1_confidence_error.png   （论文用图：双面板：置信度直方图 + 错误率曲线）
  h1_confidence_error.json  （原始数据，可自绘）

用法：
    E:\Miniconda\envs\label-studio\python.exe h1_confidence_error.py
"""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week2_test_relabel")
DATA = Path(r"E:\CodeBase_YangJunJie\Paper\实验数据\relabeled_test")  # 修正 test 已搬到 Paper
MODEL_PATH = r"E:\CodeBase_YangJunJie\ClothingInspection\runs\detect\Final_Model_V6\weights\best.pt"
CONF_INFER = 0.10
IOU_THR = 0.5
BINS = [i / 10 for i in range(11)]  # 0,0.1,...,1.0
CLS_NAMES = ["hole", "stain", "wrinkle"]


def load_gt(lbl_path):
    boxes = []
    if lbl_path.exists():
        for line in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split()
            if len(p) == 5:
                boxes.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return boxes


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def imread_unicode(path):
    """cv2.imread 无法处理含中文的路径，用 fromfile+imdecode 替代"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    model = YOLO(MODEL_PATH)
    device = "0"
    img_dir = DATA / "images"
    lbl_dir = DATA / "labels"

    dets_all = []  # (conf, correct)
    for img_path in sorted(img_dir.iterdir()):
        img = imread_unicode(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_gt(lbl_dir / (img_path.stem + ".txt"))
        gt_px = []
        for cls, x, y, bw, bh in gts:
            gt_px.append((cls, [(x - bw / 2) * w, (y - bh / 2) * h,
                                (x + bw / 2) * w, (y + bh / 2) * h]))
        res = model.predict(source=img, conf=CONF_INFER, iou=0.5, imgsz=640,
                            device=device, verbose=False)
        if not res or res[0].boxes is None:
            continue
        matched_gt = [False] * len(gt_px)
        for b in res[0].boxes:
            conf = float(b.conf[0]); cls = int(b.cls[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            best_j, best_v = -1, IOU_THR
            for j, (gcls, gbox) in enumerate(gt_px):
                if matched_gt[j]:
                    continue
                v = iou([x1, y1, x2, y2], gbox)
                if v > best_v:
                    best_v, best_j = v, j
            # 用"是否命中任一真实缺陷位置"判定（对应某个 GT 框即算正确，类别无关）
            # H1 关注误检/漏检 → 检测框是否对应真实缺陷，而非类别是否完全一致
            correct = (best_j >= 0)
            if correct:
                matched_gt[best_j] = True
            dets_all.append((conf, correct))

    # ---- 分箱 ----
    bin_n = [0] * 10
    bin_err = [0] * 10
    bin_ok = [0] * 10
    for conf, correct in dets_all:
        bi = min(9, int(conf * 10))
        bin_n[bi] += 1
        if correct:
            bin_ok[bi] += 1
        else:
            bin_err[bi] += 1

    centers = [b + 0.05 for b in range(10)]
    err_rate = [bin_err[i] / bin_n[i] if bin_n[i] else 0 for i in range(10)]
    print("分箱结果 (置信度区间: 检测数 / 错误率):")
    for i in range(10):
        print(f"  [{BINS[i]:.1f}-{BINS[i+1]:.1f})  n={bin_n[i]:4d}  err={err_rate[i]:.3f}")

    # ---- 制图 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # 左：置信度分布直方图
    confs = [c for c, _ in dets_all]
    axes[0].hist(confs, bins=BINS, color="#4c8bf5", edgecolor="white")
    axes[0].set_xlabel("Detection Confidence")
    axes[0].set_ylabel("Number of Detections")
    axes[0].set_title("(a) Confidence Distribution\n(test set, V6)")
    axes[0].grid(alpha=0.3, axis="y")
    # 右：错误率 vs 置信度
    axes[1].bar(centers, err_rate, width=0.08, color="#e74c3c", alpha=0.85)
    axes[1].set_xlabel("Detection Confidence (bin center)")
    axes[1].set_ylabel("Error Rate (wrong / total)")
    axes[1].set_title("(b) Error Rate vs Confidence\n(H1: low-confidence detections are more error-prone)")
    axes[1].set_ylim(0, 1.0)
    for i in range(10):
        if bin_n[i]:
            axes[1].text(centers[i], err_rate[i] + 0.03, f"{err_rate[i]:.0%}", ha="center", fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out_png = BASE / "h1_confidence_error.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    result = {"conf_infer": CONF_INFER, "total_detections": len(dets_all),
              "bins": [{"lo": BINS[i], "hi": BINS[i + 1], "n": bin_n[i],
                        "correct": bin_ok[i], "wrong": bin_err[i],
                        "error_rate": round(err_rate[i], 4)} for i in range(10)]}
    out_json = BASE / "h1_confidence_error.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n图: {out_png}")
    print(f"数据: {out_json}")


if __name__ == "__main__":
    main()
