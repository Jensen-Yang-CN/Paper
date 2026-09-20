# -*- coding: utf-8 -*-
"""
V6 模型在 test 集上的验证脚本（独立可运行）
============================================
用法（label-studio 环境）：
    E:\Miniconda\envs\label-studio\python.exe verify_v6_on_test.py

输出：
    1) mAP50 / mAP50-95（ultralytics 官方计算，conf=0.001）
    2) 框级 P/R/F1 @ conf=0.4（IoU=0.5，逐类）
    3) 图像级 P/R/F1（图像有/无缺陷二分类）——论文建议的主指标口径
    结果保存到 verify_v6_test_results.json

默认评估"修正后 test 标签"（relabeled_test，论文正式口径）。
加 --original 可改为评估"原始标签"（test_originals_backup），用于对比
"标签修正效应"（预期 mAP50≈0.95，与修正后 0.39 形成口径分解表）。
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week2_test_relabel")
MODEL_PATH = r"E:\CodeBase_YangJunJie\ClothingInspection\runs\detect\Final_Model_V6\weights\best.pt"
CLS_NAMES = ["hole", "stain", "wrinkle"]
IOU_THR = 0.5
IMGSZ = 640


def load_gt(lbl_path):
    """YOLO 标签 -> [(cls, x, y, w, h)]（归一化）"""
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
    import numpy as np
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", action="store_true",
                    help="评估原始(漏标)标签而非修正标签")
    ap.add_argument("--conf", type=float, default=0.4, help="手动评估阈值")
    args = ap.parse_args()

    if args.original:
        data_dir = BASE / "test_originals_backup"
        label_note = "原始(漏标)标签"
    else:
        data_dir = Path(r"E:\CodeBase_YangJunJie\Paper\实验数据\relabeled_test")  # 修正 test 已搬到 Paper
        label_note = "修正(重标)标签"
    img_dir = data_dir / "images"
    lbl_dir = data_dir / "labels"

    model = YOLO(MODEL_PATH)
    device = "0"

    # ---- 1) mAP ----
    yaml_path = data_dir / "data.yaml"
    yaml_path.write_text(f"path: {data_dir.as_posix()}\ntrain: images\nval: images\nnc: 3\nnames: {CLS_NAMES}\n",
                         encoding="utf-8")
    m = model.val(data=str(yaml_path), imgsz=IMGSZ, conf=0.001, device=device, verbose=False, plots=False)
    mAP = {"mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
           "per_class_mAP50": {CLS_NAMES[i]: round(float(v), 4) for i, v in enumerate(m.box.maps)}}
    print(f"\n===== V6 在 test（{label_note}）上的表现 =====")
    print(f"mAP50 = {mAP['mAP50']}   mAP50-95 = {mAP['mAP50-95']}   各类 mAP50 = {mAP['per_class_mAP50']}")

    # ---- 2) 框级 P/R/F1 + 图像级统计 ----
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    img_tp = img_fp = img_fn = 0
    n_img = 0
    for img_path in sorted(img_dir.iterdir()):
        img = imread_unicode(img_path)
        if img is None:
            continue
        n_img += 1
        h, w = img.shape[:2]
        gts = load_gt(lbl_dir / (img_path.stem + ".txt"))
        gt_px = []
        for cls, x, y, bw, bh in gts:
            gt_px.append((cls, [(x - bw / 2) * w, (y - bh / 2) * h,
                                (x + bw / 2) * w, (y + bh / 2) * h]))
        res = model.predict(source=img, conf=args.conf, iou=0.5, imgsz=IMGSZ,
                            device=device, verbose=False)
        dets = []
        if res and res[0].boxes is not None:
            for b in res[0].boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                dets.append((int(b.cls[0]), [x1, y1, x2, y2]))

        # 图像级
        gt_has = len(gt_px) > 0
        pred_has = len(dets) > 0
        if gt_has and pred_has:
            img_tp += 1
        elif gt_has and not pred_has:
            img_fn += 1
        elif not gt_has and pred_has:
            img_fp += 1

        # 框级匹配
        matched_gt = [False] * len(gt_px)
        for cls, box in dets:
            best_j, best_v = -1, IOU_THR
            for j, (gcls, gbox) in enumerate(gt_px):
                if matched_gt[j]:
                    continue
                v = iou(box, gbox)
                if v > best_v:
                    best_v, best_j = v, j
            if best_j >= 0 and gts[best_j][0] == cls:
                matched_gt[best_j] = True
                tp[cls] += 1
            else:
                fp[cls] += 1
        for j, (gcls, _) in enumerate(gt_px):
            if not matched_gt[j]:
                fn[gcls] += 1

    print(f"\n--- 框级指标 (conf={args.conf}, IoU=0.5) ---")
    print(f"{'类':<10}{'TP':>6}{'FP':>6}{'FN':>6}{'Precision':>12}{'Recall':>10}{'F1':>8}")
    all_tp = all_fp = all_fn = 0
    for i, name in enumerate(CLS_NAMES):
        t, f, n = tp[i], fp[i], fn[i]
        all_tp += t; all_fp += f; all_fn += n
        p = t / (t + f) if (t + f) else 0
        r = t / (t + n) if (t + n) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        print(f"{name:<10}{t:>6}{f:>6}{n:>6}{p:>12.4f}{r:>10.4f}{f1:>8.4f}")
    p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) else 0
    r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    print(f"{'ALL':<10}{all_tp:>6}{all_fp:>6}{all_fn:>6}{p:>12.4f}{r:>10.4f}{f1:>8.4f}")

    print(f"\n--- 图像级指标 (图有无缺陷, n={n_img}) ---")
    ip = img_tp / (img_tp + img_fp) if (img_tp + img_fp) else 0
    ir = img_tp / (img_tp + img_fn) if (img_tp + img_fn) else 0
    if1 = 2 * ip * ir / (ip + ir) if (ip + ir) else 0
    print(f"TP={img_tp} FP={img_fp} FN={img_fn}  Precision={ip:.4f} Recall={ir:.4f} F1={if1:.4f}")

    result = {"data": str(data_dir), "label_note": label_note, "mAP": mAP,
              "box_prf": {"ALL": {"TP": all_tp, "FP": all_fp, "FN": all_fn,
                                  "Precision": round(p, 4), "Recall": round(r, 4), "F1": round(f1, 4)}},
              "image_prf": {"TP": img_tp, "FP": img_fp, "FN": img_fn,
                            "Precision": round(ip, 4), "Recall": round(ir, 4), "F1": round(if1, 4)},
              "conf": args.conf}
    out = BASE / ("verify_v6_test_results.json" if not args.original else "verify_v6_test_original_results.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
