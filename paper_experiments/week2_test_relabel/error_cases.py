# -*- coding: utf-8 -*-
"""
错误案例收集（重标 test 上，V6）
===============================
对 test 300 张 V6(conf=0.4) 推理，与修正 GT 匹配：
收集 误检(FP，检测无GT) / 漏检(FN，GT无检测) / 类别错判 的典型案例，
保存带框标注的可视化图 + 案例清单 CSV。

用法：E:\Miniconda\envs\label-studio\python.exe error_cases.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week2_test_relabel")
DATA = Path(r"E:\CodeBase_YangJunJie\Paper\实验数据\relabeled_test")
MODEL_PATH = r"E:\CodeBase_YangJunJie\ClothingInspection\runs\detect\Final_Model_V6\weights\best.pt"
OUT = BASE / "error_cases"
CONF = 0.5
IOU_THR = 0.5
NAME = {0: "hole", 1: "stain", 2: "wrinkle"}
MAX_PER_TYPE = 5


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def load_gt(lbl):
    out = []
    if lbl.exists():
        for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split()
            if len(p) == 5:
                out.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_PATH)
    img_dir = DATA / "images"
    lbl_dir = DATA / "labels"
    cases = []
    counts = defaultdict(int)

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
        res = model.predict(source=img, conf=CONF, iou=0.5, imgsz=640, device="0", verbose=False)
        dets = []
        if res and res[0].boxes is not None:
            for b in res[0].boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                dets.append((int(b.cls[0]), [x1, y1, x2, y2], float(b.conf[0])))

        # 匹配（类别无关）
        matched_gt = [False] * len(gt_px)
        matched_det = [False] * len(dets)
        for i, (cls, box, conf) in enumerate(dets):
            best_j, best_v = -1, IOU_THR
            for j, (gcls, gbox) in enumerate(gt_px):
                if matched_gt[j]:
                    continue
                v = iou(box, gbox)
                if v > best_v:
                    best_v, best_j = v, j
            if best_j >= 0:
                matched_gt[best_j] = True
                matched_det[i] = True

        # 漏检：GT 无匹配
        for j, (gcls, gbox) in enumerate(gt_px):
            if not matched_gt[j]:
                t = "missed_" + NAME[gcls]
                if counts[t] < MAX_PER_TYPE:
                    counts[t] += 1
                    vis = img.copy()
                    for gi, (gc, gb) in enumerate(gt_px):
                        cv2.rectangle(vis, (int(gb[0]), int(gb[1])), (int(gb[2]), int(gb[3])),
                                      (0, 255, 0) if gi == j else (0, 200, 0), 2)
                    for k, (dc, db, _) in enumerate(dets):
                        cv2.rectangle(vis, (int(db[0]), int(db[1])), (int(db[2]), int(db[3])),
                                      (255, 0, 0), 1)
                    fn = f"{t}_{counts[t]:02d}.jpg"
                    cv2.imencode(".jpg", vis)[1].tofile(str(OUT / fn))
                    cases.append({"case_id": fn, "type": t, "image": img_path.name,
                                  "gt_cls": NAME[gcls], "note": "YOLO漏检该缺陷"})
        # 误检：detection 无 GT 匹配（且非类别错判）
        for i, (cls, box, conf) in enumerate(dets):
            if matched_det[i]:
                continue
            # 是否重叠某 GT（类别错判）？
            overlap = any(iou(box, gb) >= 0.3 for _, gb in gt_px)
            t = "cls_conflict" if overlap else "false_positive"
            if counts[t] < MAX_PER_TYPE:
                counts[t] += 1
                vis = img.copy()
                for gc, gb in gt_px:
                    cv2.rectangle(vis, (int(gb[0]), int(gb[1])), (int(gb[2]), int(gb[3])), (0, 255, 0), 2)
                cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                fn = f"{t}_{counts[t]:02d}.jpg"
                cv2.imencode(".jpg", vis)[1].tofile(str(OUT / fn))
                cases.append({"case_id": fn, "type": t, "image": img_path.name,
                              "gt_cls": "-", "conf": round(conf, 3),
                              "note": ("YOLO框位置与真实缺陷重叠但类别或定位错" if overlap else "YOLO误检此处无缺陷")})

    with open(OUT / "error_cases.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "type", "image", "gt_cls", "conf", "note"])
        w.writeheader()
        w.writerows(cases)
    print(f"已收集 {len(cases)} 个错误案例 -> {OUT}")
    print("类型分布:", dict(counts))
    print("清单: error_cases.csv")


if __name__ == "__main__":
    main()
