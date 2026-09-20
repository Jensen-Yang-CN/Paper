# -*- coding: utf-8 -*-
# Hard Set 实验：只在"难样本"上比较各方法的召回/F1（复用 week4 缓存）
# 难样本定义来自 hard_set.json 的面积阈值/对比度阈值（客观属性，不含模型置信度）
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

W4 = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week4_experiments")
OUT = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week5_mvp")
DATA = Path(r"E:\CodeBase_YangJunJie\Paper\实验数据\relabeled_test")
MODEL_PATH = r"E:\CodeBase_YangJunJie\Paper\模型权重\Final_Model_V6_best.pt"
IOU_THR = 0.5
CONF0, CONF_LOW = 0.40, 0.25
VCLS = {"hole": 0, "stain": 1, "wrinkle": 2}


def imread_unicode(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def load_gt(lbl):
    b = []
    if lbl.exists():
        for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split()
            if len(p) == 5:
                b.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return b


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def main():
    hs = json.loads((OUT.parent / "week2_test_relabel" / "hard_set.json").read_text(encoding="utf-8"))
    area_thr, contrast_thr = hs["area_thr"], hs["contrast_thr"]
    cc = json.loads((W4 / "cache_crop.json").read_text(encoding="utf-8"))
    model = YOLO(MODEL_PATH)

    # 每张图：GT(带 hard 标记) + 各方法检测
    data = []
    for ip in sorted((DATA / "images").iterdir()):
        img = imread_unicode(ip)
        if img is None:
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gts = load_gt(DATA / "labels" / (ip.stem + ".txt"))
        gt = []
        for cls, x, y, bw, bh in gts:
            x1 = max(0, int((x - bw / 2) * w)); y1 = max(0, int((y - bh / 2) * h))
            x2 = min(w, int((x + bw / 2) * w)); y2 = min(h, int((y + bh / 2) * h))
            reg = gray[y1:y2, x1:x2]
            contrast = float(reg.std()) if reg.size else 0.0
            hard = (bw * bh <= area_thr) or (contrast <= contrast_thr)
            gt.append({"cls": cls, "box": [x1, y1, x2, y2], "hard": hard})
        r0 = model.predict(source=img, conf=CONF0, iou=0.5, imgsz=640, device="0", verbose=False)
        d40 = [(int(b.cls[0]), [float(v) for v in b.xyxy[0].tolist()]) for b in r0[0].boxes] if r0[0].boxes is not None else []
        r = model.predict(source=img, conf=CONF_LOW, iou=0.5, imgsz=640, device="0", verbose=False)
        d25 = [(int(b.cls[0]), [float(v) for v in b.xyxy[0].tolist()]) for b in r[0].boxes] if r[0].boxes is not None else []
        cur = cc.get(ip.name, {})
        # Ours(T=0.6) 融合结果（类别以 VLM 为准）
        ours = []
        r25 = model.predict(source=img, conf=CONF_LOW, iou=0.5, imgsz=640, device="0", verbose=False)
        dets25c = [(int(b.cls[0]), [float(v) for v in b.xyxy[0].tolist()], float(b.conf[0])) for b in r25[0].boxes] if r25[0].boxes is not None else []
        for k, (cls, box, conf) in enumerate(dets25c):
            if conf >= 0.6:
                ours.append((cls, box))
            else:
                v = cur.get(str(k))
                if v and v.get("is_defect"):
                    ours.append((VCLS.get(v.get("category"), cls), box))
        data.append({"gt": gt, "d40": d40, "d25": d25, "ours": ours})

    def hard_eval(dets_key):
        h_tp = h_tot = 0
        e_tp = e_tot = 0
        cls_tp = {0: 0, 1: 0, 2: 0}
        cls_tot = {0: 0, 1: 0, 2: 0}
        for d in data:
            gt = d["gt"]
            dets = d[gt_key_map[dets_key]]
            matched = [False] * len(gt)
            for cls, box in dets:
                bj, bv = -1, IOU_THR
                for j, g in enumerate(gt):
                    if matched[j]:
                        continue
                    if g["cls"] != cls:
                        continue
                    v = iou(box, g["box"])
                    if v > bv:
                        bv, bj = v, j
                if bj >= 0:
                    matched[bj] = True
            for j, g in enumerate(gt):
                cls_tot[g["cls"]] += 1
                if matched[j]:
                    cls_tp[g["cls"]] += 1
                if g["hard"]:
                    h_tot += 1
                    if matched[j]:
                        h_tp += 1
                else:
                    e_tot += 1
                    if matched[j]:
                        e_tp += 1
        hr = h_tp / h_tot if h_tot else 0
        er = e_tp / e_tot if e_tot else 0
        per_cls = {n: round(cls_tp[i] / cls_tot[i], 4) if cls_tot[i] else 0
                   for i, n in enumerate(["hole", "stain", "wrinkle"])}
        return hr, h_tot, er, e_tot, per_cls, cls_tot

    gt_key_map = {"yolo40": "d40", "yolo25": "d25", "ours": "ours"}
    print(f"Hard Set 阈值：面积≤{area_thr:.4f} 或 对比度≤{contrast_thr:.2f}")
    res = {}
    for key, label in [("yolo40", "YOLO@0.4"), ("yolo25", "YOLO@0.25(控制)"), ("ours", "Ours(T=0.6)")]:
        hr, htot, er, etot, per_cls, cls_tot = hard_eval(key)
        res[key] = {"hard_recall": round(hr, 4), "hard_boxes": htot,
                    "easy_recall": round(er, 4), "easy_boxes": etot,
                    "per_class_recall": per_cls, "class_boxes": cls_tot}
        print(f"  {label:<18} Hard={hr:.4f}({htot})  Easy={er:.4f}({etot})  各类召回={per_cls}")
    (OUT / "hard_set_experiment.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已保存: hard_set_experiment.json")


if __name__ == "__main__":
    main()
