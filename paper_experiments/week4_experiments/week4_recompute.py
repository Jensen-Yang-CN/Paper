# -*- coding: utf-8 -*-
# 第四周收尾：给四组对比/消融补算"类别无关"与"图像级"，输出完整统一结果表
# 复用 VLM 缓存（cache_crop.json / cache_whole.json），不重复调 API；只需重新跑一次 YOLO 预计算
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week4_experiments")
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


def match(dets, gt_px, mode):
    tp = fp = 0
    mg = [False] * len(gt_px)
    for d in dets:
        cls, box = d[0], d[1]
        bj, bv = -1, IOU_THR
        for j, (gc, gb) in enumerate(gt_px):
            if mg[j]:
                continue
            if mode == "strict" and gc != cls:
                continue
            v = iou(box, gb)
            if v > bv:
                bv, bj = v, j
        if bj >= 0:
            mg[bj] = True; tp += 1
        else:
            fp += 1
    return tp, fp, len(gt_px) - sum(mg)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return round(p, 4), round(r, 4), round(f1, 4)


def main():
    cc = json.loads((BASE / "cache_crop.json").read_text(encoding="utf-8"))
    cw = json.loads((BASE / "cache_whole.json").read_text(encoding="utf-8"))
    model = YOLO(MODEL_PATH)
    img_dir = DATA / "images"; lbl_dir = DATA / "labels"
    imgs = []
    for ip in sorted(img_dir.iterdir()):
        img = imread_unicode(ip)
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_gt(lbl_dir / (ip.stem + ".txt"))
        gt = [(cls, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
              for cls, x, y, bw, bh in gts]
        r0 = model.predict(source=img, conf=CONF0, iou=0.5, imgsz=640, device="0", verbose=False)
        d40 = [(int(b.cls[0]), [float(v) for v in b.xyxy[0].tolist()], float(b.conf[0]))
               for b in r0[0].boxes] if r0 and r0[0].boxes is not None else []
        r = model.predict(source=img, conf=CONF_LOW, iou=0.5, imgsz=640, device="0", verbose=False)
        d25 = [(int(b.cls[0]), [float(v) for v in b.xyxy[0].tolist()], float(b.conf[0]))
               for b in r[0].boxes] if r and r[0].boxes is not None else []
        imgs.append({"gt": gt, "d40": d40, "d25": d25,
                     "cur": cc.get(ip.name, {}), "whole": cw.get(ip.name)})

    def fused(T):
        per = []
        for d in imgs:
            ds = []
            for k, (cls, box, conf) in enumerate(d["d25"]):
                if conf >= T:
                    ds.append((cls, box, conf))
                else:
                    v = d["cur"].get(str(k))
                    if v and v.get("is_defect"):
                        ds.append((VCLS.get(v.get("category"), cls), box, conf))
            per.append(ds)
        return per

    def nocrop():
        return [[(c, b, cf) for c, b, cf in d["d25"]] if d["whole"] and d["whole"].get("is_defect") else [] for d in imgs]

    def agg(per, mode, name):
        tot = {"tp": 0, "fp": 0, "fn": 0}
        itp = ifp = ifn = 0
        for d, ds in zip(imgs, per):
            t, f, n = match(ds, d["gt"], mode)
            tot["tp"] += t; tot["fp"] += f; tot["fn"] += n
            g = bool(d["gt"]); p = bool(ds)
            if g and p: itp += 1
            elif g: ifn += 1
            elif p: ifp += 1
        p, r, f1 = prf(tot["tp"], tot["fp"], tot["fn"])
        ip, ir, if1 = prf(itp, ifp, ifn)
        return {"P": p, "R": r, "F1": f1, "imgF1": if1, "imgTP": itp, "imgFP": ifp, "imgFN": ifn}

    configs = {
        "A_yolo40": [list(d["d40"]) for d in imgs],
        "control_yolo25": [list(d["d25"]) for d in imgs],
        "B_allverify": fused(2.0),
        "C_nocrop": nocrop(),
        "D_ours07": fused(0.7),
        "D_ours05": fused(0.5),
    }
    lines = []
    print(f"{'方案':<16}{'P(strict)':>10}{'R(strict)':>10}{'F1(strict)':>10}{'F1(agnostic)':>14}{'imgF1':>8}")
    res = {}
    names = ["A_yolo40", "control_yolo25", "B_allverify", "C_nocrop", "D_ours07", "D_ours05"]
    for name in names:
        s = agg(configs[name], "strict", name)
        a = agg(configs[name], "agnostic", name)
        res[name] = {"strict": s, "agnostic": a}
        print(f"{name:<16}{s['P']:>10.4f}{s['R']:>10.4f}{s['F1']:>10.4f}{a['F1']:>14.4f}{s['imgF1']:>8.4f}")
    (BASE / "week4_results_full.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已保存: week4_results_full.json")


if __name__ == "__main__":
    main()
