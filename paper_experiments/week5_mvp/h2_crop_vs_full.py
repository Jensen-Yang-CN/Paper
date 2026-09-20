# -*- coding: utf-8 -*-
# H2 直接证据：同一批检测框，VLM 用"裁剪图"判断 vs 用"整图"判断，谁更准
# 复用 week4 的 VLM 缓存（cache_crop.json / cache_whole.json），不新增 API 调用
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week4_experiments")
OUT = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week5_mvp")
DATA = Path(r"E:\CodeBase_YangJunJie\Paper\实验数据\relabeled_test")
MODEL_PATH = r"E:\CodeBase_YangJunJie\Paper\模型权重\Final_Model_V6_best.pt"
IOU_THR = 0.5
CONF_LOW = 0.25


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
    cc = json.loads((BASE / "cache_crop.json").read_text(encoding="utf-8"))
    cw = json.loads((BASE / "cache_whole.json").read_text(encoding="utf-8"))
    model = YOLO(MODEL_PATH)

    # 统计：对"检测框是否为真实缺陷"这一判断，裁剪图 vs 整图 的准确率
    crop_tp = crop_tn = crop_fp = crop_fn = 0
    full_tp = full_tn = full_fp = full_fn = 0
    n_det = 0

    for ip in sorted((DATA / "images").iterdir()):
        img = imread_unicode(ip)
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_gt(DATA / "labels" / (ip.stem + ".txt"))
        gt = [(c, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
              for c, x, y, bw, bh in gts]
        r = model.predict(source=img, conf=CONF_LOW, iou=0.5, imgsz=640, device="0", verbose=False)
        dets = [(int(b.cls[0]), [float(v) for v in b.xyxy[0].tolist()]) for b in r[0].boxes] \
            if r and r[0].boxes is not None else []
        cur = cc.get(ip.name, {})
        whole = cw.get(ip.name)

        for k, (cls, box) in enumerate(dets):
            # 真值：该检测框是否命中同类别真实缺陷
            is_real = any(gc == cls and iou(box, gb) >= IOU_THR for gc, gb in gt)
            cv_v = cur.get(str(k))
            if cv_v is None:
                continue
            crop_says = bool(cv_v.get("is_defect"))
            full_says = bool(whole.get("is_defect")) if whole else False
            n_det += 1
            if is_real and crop_says:
                crop_tp += 1
            elif is_real and not crop_says:
                crop_fn += 1
            elif (not is_real) and crop_says:
                crop_fp += 1
            else:
                crop_tn += 1
            if is_real and full_says:
                full_tp += 1
            elif is_real and not full_says:
                full_fn += 1
            elif (not is_real) and full_says:
                full_fp += 1
            else:
                full_tn += 1

    def m(tp, fp, fn, tn):
        acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        return {"acc": round(acc, 4), "P": round(p, 4), "R": round(r, 4), "F1": round(f1, 4)}

    crop = m(crop_tp, crop_fp, crop_fn, crop_tn)
    full = m(full_tp, full_fp, full_fn, full_tn)
    print(f"样本：{n_det} 个检测框（判断'该框是否真实缺陷'）")
    print(f"VLM(裁剪图): TP{crop_tp} FP{crop_fp} FN{crop_fn} TN{crop_tn}  ->  {crop}")
    print(f"VLM(整图)  : TP{full_tp} FP{full_fp} FN{full_fn} TN{full_tn}  ->  {full}")

    res = {"n_det": n_det,
           "vlm_on_crop": {"tp": crop_tp, "fp": crop_fp, "fn": crop_fn, "tn": crop_tn, **crop},
           "vlm_on_full_image": {"tp": full_tp, "fp": full_fp, "fn": full_fn, "tn": full_tn, **full}}
    (OUT / "h2_crop_vs_full.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已保存: h2_crop_vs_full.json")


if __name__ == "__main__":
    main()
