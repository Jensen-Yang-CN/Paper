# -*- coding: utf-8 -*-
"""
基线评估 · 在修正标签(relabeled_val)上重算 V6 真实指标
======================================================
1) ultralytics model.val() → 每类 mAP50 / mAP50-95（全 271 张）
2) 手动评估 conf=0.4 的 P/R/F1（IoU=0.5，框级匹配）：
   - 全量 relabeled_val（271 张，含泄漏图，供参考）
   - 无泄漏子集 clean_val（109 张，诚实泛化数字，论文用这个）
输出：baseline_results.json

运行：E:\Miniconda\envs\label-studio\python.exe eval_baseline.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week1_vlm_probe")
VAL = BASE / "relabeled_val"
MODEL_PATH = r"E:\CodeBase_YangJunJie\ClothingInspection\runs\detect\Final_Model_V6\weights\best.pt"
CONF = 0.4
IOU_THR = 0.5
CLS_NAMES = ["hole", "stain", "wrinkle"]


def load_gt(lbl_path):
    boxes = []
    if lbl_path.exists():
        for line in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split()
            if len(p) == 5:
                cls, x, y, w, h = float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
                boxes.append((int(cls), x, y, w, h))
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


def eval_split(model, names, val_dir, device, conf=CONF):
    """对一组短名图片做 conf 的框级评估"""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    img_dir = val_dir / "images"
    lbl_dir = val_dir / "labels"
    for name in names:
        # 按短名找实际图片文件（扩展名可能是 jpg/png）
        cand = list(img_dir.glob(name + ".*"))
        if not cand:
            print(f"[warn] 找不到图片 {name}")
            continue
        img = imread_unicode(cand[0])
        if img is None:
            print(f"[warn] 读图失败 {name}")
            continue
        h, w = img.shape[:2]
        gts = load_gt(lbl_dir / (name + ".txt"))
        gt_px = []
        for cls, x, y, bw, bh in gts:
            x1 = (x - bw / 2) * w
            y1 = (y - bh / 2) * h
            x2 = (x + bw / 2) * w
            y2 = (y + bh / 2) * h
            gt_px.append((cls, [x1, y1, x2, y2]))

        res = model.predict(source=img, conf=conf, iou=0.5, imgsz=640, device=device, verbose=False)
        dets = []
        if res and res[0].boxes is not None:
            for b in res[0].boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                dets.append((int(b.cls[0]), [x1, y1, x2, y2]))

        matched_gt = [False] * len(gt_px)
        for cls, box in dets:
            best_j, best_iou = -1, IOU_THR
            for j, (gcls, gbox) in enumerate(gt_px):
                if matched_gt[j]:
                    continue
                v = iou(box, gbox)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0 and gts[best_j][0] == cls:
                matched_gt[best_j] = True
                tp[cls] += 1
            else:
                fp[cls] += 1
        for j, (gcls, _) in enumerate(gt_px):
            if not matched_gt[j]:
                fn[gcls] += 1
    return tp, fp, fn


def summarize(tp, fp, fn):
    out = {}
    for i, name in enumerate(CLS_NAMES):
        t, f, n = tp[i], fp[i], fn[i]
        p = t / (t + f) if (t + f) else 0
        r = t / (t + n) if (t + n) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        out[name] = {"TP": t, "FP": f, "FN": n,
                     "Precision": round(p, 4), "Recall": round(r, 4), "F1": round(f1, 4)}
    t = sum(tp.values()); f = sum(fp.values()); n = sum(fn.values())
    p = t / (t + f) if (t + f) else 0
    r = t / (t + n) if (t + n) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    out["ALL"] = {"TP": t, "FP": f, "FN": n,
                  "Precision": round(p, 4), "Recall": round(r, 4), "F1": round(f1, 4)}
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="val", choices=["val", "test"],
                    help="val=修正 val（271，含 clean 子集）；test=修正 test（300，论文正式集）")
    ap.add_argument("--conf", type=float, default=CONF, help="手动评估的置信度阈值")
    args = ap.parse_args()
    CONF_USE = args.conf

    PAPER_DATA = Path(r"E:\CodeBase_YangJunJie\Paper\实验数据")  # 修正数据集已搬运到 Paper
    if args.mode == "val":
        VAL_DIR = PAPER_DATA / "relabeled_val"
        clean_list = VAL_DIR / "clean_val_list.txt"
    else:
        VAL_DIR = PAPER_DATA / "relabeled_test"
        clean_list = None

    # 写 data.yaml（ultralytics val 用）
    yaml_path = VAL_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {VAL_DIR.as_posix()}\ntrain: images\nval: images\nnc: 3\nnames: {CLS_NAMES}\n",
        encoding="utf-8")

    model = YOLO(MODEL_PATH)
    device = "0"

    # ---- 1) mAP（ultralytics 官方计算）----
    print(f"计算 mAP（{args.mode} 全量）...")
    m = model.val(data=str(yaml_path), imgsz=640, conf=0.001, device=device, verbose=False, plots=False)
    mAP = {"mAP50": round(float(m.box.map50), 4),
           "mAP50-95": round(float(m.box.map), 4),
           "per_class_mAP50": {CLS_NAMES[i]: round(float(v), 4) for i, v in enumerate(m.box.maps)}}
    print("  mAP50:", mAP["mAP50"], " mAP50-95:", mAP["mAP50-95"], " 各类mAP50:", mAP["per_class_mAP50"])

    # ---- 2) 框级 P/R/F1 ----
    all_names = sorted(p.stem for p in (VAL_DIR / "images").glob("*"))
    print(f"手动评估 conf={CONF_USE}（{args.mode} 全量 {len(all_names)} 张）...")
    tp, fp, fn = eval_split(model, all_names, VAL_DIR, device, conf=CONF_USE)
    full = summarize(tp, fp, fn)
    print("  全量:", json.dumps(full, ensure_ascii=False))

    clean = None
    if clean_list is not None and clean_list.exists():
        clean_names = [l.strip() for l in clean_list.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"手动评估 conf={CONF_USE}（无泄漏 {len(clean_names)} 张）...")
        tp, fp, fn = eval_split(model, clean_names, VAL_DIR, device, conf=CONF_USE)
        clean = summarize(tp, fp, fn)
        print("  无泄漏:", json.dumps(clean, ensure_ascii=False))

    result = {
        "mode": args.mode,
        "model": MODEL_PATH,
        "conf": CONF_USE, "iou": IOU_THR, "imgsz": 640,
        "eval_date": "2026-08-29",
        "mAP": mAP,
        "prf_full": full,
        "prf_clean": clean,
        "note": "test 模式为论文正式指标（重标 300 张）；val 模式含无泄漏子集仅供参考",
    }
    out_path = BASE / f"baseline_{args.mode}_results.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
