# -*- coding: utf-8 -*-
"""
难样本提取脚本（第一周 · VLM 可行性探针的数据准备）
====================================================
目的：从现有数据集里自动挑出"YOLO 处理不好"的样本，裁剪成局部图，
     供 VLM 复核测试使用（判断 VLM 能否在裁剪图上确认/找回/拒绝瑕疵）。

运行（用 label-studio 环境）：
    set DASHSCOPE_API_KEY=sk-xxxx          # 仅 test_vlm.py 需要
    E:/Miniconda/envs/label-studio/python.exe extract_hard_samples.py --split val

输出：
    <out>/
        hard_samples_meta.csv   # 每个难样本的信息（供 test_vlm.py 使用）
        confidence_stats.json   # 置信度分布 + 各置信度区间的错误率（H1 证据）
        crops/                  # 各类裁剪图（VLM 的输入）
        annotated/              # 少量带标注的整图（人工复核用）

说明：可行性探针阶段用 val 即可（val 标签基本干净）；想凑更多样本可 --split train，
     但最终论文指标必须在新划分的 test 集上重算，探针阶段不要求严格隔离。
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

# ================= 可配置常量 =================
DATASET_ROOT = Path(r"E:\CodeBase_YangJunJie\ClothingInspection\Final_Merged_Dataset_v3")
MODEL_PATH = r"E:\CodeBase_YangJunJie\ClothingInspection\runs\detect\Final_Model_V6\weights\best.pt"
CLASS_NAMES = {0: "hole", 1: "stain", 2: "wrinkle"}

# 判定"低置信度"的上限：低于它视为需要 VLM 复核的候选
CONF_LOW = 0.5
# 推理时的置信度阈值：设低一点，把可能的误检/弱检测都暴露出来
CONF_INFER = 0.10
IOU_MATCH = 0.5
# 裁剪时向四周扩边的比例
CROP_MARGIN = 0.30
MIN_CROP_SIZE = 96
# 每个类型最多保留多少样本（避免过多）
MAX_PER_TYPE = 80
# 正常区域（无瑕疵图）最多抽几张
MAX_NORMAL = 40


def parse_gt_label(txt_path):
    """读取 YOLO 格式标签 -> [(cls, cx, cy, w, h)]（归一化坐标）"""
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) == 5:
            c, x, y, w, h = (float(p) for p in parts)
            boxes.append((int(c), x, y, w, h))
    return boxes


def norm_to_pixel(box, img_w, img_h):
    """归一化 (cx,cy,w,h) -> 像素 (x1,y1,x2,y2)"""
    _, x, y, w, h = box
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return [x1, y1, x2, y2]


def iou(a, b):
    """两个像素框 (x1,y1,x2,y2) 的 IoU"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def crop_with_margin(img, box, margin=CROP_MARGIN, min_size=MIN_CROP_SIZE):
    """按框裁剪并扩边，返回 (crop, out_box 像素坐标)"""
    h, w = img.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    pad_x, pad_y = bw * margin, bh * margin
    x1 = max(0, int(box[0] - pad_x))
    y1 = max(0, int(box[1] - pad_y))
    x2 = min(w, int(box[2] + pad_x))
    y2 = min(h, int(box[3] + pad_y))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        # 目标太小：直接以框中心取 min_size 见方
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = min_size // 2
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
    crop = img[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def main():
    parser = argparse.ArgumentParser(description="提取 YOLO 难样本（低置信度/漏检/误检）")
    parser.add_argument("--split", default="val", choices=["val", "train", "test"],
                        help="用哪个划分提取难样本；可行性探针用 val 即可")
    parser.add_argument("--out", default=r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week1_vlm_probe\hard_samples",
                        help="输出目录")
    parser.add_argument("--device", default="0", help="GPU 编号，cpu 表示用 CPU")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    crops_dir = out_dir / "crops"
    annot_dir = out_dir / "annotated"
    for d in (crops_dir, annot_dir):
        d.mkdir(parents=True, exist_ok=True)

    img_dir = DATASET_ROOT / "images" / args.split
    lbl_dir = DATASET_ROOT / "labels" / args.split
    if not img_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {img_dir}")

    model = YOLO(MODEL_PATH)
    device = args.device if args.device != "cpu" else "cpu"

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)
    print(f"[info] split={args.split} images={len(image_paths)}")

    meta_rows = []
    type_count = {}
    # 置信度统计：conf bin -> [tp, fp]
    conf_bins = {round(b, 1): [0, 0] for b in [i / 10 for i in range(10)]}
    no_gt_images = []  # 无标注的图（用于抽 normal_region）
    n_tp = n_fp = n_fn = 0

    def add_case(case_type, img, gt_px, dets, det2gt, crop_box, gt_cls, det_cls, det_conf):
        nonlocal type_count
        type_count.setdefault(case_type, 0)
        if type_count[case_type] >= MAX_PER_TYPE:
            return
        crop, out_box = crop_with_margin(img, crop_box)
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return
        if crop.shape[0] < MIN_CROP_SIZE or crop.shape[1] < MIN_CROP_SIZE:
            scale = MIN_CROP_SIZE / min(crop.shape[0], crop.shape[1])
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        type_count[case_type] += 1
        cid = f"{case_type}_{type_count[case_type]:03d}"
        crop_path = crops_dir / f"{cid}.jpg"
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        meta_rows.append({
            "case_id": cid, "case_type": case_type,
            "image": Path(img_path).name,
            "gt_cls": CLASS_NAMES.get(gt_cls, "none") if gt_cls is not None else "none",
            "det_cls": CLASS_NAMES.get(det_cls, "none") if det_cls is not None else "none",
            "det_conf": round(det_conf, 4) if det_conf is not None else "",
            "bbox": ";".join(str(int(v)) for v in out_box),
            "crop_path": str(crop_path),
        })
        # 保存少量带标注的整图，便于人工复核
        if type_count[case_type] <= 8:
            vis = img.copy()
            for g in gt_px:
                cv2.rectangle(vis, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 255, 0), 2)
            for i, d in enumerate(dets):
                col = (0, 0, 255) if det2gt[i] < 0 else (255, 0, 0)
                cv2.rectangle(vis, (int(d["box"][0]), int(d["box"][1])),
                              (int(d["box"][2]), int(d["box"][3])), col, 2)
            cv2.imwrite(str(annot_dir / f"{cid}_full.jpg"), vis)

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] 无法读取: {img_path.name}")
            continue
        h, w = img.shape[:2]

        gts = parse_gt_label(lbl_dir / (img_path.stem + ".txt"))
        gt_px = [norm_to_pixel(g, w, h) for g in gts]

        # ---- YOLO 推理（低阈值，暴露所有候选）----
        results = model.predict(source=img, conf=CONF_INFER, iou=0.5,
                                imgsz=640, device=device, verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                dets.append({"box": [x1, y1, x2, y2], "conf": float(box.conf[0]),
                             "cls": int(box.cls[0])})

        # ---- 贪心匹配 det <-> gt ----
        det2gt = [-1] * len(dets)
        gt2det = [-1] * len(gt_px)
        for i, d in enumerate(dets):
            best_j, best_iou = -1, IOU_MATCH
            for j, g in enumerate(gt_px):
                if gt2det[j] >= 0:
                    continue
                v = iou(d["box"], g)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                det2gt[i] = best_j
                gt2det[best_j] = i

        if len(gt_px) == 0:
            no_gt_images.append(str(img_path))
            continue

        # ---- 按类型收集 ----
        for i, d in enumerate(dets):
            conf_bin = round(int(d["conf"] * 10) / 10, 1)
            if det2gt[i] >= 0:
                conf_bins[conf_bin][0] += 1
                n_tp += 1
                gt_cls = gts[det2gt[i]][0]
                if gt_cls == d["cls"]:
                    ct = "low_conf_tp" if d["conf"] < CONF_LOW else "high_conf_tp"
                else:
                    ct = "cls_conflict"  # 框对了但类别判错，VLM 能否纠正类别
                add_case(ct, img, gt_px, dets, det2gt, d["box"], gt_cls, d["cls"], d["conf"])
            else:
                conf_bins[conf_bin][1] += 1
                n_fp += 1
                add_case("false_positive", img, gt_px, dets, det2gt, d["box"],
                         None, d["cls"], d["conf"])
        for j, g in enumerate(gt_px):
            if gt2det[j] < 0:
                n_fn += 1
                add_case("missed_gt", img, gt_px, dets, det2gt, g,
                         gts[j][0], None, None)

    # ---- 正常区域（无瑕疵图里随机抽裁剪）----
    random.shuffle(no_gt_images)
    for img_path in no_gt_images[:MAX_NORMAL]:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        csize = int(min(h, w) * 0.5)
        if csize < 64:
            continue
        x1 = random.randint(0, max(0, w - csize))
        y1 = random.randint(0, max(0, h - csize))
        add_case("normal_region", img, [], [], [], [x1, y1, x1 + csize, y1 + csize],
                 None, None, None)

    # ---- 写 meta CSV ----
    meta_path = out_dir / "hard_samples_meta.csv"
    with open(meta_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()) if meta_rows else
                                ["case_id", "case_type", "image", "gt_cls", "det_cls",
                                 "det_conf", "bbox", "crop_path"])
        writer.writeheader()
        writer.writerows(meta_rows)

    # ---- 写置信度统计（H1 证据：低置信度更容易错）----
    stats = {
        "split": args.split, "conf_infer": CONF_INFER,
        "total_detections": n_tp + n_fp,
        "tp": n_tp, "fp": n_fp, "missed_gt": n_fn,
        "per_class_count": {CLASS_NAMES[c]: type_count.get(ct, 0)
                            for c in CLASS_NAMES for ct in ["low_conf_tp", "missed_gt"]},
        "conf_bins": {},
    }
    for b in sorted(conf_bins):
        tp, fp = conf_bins[b]
        err = fp / (tp + fp) if (tp + fp) else None
        stats["conf_bins"][str(b)] = {"tp": tp, "fp": fp,
                                      "error_rate": round(err, 4) if err is not None else None}
    with open(out_dir / "confidence_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n===== 提取完成 =====")
    print(f"总检测框: {n_tp + n_fp} (TP={n_tp}, FP={n_fp})  漏检GT: {n_fn}")
    for ct in ["low_conf_tp", "high_conf_tp", "cls_conflict", "missed_gt",
               "false_positive", "normal_region"]:
        print(f"  {ct:15s}: {type_count.get(ct, 0)}")
    print(f"\nmeta: {meta_path}")
    print(f"stats: {out_dir / 'confidence_stats.json'}")
    print("下一步: 打开 crops/ 目录人工抽查几类裁剪图，然后跑 test_vlm.py")


if __name__ == "__main__":
    main()
