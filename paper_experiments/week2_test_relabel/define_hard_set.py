# -*- coding: utf-8 -*-
"""
Hard Set 定义（按标注客观属性，不用模型置信度）
================================================
从修正 test 集（relabeled_test）的标注框出发，按"客观难度属性"定义疑难子集：
  1) 框面积小：缺陷占整图面积比例小 → 小目标难检
  2) 对比度低：框内区域灰度标准差低 / 梯度低 → 与背景难区分
输出 hard_set.json（每个难样本框 + 判定依据）+ 分布统计。
原则：**只依赖标注与图像本身属性，不依赖模型置信度**，避免与路由机制循环论证。

用法：E:\Miniconda\envs\label-studio\python.exe define_hard_set.py
"""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DATA = Path(r"E:\CodeBase_YangJunJie\Paper\实验数据\relabeled_test")
OUT = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week2_test_relabel")
AREA_PCT = 0.30      # 面积最小的 30% 视为"小目标"
CONTRAST_PCT = 0.30  # 对比度最低的 30% 视为"低对比"


def imread_unicode(path):
    """cv2.imread 无法处理含中文的路径，用 fromfile+imdecode 替代"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def load_gt(lbl_path):
    boxes = []
    if lbl_path.exists():
        for line in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split()
            if len(p) == 5:
                boxes.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return boxes


def main():
    img_dir = DATA / "images"
    lbl_dir = DATA / "labels"
    recs = []  # {image, cls, area, contrast, w, h}
    for img_path in sorted(img_dir.iterdir()):
        img = imread_unicode(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gts = load_gt(lbl_dir / (img_path.stem + ".txt"))
        for cls, x, y, bw, bh in gts:
            area = bw * bh
            x1 = int((x - bw / 2) * w); y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w); y2 = int((y + bh / 2) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            region = gray[y1:y2, x1:x2]
            contrast = float(region.std()) if region.size > 0 else 0.0
            recs.append({"image": img_path.name, "cls": cls, "area": area,
                         "contrast": contrast})

    name = {0: "hole", 1: "stain", 2: "wrinkle"}
    # 阈值：面积分位 / 对比度分位（仅对有框的样本）
    areas = sorted(r["area"] for r in recs)
    contrasts = sorted(r["contrast"] for r in recs)
    area_thr = areas[int(len(areas) * AREA_PCT)]
    contrast_thr = contrasts[int(len(contrasts) * CONTRAST_PCT)]

    hard = []
    per_cls = defaultdict(int)
    for r in recs:
        small = r["area"] <= area_thr
        low_contrast = r["contrast"] <= contrast_thr
        if small or low_contrast:
            hard.append({**r, "cls_name": name[r["cls"]],
                         "small_area": small, "low_contrast": low_contrast})
            per_cls[name[r["cls"]]] += 1

    print(f"总标注框: {len(recs)}")
    print(f"面积阈值: {area_thr:.4f}（≤ 视为小目标）  对比度阈值: {contrast_thr:.2f}（≤ 视为低对比）")
    print(f"Hard Set 框数: {len(hard)}（占 {len(hard)/len(recs):.1%}） 各类: {dict(per_cls)}")

    out = {"total_boxes": len(recs),
           "area_percentile": AREA_PCT, "area_thr": area_thr,
           "contrast_percentile": CONTRAST_PCT, "contrast_thr": contrast_thr,
           "hard_box_count": len(hard),
           "hard_boxes": hard}
    (OUT / "hard_set.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已保存: {OUT / 'hard_set.json'}")
    print("说明：Hard Set 仅按框面积/对比度客观属性定义，不含模型置信度，避免与路由机制循环论证。")


if __name__ == "__main__":
    main()
