# -*- coding: utf-8 -*-
"""
重标注 · 第三步：生成修正后的 YOLO 标签
========================================
读取 relabel_candidates.json + relabel_results.json（人工确认结果），
生成修正后的标注目录：
  relabeled_val/
    images/   （复制原图）
    labels/   （修正后的 YOLO 格式 txt：cls xc yc w h，归一化）

用法：确认完所有图片后运行：
  E:\Miniconda\envs\label-studio\python.exe gen_relabeled_labels.py

输出统计：每类框数、新增/删除/修改数量、与原始标注的差异概览。
"""
import json
import shutil
from collections import Counter
from pathlib import Path

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week1_vlm_probe")
DATASET_ROOT = Path(r"E:\CodeBase_YangJunJie\ClothingInspection\Final_Merged_Dataset_v3")
CLS_ID = {"hole": 0, "stain": 1, "wrinkle": 2}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="val", choices=["val", "test"],
                    help="val=重标后的 val（默认）；test=重标后的 test（图片在 train 目录）")
    args = ap.parse_args()

    if args.mode == "val":
        CAND = BASE / "relabel" / "relabel_candidates.json"
        RESULTS = BASE / "relabel" / "relabel_results.json"
        SRC_IMG = DATASET_ROOT / "images" / "val"
        SRC_LBL = DATASET_ROOT / "labels" / "val"
        OUT = BASE / "relabeled_val"
    else:
        CAND = BASE / "relabel" / "relabel_test_candidates.json"
        RESULTS = BASE / "relabel" / "relabel_test_results.json"
        SRC_IMG = DATASET_ROOT / "images" / "train"   # test 图目前仍在 train 目录
        SRC_LBL = DATASET_ROOT / "labels" / "train"    # 对比用的原始标签
        OUT = BASE / "relabeled_test"

    if not RESULTS.exists():
        raise SystemExit("[error] %s 不存在，先完成人工确认" % RESULTS)
    cands = json.loads(CAND.read_text(encoding="utf-8"))
    results = {r["image"]: r for r in json.loads(RESULTS.read_text(encoding="utf-8"))}

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)

    # 原始文件名超长（Roboflow，最长 176 字符），输出用短名 + 映射表，避免 MAX_PATH 问题
    mapping = []
    done = 0
    stats = Counter()
    diff = Counter()
    skipped = []
    for idx, d in enumerate(cands):
        img_name = d["image"]
        res = results.get(img_name)
        if res is None:
            skipped.append(img_name)
            continue
        short = f"v{idx + 1:05d}"

        # 汇总最终框：候选(keep=true) + 新增框
        final_boxes = []
        for c in d["candidates"]:
            act = (res.get("actions") or {}).get(c["id"], {})
            keep = act.get("keep", c.get("auto_keep", True) is not False)
            if not keep:
                continue
            cat = act.get("category") or c.get("vlm_category") or c.get("yolo_cls")
            if cat not in CLS_ID:
                continue
            final_boxes.append((CLS_ID[cat], c["box"]))
        for a in res.get("added") or []:
            cat = a.get("category")
            if cat not in CLS_ID:
                continue
            final_boxes.append((CLS_ID[cat], [int(v) for v in a["box"]]))

        # 原始标注（用于差异统计）
        orig = []
        lbl = SRC_LBL / (Path(img_name).stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = line.split()
                if len(p) == 5:
                    orig.append((int(float(p[0])), [float(p[1]), float(p[2]), float(p[3]), float(p[4])]))

        # 写修正标签（YOLO 归一化，短文件名）
        w, h = d["width"], d["height"]
        lines = []
        for cls, box in final_boxes:
            x1, y1, x2, y2 = box
            x1 = max(0, min(w, x1)); y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2)); y2 = max(0, min(h, y2))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (OUT / "labels" / (short + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        shutil.copy(SRC_IMG / img_name, OUT / "images" / (short + Path(img_name).suffix.lower()))
        mapping.append({"short": short, "original": img_name,
                        "suffix": Path(img_name).suffix.lower(),
                        "width": w, "height": h})

        # 统计
        for cls, _ in final_boxes:
            stats["final_" + ["hole", "stain", "wrinkle"][cls]] += 1
        if len(orig) == 0 and len(final_boxes) > 0:
            diff["原无标注->新增框"] += 1
        if len(orig) > 0 and len(final_boxes) == 0:
            diff["原有标注->全部删除"] += 1
        if len(orig) == len(final_boxes) == 0:
            diff["保持无瑕疵"] += 1
        done += 1

    # 写入短名-原名映射表
    import csv
    with open(OUT / "name_mapping.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["short", "original", "suffix", "width", "height"])
        w.writeheader()
        w.writerows(mapping)

    print(f"已生成 {done} 张修正标签；未确认 {len(skipped)} 张：{skipped[:5]}")
    print("类别框数:", dict(stats))
    print("差异概览:", dict(diff))
    print(f"输出目录: {OUT}")
    print(f"映射表: {OUT / 'name_mapping.csv'}")
    print("下一步：把 data.yaml 的 val 指向 relabeled_val，重新评估基线")


if __name__ == "__main__":
    main()
