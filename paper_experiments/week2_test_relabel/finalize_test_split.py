# -*- coding: utf-8 -*-
"""
数据结构收尾：把 test 图从 train 物理移出 + 生成最终 data.yaml
=============================================================
1) 将 test/test_images.txt 里的 300 张图（及其原始标签）从
   Final_Merged_Dataset_v3/images|labels/train 移到 backup（防止反向泄漏）
2) 生成论文用最终 data.yaml：
   train: Final_Merged_Dataset_v3/images/train（移出后 2155 张）
   val:   relabeled_val/images（271 张，修正标签）
   test:  relabeled_test/images（300 张，修正标签）
"""
import json
import shutil
from pathlib import Path

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week2_test_relabel")
DATASET = Path(r"E:\CodeBase_YangJunJie\ClothingInspection\Final_Merged_Dataset_v3")
TEST_LIST = BASE / "test_images.txt"
BACKUP = BASE / "test_originals_backup"
OUT_YAML = BASE / "final_data.yaml"

names = [l.strip() for l in TEST_LIST.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"待移出: {len(names)} 张")

moved_img = moved_lbl = 0
for n in names:
    src_img = DATASET / "images" / "train" / n
    src_lbl = DATASET / "labels" / "train" / (Path(n).stem + ".txt")
    if src_img.exists():
        dst = BACKUP / "images" / n
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_img), str(dst))
        moved_img += 1
    if src_lbl.exists():
        dst = BACKUP / "labels" / (Path(n).stem + ".txt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_lbl), str(dst))
        moved_lbl += 1

train_left = len([p for p in (DATASET / "images" / "train").iterdir()])
print(f"移出图片 {moved_img} 张, 标签 {moved_lbl} 个")
print(f"train 剩余图片: {train_left} 张")

# 最终 data.yaml
yaml_text = f"""# 论文最终数据集划分（2026-08-29 定版）
# train: 原始 train（移出 300 张 test 后）；标注为原始（有漏标，仅作训练历史）
# val:   重标 val（271 张，修正标签，短名 v00001-v00271）
# test:  重标 test（300 张，修正标签，短名 v00001-v00300）
path: {BASE.as_posix()}
train: {DATASET.as_posix()}/images/train
val: {BASE.as_posix()}/relabeled_val/images
test: {BASE.as_posix()}/relabeled_test/images
nc: 3
names:
  0: hole
  1: stain
  2: wrinkle
"""
OUT_YAML.write_text(yaml_text, encoding="utf-8")
print(f"data.yaml 已写入: {OUT_YAML}")
