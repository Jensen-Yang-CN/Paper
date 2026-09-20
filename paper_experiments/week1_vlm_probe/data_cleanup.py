# -*- coding: utf-8 -*-
"""
数据整理 · 孤儿标签清理 + 跨源重复检测
======================================
1) 扫描 Final_Merged_Dataset_v3/images/{train,val} 与 labels/{train,val}：
   - 孤儿标签（无对应图片的 txt）-> 移到 backup 目录（不直接删除）
   - 统计无标签的图片（漏标，仅报告）
2) MD5 查重：train/val 内部及跨集重复图片（含 relabeled_val 的副本）

运行：E:\Miniconda\envs\label-studio\python.exe data_cleanup.py
"""
import hashlib
import shutil
from collections import Counter
from pathlib import Path

DATASET = Path(r"E:\CodeBase_YangJunJie\ClothingInspection\Final_Merged_Dataset_v3")
RELABELED = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week1_vlm_probe\relabeled_val")
BACKUP = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\data_cleanup_backup")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    # ---------- 1) 孤儿标签 ----------
    total_orphan = 0
    for split in ("train", "val"):
        img_dir = DATASET / "images" / split
        lbl_dir = DATASET / "labels" / split
        img_stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in EXTS}
        orphans = []
        if lbl_dir.exists():
            for lbl in lbl_dir.glob("*.txt"):
                if lbl.stem not in img_stems:
                    orphans.append(lbl)
        no_label_imgs = [p for p in img_dir.iterdir()
                         if p.suffix.lower() in EXTS
                         and not (lbl_dir / (p.stem + ".txt")).exists()]
        print(f"[{split}] 图片 {len(img_stems)} 张 | 标签文件 {len(list(lbl_dir.glob('*.txt'))) if lbl_dir.exists() else 0} 个")
        print(f"  -> 孤儿标签(无图): {len(orphans)} 个")
        print(f"  -> 无标签图片(漏标候选): {len(no_label_imgs)} 张")
        if orphans:
            dest = BACKUP / "orphan_labels" / split
            dest.mkdir(parents=True, exist_ok=True)
            for o in orphans:
                shutil.move(str(o), str(dest / o.name))
            total_orphan += len(orphans)
    print(f"\n孤儿标签共移动 {total_orphan} 个到 {BACKUP / 'orphan_labels'}")

    # ---------- 2) 跨源 MD5 查重 ----------
    hash_map = {}  # md5 -> [(split, name)]
    for split in ("train", "val"):
        for p in (DATASET / "images" / split).iterdir():
            if p.suffix.lower() in EXTS:
                hash_map.setdefault(md5(p), []).append(f"{split}/{p.name}")
    # relabeled_val 与原集对比
    rel_map = {}
    for p in (RELABELED / "images").iterdir():
        if p.suffix.lower() in EXTS:
            rel_map.setdefault(md5(p), []).append(p.name)

    dup_internal = {h: v for h, v in hash_map.items() if len(v) > 1}
    dup_cross = []
    for h, v in rel_map.items():
        if h in hash_map:
            dup_cross.append((v, hash_map[h]))

    print(f"\n===== 查重结果 =====")
    print(f"原数据集内部重复（同 hash 多个名字）: {len(dup_internal)} 组")
    for h, v in list(dup_internal.items())[:10]:
        print("   ", v)
    print(f"relabeled_val 与原数据集重复: {len(dup_cross)} 组")
    for v, orig in dup_cross[:10]:
        print("   ", v, "<->", orig)

    # 训练/验证集间泄漏检查：同一图片是否同时出现在 train 和 val
    train_hashes = {md5(p) for p in (DATASET / "images" / "train").iterdir() if p.suffix.lower() in EXTS}
    val_hashes = {md5(p) for p in (DATASET / "images" / "val").iterdir() if p.suffix.lower() in EXTS}
    leak = train_hashes & val_hashes
    print(f"train 与 val 完全相同的图片数（泄漏风险）: {len(leak)}")


if __name__ == "__main__":
    main()
