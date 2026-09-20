# -*- coding: utf-8 -*-
"""
数据整理 · 无泄漏切分
====================
1) 计算 train/val 全量图片 MD5
2) clean_val = val 中未在 train 出现的图片（无泄漏子集，用于诚实基线评估）
3) 内部去重：train 内同 hash 只保留一张（记录重复组数）
4) 分层抽 test：从去重后的 train 池中按类别（hole/stain/wrinkle）分层抽 ~300 张，
   与 val/relabeled_val 无 hash 重叠

输出：
  relabeled_val/clean_val_list.txt   # 短名（v000xx），无泄漏 val 子集
  test/test_images.txt               # 原始文件名（需后续重标注）
  split_stats.json                   # 各类统计
"""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DATASET = Path(r"E:\CodeBase_YangJunJie\ClothingInspection\Final_Merged_Dataset_v3")
RELABELED = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week1_vlm_probe\relabeled_val")
OUT = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week1_vlm_probe")
TEST_SIZE = 300
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
random.seed(2026)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label_classes(lbl_path):
    """返回该图标注中出现的类别集合"""
    cls = set()
    if lbl_path.exists():
        for line in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split()
            if len(p) == 5:
                cls.add(int(float(p[0])))
    return cls


def main():
    img_dir = DATASET / "images"
    lbl_dir = DATASET / "labels"

    # ---- 1) hash ----
    train_imgs = [p for p in (img_dir / "train").iterdir() if p.suffix.lower() in EXTS]
    val_imgs = [p for p in (img_dir / "val").iterdir() if p.suffix.lower() in EXTS]
    train_hash = {md5(p): p.name for p in train_imgs}
    val_hash = {md5(p): p.name for p in val_imgs}

    # ---- 2) clean_val：val 中未出现在 train 的 ----
    leaked = [n for h, n in val_hash.items() if h in train_hash]
    clean_val_orig = [n for h, n in val_hash.items() if h not in train_hash]
    # 短名映射
    mapping = {}
    with open(RELABELED / "name_mapping.csv", encoding="utf-8-sig") as f:
        import csv
        for row in csv.DictReader(f):
            mapping[row["original"]] = row["short"]
    clean_val_short = sorted(mapping[n] for n in clean_val_orig if n in mapping)
    (RELABELED / "clean_val_list.txt").write_text("\n".join(clean_val_short), encoding="utf-8")

    # ---- 3) train 内部去重 ----
    seen = {}
    dup_groups = 0
    unique_train = []
    for p in train_imgs:
        h = md5(p)
        if h in seen:
            dup_groups += 1
            continue
        seen[h] = p.name
        unique_train.append(p)

    # ---- 4) 分层抽 test（从 unique_train 中排除与 val 泄漏的）----
    val_hash_set = set(val_hash.keys())
    pool = [p for p in unique_train if md5(p) not in val_hash_set]
    # 按"含哪些类别标注"分组（原始标注有漏标，仅作分层依据；test 标签后续要重标）
    has_hole, has_stain, has_wrinkle, clean = [], [], [], []
    for p in pool:
        c = label_classes(lbl_dir / "train" / (p.stem + ".txt"))
        if not c:
            clean.append(p.name)
        else:
            if 0 in c:
                has_hole.append(p.name)
            if 1 in c:
                has_stain.append(p.name)
            if 2 in c:
                has_wrinkle.append(p.name)

    # 分层策略：hole 稀有，全部纳入；stain/wrinkle 按比例；干净图（负样本）留一部分
    random.shuffle(has_hole); random.shuffle(has_stain)
    random.shuffle(has_wrinkle); random.shuffle(clean)
    chosen = set(has_hole)                       # hole 全要（12 张）
    n_stain = min(120, len(has_stain))
    n_wrink = min(120, len(has_wrinkle))
    n_clean = TEST_SIZE - len(chosen) - n_stain - n_wrink
    if n_clean < 0:
        n_clean = 0
    chosen |= set(has_stain[:n_stain])
    chosen |= set(has_wrinkle[:n_wrink])
    chosen |= set(clean[:n_clean])
    # 若不足 TEST_SIZE，从剩余池随机补足
    rest = [n for n in (has_stain + has_wrinkle + clean) if n not in chosen]
    random.shuffle(rest)
    for n in rest:
        if len(chosen) >= TEST_SIZE:
            break
        chosen.add(n)

    test_path = OUT / "test"
    test_path.mkdir(parents=True, exist_ok=True)
    (test_path / "test_images.txt").write_text("\n".join(sorted(chosen)), encoding="utf-8")

    # ---- 统计 ----
    stats = {
        "train_total": len(train_imgs), "train_unique": len(unique_train),
        "train_internal_dup_groups": dup_groups,
        "val_total": len(val_imgs), "val_leaked": len(leaked),
        "clean_val": len(clean_val_short),
        "test_pool_unique": len(pool), "test_selected": len(chosen),
    }
    # test 各类别图片数
    test_cls = Counter()
    for n in chosen:
        for c in label_classes(lbl_dir / "train" / (Path(n).stem + ".txt")):
            test_cls["hole" if c == 0 else "stain" if c == 1 else "wrinkle"] += 1
    stats["test_class_coverage"] = dict(test_cls)

    (OUT / "split_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nclean_val 清单: {RELABELED / 'clean_val_list.txt'} ({len(clean_val_short)} 张)")
    print(f"test 清单: {test_path / 'test_images.txt'} ({len(chosen)} 张，标签待重标)")
    print("\n注意：test 集标签来自原始（有漏标/误标），正式使用前需用 VLM 辅助重标")


if __name__ == "__main__":
    main()
