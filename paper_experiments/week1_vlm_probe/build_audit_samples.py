# -*- coding: utf-8 -*-
"""
标注审计 · 抽样清单生成
======================
从已提取的难样本 + VLM 全量结果里，按策略抽样出需要人工目测核实的图片清单，
输出 audit_samples.json 供 audit_server.py 使用。

抽样策略（约 46 张）：
  - false_positive 中 VLM 判"有缺陷"的 12 张   -> 验证 VLM 是否过度报告 / 是否漏标
  - false_positive 中 VLM 判"无缺陷"的 6 张    -> 验证 VLM 是否漏报（对照组）
  - normal_region 中 VLM 判"有缺陷"的 12 张    -> 量化漏标率（关键）
  - normal_region 中 VLM 判"无缺陷"的 3 张     -> 对照组
  - high_conf_tp 中 VLM 判"有缺陷"的 5 张      -> 阳性对照（人应看出缺陷）
  - high_conf_tp 中 VLM 判"无缺陷"的 5 张      -> 验证 VLM 漏报率
  - missed_gt 中 VLM 判"有缺陷"的 3 张         -> 阳性对照（人应看出缺陷）
"""
import csv
import json
import random
from pathlib import Path

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week1_vlm_probe")
META = BASE / "hard_samples" / "hard_samples_meta.csv"
VLM_RESULTS = BASE / "hard_samples" / "vlm_results_v3plus_full.csv"
OUT = BASE / "hard_samples" / "audit_samples.json"

random.seed(2026)

def load_rows():
    meta = {r["case_id"]: r for r in csv.DictReader(open(META, encoding="utf-8-sig"))}
    vlm = {}
    for r in csv.DictReader(open(VLM_RESULTS, encoding="utf-8-sig")):
        pred = None
        try:
            pred = json.loads(r["vlm_result"]) if r["vlm_result"] else None
        except Exception:
            pass
        vlm[r["case_id"]] = {
            "verdict": r["verdict"],
            "category": (pred or {}).get("category"),
            "is_defect": (pred or {}).get("is_defect"),
        }
    return meta, vlm

def pick(meta, vlm, case_type, verdict, n, exclude=None):
    """按 case_type + VLM verdict 抽样 n 张"""
    exclude = exclude or set()
    cands = [cid for cid, r in meta.items()
             if r["case_type"] == case_type
             and cid in vlm and vlm[cid]["verdict"] == verdict
             and cid not in exclude]
    random.shuffle(cands)
    return cands[:n]

def main():
    meta, vlm = load_rows()
    picked = []
    seen = set()

    # 各类抽样
    groups = [
        ("false_positive", "disagree", 12, "fp_flagged"),   # VLM说有，验证是否真漏标/过度报告
        ("false_positive", "agree",    6,  "fp_clean"),     # VLM说无，对照
        ("normal_region",  "disagree", 12, "normal_flagged"),# 无标注图被VLM标出 -> 漏标率
        ("normal_region",  "agree",    3,  "normal_clean"),  # 无标注图VLM也说无
        ("high_conf_tp",   "agree",    5,  "tp_agree"),      # 阳性对照
        ("high_conf_tp",   "disagree", 5,  "tp_disagree"),   # VLM漏报检查
        ("missed_gt",      "agree",    3,  "missed_agree"),  # 阳性对照
    ]
    for case_type, verdict, n, label in groups:
        cids = pick(meta, vlm, case_type, verdict, n, exclude=seen)
        for cid in cids:
            seen.add(cid)
            r = meta[cid]
            v = vlm[cid]
            picked.append({
                "id": cid,
                "group": label,
                "case_type": r["case_type"],
                "img": f"/img/{cid}.jpg",
                "gt_cls": r["gt_cls"],
                "vlm_is_defect": v["is_defect"],
                "vlm_category": v["category"],
            })

    # 打乱顺序，避免连续同类
    random.shuffle(picked)
    for i, p in enumerate(picked):
        p["idx"] = i + 1
    OUT.write_text(json.dumps(picked, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"共抽样 {len(picked)} 张 -> {OUT}")
    from collections import Counter
    print("分组统计:", dict(Counter(p["group"] for p in picked)))

if __name__ == "__main__":
    main()
