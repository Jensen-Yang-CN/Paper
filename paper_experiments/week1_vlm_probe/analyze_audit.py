# -*- coding: utf-8 -*-
"""标注审计结果分析"""
import json
from collections import defaultdict

BASE = r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week1_vlm_probe\hard_samples"
samples = json.load(open(BASE + r"\audit_samples.json", encoding="utf-8"))
answers = json.load(open(BASE + r"\audit_results.json", encoding="utf-8"))
ans = {a["id"]: a["answer"] for a in answers}
DEF = ("hole", "stain", "wrinkle")


def is_def(a):
    return a in DEF


print("=" * 76)
print("一、按抽样分组统计（用户目测 vs 标注 vs 千问）")
print("=" * 76)
stats = defaultdict(lambda: {"n": 0, "user_def": 0, "user_clean": 0, "user_unsure": 0,
                             "vlm_def": 0, "gt_none": 0, "gt_any": 0})
for s in samples:
    a = ans.get(s["id"], "MISSING")
    g = stats[s["group"]]
    g["n"] += 1
    if is_def(a):
        g["user_def"] += 1
    elif a == "clean":
        g["user_clean"] += 1
    else:
        g["user_unsure"] += 1
    if s["vlm_is_defect"]:
        g["vlm_def"] += 1
    if s["gt_cls"] == "none":
        g["gt_none"] += 1
    else:
        g["gt_any"] += 1

names = {
    "normal_flagged": "无标注图,千问说有",
    "normal_clean": "无标注图,千问说无",
    "fp_flagged": "YOLO误检,千问说有",
    "fp_clean": "YOLO误检,千问说无",
    "tp_agree": "已标瑕疵,千问说对",
    "tp_disagree": "已标瑕疵,千问说无",
    "missed_agree": "YOLO漏检,千问说对",
}
order = ["normal_flagged", "normal_clean", "fp_flagged", "fp_clean",
         "tp_agree", "tp_disagree", "missed_agree"]
for g in order:
    v = stats[g]
    print("%-22s n=%2d | 你看有:%2d 无:%2d 不确定:%2d | 标注有:%2d 无:%2d | 千问说有:%2d" % (
        names[g], v["n"], v["user_def"], v["user_clean"], v["user_unsure"],
        v["gt_any"], v["gt_none"], v["vlm_def"]))

print()
print("=" * 76)
print("二、关键指标")
print("=" * 76)
nf, nc = stats["normal_flagged"], stats["normal_clean"]
nog = nf["n"] + nc["n"]
nog_def = nf["user_def"] + nc["user_def"]
print("1) 漏标率估计：无标注图中你看出的瑕疵占比 = %d/%d = %.0f%%" % (nog_def, nog, nog_def / nog))
print("   （无标注图样本是随机裁剪，整图级漏标率只会更高）")

rows = [(s, ans.get(s["id"], "MISSING")) for s in samples]
vlm_def_total = sum(1 for s, a in rows if s["vlm_is_defect"])
vlm_def_agree = sum(1 for s, a in rows if s["vlm_is_defect"] and is_def(a))
print("2) 千问'说有瑕疵'的准确率：%d/%d = %.0f%%" % (vlm_def_agree, vlm_def_total, vlm_def_agree / vlm_def_total))

vlm_clean_total = sum(1 for s, a in rows if not s["vlm_is_defect"])
vlm_clean_user_def = sum(1 for s, a in rows if not s["vlm_is_defect"] and is_def(a))
print("3) 千问'说无瑕疵'但你看出的：%d/%d = %.0f%%" % (vlm_clean_user_def, vlm_clean_total,
                                                   vlm_clean_user_def / vlm_clean_total if vlm_clean_total else 0))

tp = stats["tp_agree"]
td = stats["tp_disagree"]
print("4) 阳性对照：千问说对的 %d 张你确认有 %d；千问说无的 %d 张你确认有 %d" % (
    tp["n"], tp["user_def"], td["n"], td["user_def"]))

print()
print("=" * 76)
print("三、逐条不一致明细（需要人工复查的样本）")
print("=" * 76)
for s, a in rows:
    user_def = is_def(a)
    flags = []
    if a != "unsure" and s["vlm_is_defect"] != user_def:
        flags.append("人-VLM不一致")
    if s["gt_cls"] == "none" and user_def:
        flags.append("漏标实锤")
    if s["gt_cls"] != "none" and not user_def and a != "unsure":
        flags.append("标注可能误标")
    if flags:
        vlm_txt = ("有(%s)" % s["vlm_category"]) if s["vlm_is_defect"] else "无"
        print("%-22s 组=%-14s 标注=%-7s 千问=%-9s 你=%-6s | %s" % (
            s["id"], s["group"], s["gt_cls"], vlm_txt, a, " | ".join(flags)))
