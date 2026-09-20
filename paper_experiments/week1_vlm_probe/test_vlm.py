# -*- coding: utf-8 -*-
"""
VLM 可行性探针（第一周 · Qwen-VL 复核测试）
==========================================
用法：
    set DASHSCOPE_API_KEY=sk-xxx
    E:/Miniconda/envs/label-studio/python.exe test_vlm.py --meta hard_samples_meta.csv --limit 10

逻辑：
    对每个难样本裁剪图调用 Qwen-VL（dashscope OpenAI 兼容接口），让模型判断
    "该区域是否有瑕疵 + 类别 + 置信度"，与真实标签对比，输出各 case_type 的一致率。

判定标准（作为论文方向是否跑通的依据）：
    - low_conf_tp / high_conf_tp / missed_gt / cls_conflict  期望 VLM 回答 is_defect=True
    - false_positive / normal_region                         期望 VLM 回答 is_defect=False
    成立条件（建议）：
        难样本类(low_conf_tp + missed_gt)一致率 >= 70%  且
        误检拒绝率(false_positive) >= 60%               且
        正常区域误报率(normal_region) 尽量低
    不满足 -> 先调 prompt / 换模型(qwen-vl-max) / 换输入方式，再试一次；仍不行 -> 讨论 pivot。

说明：
    - 模型名可通过环境变量 VLM_MODEL 覆盖（默认 qwen-vl-plus，备选 qwen-vl-max / qwen3-vl-plus）
    - 本版只发裁剪图；后续 H2 实验再对比"裁剪图 + 原图/标注框"两种输入
"""
import argparse
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests

API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = os.environ.get("VLM_MODEL", "qwen-vl-plus")

# 提示词 v2：强调"质检场景褶皱=瑕疵"（召回好，但正常区域误报高）
PROMPT_V2 = """你是服装与面料质检流水线的 AI 质检员。现在给你一张质检图片的局部裁剪图，请判断该区域是否存在瑕疵。

【背景】这是质检场景，布料是平铺拍摄的，与日常衣物照片不同。质检要求"尽量不漏报"。

以下三类情况都属于瑕疵，必须检出：
- hole 破洞：布料上的孔洞、穿透破口（可能非常小，注意观察深色小点/缺口）
- stain 污渍：油渍、水渍、污点、任何颜色异常区域
- wrinkle 褶皱：布料褶皱、折痕、死褶、明显不平整——【注意：质检场景中布料的褶皱/折痕属于瑕疵 wrinkle，不是正常状态，不能当作普通衣褶忽略】

判断规则：
1. 只要区域内有上述任何一种情况，is_defect 就必须为 true；
2. 瑕疵可能微小、颜色对比可能很弱，请放大观察纹理、颜色与明暗差异；
3. 只有完全平整、干净、无任何异常的区域才判 is_defect=false。

只输出 JSON，不要输出其他任何内容：
{"is_defect": true 或 false, "category": "hole" 或 "stain" 或 "wrinkle" 或 "none", "confidence": 0到1的小数, "reason": "一句话中文理由"}"""

# 提示词 v3：加入"什么不是瑕疵"的负例（压制误报，但把真实缺陷也压没了）
PROMPT_V3 = """你是服装与面料质检流水线的 AI 质检员。现在给你一张质检图片的局部裁剪图，请判断该区域是否存在瑕疵。

【背景】这是质检场景，布料平铺拍摄。以下三类是必须检出的瑕疵：
- hole 破洞：布料上真正穿透的孔洞，能看到孔洞内部或透过看到背景
- stain 污渍：油渍、水渍、污点、颜色异常区域
- wrinkle 褶皱：明显褶皱、折痕、死褶、不平整（质检场景中褶皱属于瑕疵，不是正常衣褶）

【重要：以下情况不是瑕疵，绝不能误报】
- 布料本身的编织纹理、纤维间隙、织造纹路 → 不是破洞
- 布边、毛边、线头、裁剪边缘 → 不是破洞（除非有明显穿透破口）
- 拍摄光照造成的明暗差异、阴影 → 不是污渍
- 布料自然垂坠形成的大弧度 → 不是褶皱（只有明显的、集中的折痕/死褶才算）

判断规则：
1. 只有确认存在上述瑕疵才 is_defect=true；拿不准就判 false；
2. hole 必须有"穿透"的视觉证据（看到孔洞内部或背景），仅仅是纹理间隙不算；
3. 完全平整、干净、纹理均匀的区域判 is_defect=false。

只输出 JSON，不要输出其他任何内容：
{"is_defect": true 或 false, "category": "hole" 或 "stain" 或 "wrinkle" 或 "none", "confidence": 0到1的小数, "reason": "一句话中文理由"}"""

# 通过环境变量 VLM_PROMPT_VERSION 选择：v2（默认，召回优先）/ v3（负例压制）
PROMPT = PROMPT_V2 if os.environ.get("VLM_PROMPT_VERSION", "v2") == "v2" else PROMPT_V3

# 每种 case_type 期望的 is_defect
EXPECTED = {
    "low_conf_tp": True,
    "high_conf_tp": True,
    "missed_gt": True,
    "cls_conflict": True,
    "false_positive": False,
    "normal_region": False,
}


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_vlm(image_path, api_key, timeout=60):
    data_url = f"data:image/jpeg;base64,{encode_image(image_path)}"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": PROMPT},
            ]},
        ],
        "temperature": 0.2,
    }
    resp = requests.post(API_URL,
                         headers={"Authorization": f"Bearer {api_key}"},
                         json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_json_response(content):
    """从模型输出里提取 JSON（容忍 markdown 代码块/前后废话）"""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Qwen-VL 难样本复核探针")
    ap.add_argument("--meta", required=True, help="extract_hard_samples.py 生成的 meta csv")
    ap.add_argument("--out", default=None, help="结果 csv 输出路径")
    ap.add_argument("--limit", type=int, default=0, help="最多测试多少张，0=全部")
    ap.add_argument("--case-type", default=None, help="只测某类，如 false_positive")
    ap.add_argument("--sleep", type=float, default=0.3, help="每次调用间隔(秒)，防止限流")
    args = ap.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        sys.exit("[error] 请先设置环境变量 DASHSCOPE_API_KEY（dashscope 控制台获取）")

    out_path = args.out or (Path(args.meta).with_name("vlm_results.csv"))
    with open(args.meta, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if args.case_type:
        rows = [r for r in rows if r["case_type"] == args.case_type]
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        sys.exit("[error] meta 为空或筛选后无样本")

    results, stat = [], {}
    for i, row in enumerate(rows):
        crop = row["crop_path"]
        if not os.path.exists(crop):
            print(f"[skip] 缺文件: {crop}")
            continue
        expected = EXPECTED.get(row["case_type"])
        verdict, parsed, error = "skip", None, ""
        try:
            content = call_vlm(crop, api_key)
            parsed = parse_json_response(content)
            if parsed is None:
                error = f"JSON解析失败: {content[:120]}"
                verdict = "error"
            else:
                pred = bool(parsed.get("is_defect"))
                verdict = "agree" if pred == expected else "disagree"
        except Exception as e:
            error = str(e)[:120]
            verdict = "error"
        results.append({**row, "expected_defect": str(expected), "vlm_result": json.dumps(parsed, ensure_ascii=False),
                        "verdict": verdict, "error": error})
        stat.setdefault(row["case_type"], {"n": 0, "agree": 0})
        if verdict in ("agree", "disagree"):
            stat[row["case_type"]]["n"] += 1
            if verdict == "agree":
                stat[row["case_type"]]["agree"] += 1
        print(f"[{i+1}/{len(rows)}] {row['case_id']} {row['case_type']:<15s} -> {verdict}"
              f"{(' ' + json.dumps(parsed, ensure_ascii=False)) if parsed else (' | ' + error)}")
        time.sleep(args.sleep)

    print("\n===== 汇总 =====")
    total_n = total_agree = 0
    for k, v in stat.items():
        acc = v["agree"] / v["n"] if v["n"] else 0
        total_n += v["n"]
        total_agree += v["agree"]
        print(f"{k:15s} n={v['n']:3d} agree={v['agree']:3d} acc={acc:.1%}")
    if total_n:
        print(f"{'ALL':15s} n={total_n:3d} agree={total_agree:3d} acc={total_agree/total_n:.1%}")

    # 难样本子集（决定方向的核心指标）
    hard = [v for k, v in stat.items() if k in ("low_conf_tp", "missed_gt")]
    hn = sum(v["n"] for v in hard)
    ha = sum(v["agree"] for v in hard)
    if hn:
        print(f"\n>> 难样本(low_conf_tp+missed_gt) 一致率: {ha}/{hn} = {ha/hn:.1%}  "
              f"{'== 达到 70% 目标，方向可继续 ==' if ha/hn >= 0.7 else '== 未达标，先调 prompt/模型再试 =='}")

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
