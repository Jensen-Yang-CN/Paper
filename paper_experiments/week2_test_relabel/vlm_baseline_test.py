# -*- coding: utf-8 -*-
"""
VLM 复核 Baseline 正式化（在重标 test 上评估 qwen3-vl-plus）
============================================================
对 test 300 张：用 qwen3-vl-plus + 质检提示词（整图）判断"有无瑕疵"，
与图像级 GT 对比，输出 P/R/F1 —— 即论文的 "VLM 独立判断" 基线。

用法：set DASHSCOPE_API_KEY=... 然后:
    E:\Miniconda\envs\label-studio\python.exe vlm_baseline_test.py
"""
import base64
import json
import os
import time
from pathlib import Path

import numpy as np
import requests

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week2_test_relabel")
DATA = Path(r"E:\CodeBase_YangJunJie\Paper\实验数据\relabeled_test")
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = os.environ.get("VLM_MODEL", "qwen3-vl-plus")

PROMPT = """你是服装与面料质检流水线的 AI 质检员。请判断这张质检图片中是否存在以下瑕疵之一：
- hole 破洞：布料上的孔洞、穿透破口（可能很小）
- stain 污渍：油渍、水渍、污点、颜色异常区域
- wrinkle 褶皱：明显褶皱、折痕、死褶、不平整（质检场景中褶皱属于瑕疵，不能当作普通衣褶忽略）

只要图中存在上述任何一种情况，is_defect 就为 true；只有完全平整干净才为 false。
只输出 JSON：
{"is_defect": true 或 false, "confidence": 0到1的小数, "reason": "一句话中文理由"}"""


def imread_unicode(path):
    import cv2
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def load_gt(lbl_path):
    boxes = []
    if lbl_path.exists():
        for line in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split()
            if len(p) == 5:
                boxes.append((float(p[0]), float(p[1]), float(p[2]), float(p[3])))
    return boxes


def parse_json(t):
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return None


def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("请设置 DASHSCOPE_API_KEY")
    img_dir = DATA / "images"
    lbl_dir = DATA / "labels"
    tp = fp = fn = tn = 0
    rows = []
    for img_path in sorted(img_dir.iterdir()):
        gt_has = len(load_gt(lbl_dir / (img_path.stem + ".txt"))) > 0
        b64 = base64.b64encode(open(str(img_path), "rb").read()).decode()
        payload = {"model": MODEL, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": PROMPT}]}], "temperature": 0.2}
        pred = None
        try:
            r = requests.post(API_URL, headers={"Authorization": f"Bearer {api_key}"},
                              json=payload, timeout=90)
            if r.status_code == 200:
                pred = parse_json(r.json()["choices"][0]["message"]["content"])
            else:
                pred = {"error": f"HTTP {r.status_code}"}
        except Exception as e:
            pred = {"error": str(e)[:80]}
        pred_has = bool(pred and pred.get("is_defect"))
        if gt_has and pred_has:
            tp += 1
        elif gt_has and not pred_has:
            fn += 1
        elif not gt_has and pred_has:
            fp += 1
        else:
            tn += 1
        rows.append({"image": img_path.name, "gt_has": gt_has,
                     "vlm_pred": pred, "vlm_has": pred_has})
        time.sleep(0.25)

    n = tp + fp + fn + tn
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    print(f"\n===== VLM(整图) 图像级基线 (n={n}) =====")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")

    result = {"model": MODEL, "n": n, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
              "Precision": round(p, 4), "Recall": round(r, 4), "F1": round(f1, 4),
              "rows": rows}
    out = BASE / "vlm_baseline_test.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
