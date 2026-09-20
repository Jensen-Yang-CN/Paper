# -*- coding: utf-8 -*-
"""
重标预筛 · 失败重试（账户欠费导致的部分失败）
============================================
1) 探测账户是否可用（1 次最小调用）
2) 重试：yolo 候选里 vlm_defect 为 None（判定失败）的 → 重新裁剪 + 千问判定
3) 重试：出现过失败判定的图片 → 重跑全图补漏扫描（补回漏检覆盖）
只更新失败项，已成功的不动。可重复运行（幂等，断点续跑安全）。

运行：E:\Miniconda\envs\label-studio\python.exe relabel_retry.py
"""
import argparse
import base64
import json
import os
import time
from pathlib import Path

import cv2
import requests

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week2_test_relabel")
CAND = BASE / "relabel_test_candidates.json"
IMG_DIR = Path(r"E:\CodeBase_YangJunJie\ClothingInspection\Final_Merged_Dataset_v3\images\train")

API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = os.environ.get("VLM_MODEL", "qwen3-vl-plus")
CONF_INFER = 0.10
CROP_MARGIN = 0.30
MIN_CROP = 96

PROMPT_VERIFY = """你是服装与面料质检流水线的 AI 质检员。现在给你一张质检图片的局部裁剪图，请判断该区域是否存在瑕疵。

【背景】这是质检场景，布料是平铺拍摄的，与日常衣物照片不同。

以下三类情况都属于瑕疵，必须检出：
- hole 破洞：布料上的孔洞、穿透破口（可能非常小）
- stain 污渍：油渍、水渍、污点、任何颜色异常区域
- wrinkle 褶皱：布料褶皱、折痕、死褶、明显不平整（质检场景中褶皱属于瑕疵，不能当作普通衣褶忽略）

判断规则：
1. 只要区域内有上述任何一种情况，is_defect 就必须为 true；
2. 只有完全平整、干净、无任何异常的区域才判 is_defect=false。

只输出 JSON：
{"is_defect": true 或 false, "category": "hole" 或 "stain" 或 "wrinkle" 或 "none", "confidence": 0到1的小数, "reason": "一句话中文理由"}"""

PROMPT_BOXES = """你是服装与面料质检系统的 AI 质检员。请检查整张图片，找出图中所有布料瑕疵，并用矩形框标出每一处。

瑕疵类别：
- hole 破洞：布料上穿透的孔洞（可能很小）
- stain 污渍：油渍、水渍、污点、颜色异常区域
- wrinkle 褶皱：明显褶皱、折痕、死褶、不平整（质检场景中褶皱属于瑕疵）

【注意】织物编织纹理、布边毛边、线头、光照阴影 不是瑕疵，不要标出。

输出 JSON（不要输出其他任何内容）：
{"defects": [{"category": "hole", "box": [x1, y1, x2, y2]}, {"category": "stain", "box": [...]}]}
坐标规则：box 为归一化坐标乘以 1000 后的整数，左上角为原点，范围 0~1000。
如果完全没有瑕疵，输出 {"defects": []}。"""


def call_vlm(img_path, prompt, api_key, timeout=90):
    b64 = base64.b64encode(open(img_path, "rb").read()).decode()
    payload = {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": prompt}]}], "temperature": 0.2}
    r = requests.post(API_URL, headers={"Authorization": f"Bearer {api_key}"},
                      json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:150]}")
    return r.json()["choices"][0]["message"]["content"]


def parse_json(content):
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(text[s:e + 1])
    except Exception:
        return None


def crop_with_margin(img, box, margin=CROP_MARGIN, min_size=MIN_CROP):
    h, w = img.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    pad_x, pad_y = bw * margin, bh * margin
    x1 = max(0, int(box[0] - pad_x)); y1 = max(0, int(box[1] - pad_y))
    x2 = min(w, int(box[2] + pad_x)); y2 = min(h, int(box[3] + pad_y))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = min_size // 2
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
    return img[y1:y2, x1:x2]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def probe(api_key):
    """1 次最小调用验证账户可用"""
    img_path = IMG_DIR / "000038_jpg.rf.c39b2b267330b9064b8e62db907fc582.jpg"
    if not img_path.exists():
        img_path = next(IMG_DIR.glob("*.jpg"))
    content = call_vlm(str(img_path), "回复：OK", api_key, timeout=30)
    return "OK" in content or True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="最多重试多少个候选（调试）")
    ap.add_argument("--fill-scan", action="store_true",
                    help="对没有 vlm 候选的图片补全图扫描（补漏检覆盖）")
    args = ap.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("[error] 设置 DASHSCOPE_API_KEY")

    print("探测账户可用性...")
    try:
        probe(api_key)
        print("  账户可用，开始重试")
    except Exception as e:
        raise SystemExit(f"[error] 账户仍不可用（可能未充值或还在处理）：{str(e)[:120]}")

    data = json.loads(CAND.read_text(encoding="utf-8"))
    total_fixed_cand = 0
    total_fixed_scan = 0

    for di, d in enumerate(data):
        img_path = IMG_DIR / d["image"]
        img = cv2.imread(str(img_path)) if img_path.exists() else None
        if img is None:
            print(f"[warn] 读图失败 {d['image']}")
            continue

        # ---- 1) 重试失败的裁剪判定 ----
        for c in d["candidates"]:
            if c["source"] != "yolo" or c.get("vlm_defect") is not None:
                continue
            if args.limit and total_fixed_cand >= args.limit:
                break
            crop = crop_with_margin(img, c["box"])
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            if crop.shape[0] < MIN_CROP or crop.shape[1] < MIN_CROP:
                s = MIN_CROP / min(crop.shape[0], crop.shape[1])
                crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            tmp = CAND.parent / f"_retry_{di:04d}.jpg"
            cv2.imwrite(str(tmp), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            try:
                pred = parse_json(call_vlm(str(tmp), PROMPT_VERIFY, api_key)) or {}
                c["vlm_defect"] = pred.get("is_defect")
                c["vlm_category"] = pred.get("category")
                c["vlm_conf"] = pred.get("confidence")
                c["vlm_reason"] = pred.get("reason", "")
                c["auto_keep"] = bool(c["vlm_defect"] is True and (c["vlm_conf"] or 0) >= 0.85)
                total_fixed_cand += 1
            except Exception as e:
                print(f"[warn] 重试仍失败 {d['image']} {c['id']}: {str(e)[:80]}")
            tmp.unlink(missing_ok=True)
            time.sleep(args.sleep)

        # ---- 2) 全图补漏扫描（重试失败的 或 --fill-scan 补全部缺失的）----
        if args.fill_scan:
            need_scan = not any(c["source"] == "vlm" for c in d["candidates"])
        else:
            need_scan = any(c["source"] == "yolo" and c.get("vlm_defect") is None for c in d["candidates"])
        if not need_scan:
            continue
        if args.limit and total_fixed_scan >= args.limit // 2:
            break
        try:
            pred = parse_json(call_vlm(str(img_path), PROMPT_BOXES, api_key, timeout=120)) or {}
            boxes = pred.get("defects", []) or []
        except Exception as e:
            print(f"[warn] 全图重扫失败 {d['image']}: {str(e)[:80]}")
            continue
        w, h = d["width"], d["height"]
        added = 0
        for vb in boxes:
            try:
                cat = str(vb.get("category", "")).lower()
                if cat not in ("hole", "stain", "wrinkle"):
                    continue
                bx = vb.get("box")
                if not bx or len(bx) != 4:
                    continue
                x1 = max(0, min(w, int(bx[0]) / 1000 * w))
                y1 = max(0, min(h, int(bx[1]) / 1000 * h))
                x2 = max(0, min(w, int(bx[2]) / 1000 * w))
                y2 = max(0, min(h, int(bx[3]) / 1000 * h))
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                dup = any(iou([x1, y1, x2, y2], c["box"]) > 0.3 for c in d["candidates"])
                if dup:
                    continue
                d["candidates"].append({
                    "id": f"{d['image']}_v{len([c for c in d['candidates'] if c['source']=='vlm'])}",
                    "source": "vlm", "box": [int(x1), int(y1), int(x2), int(y2)],
                    "yolo_cls": "-", "yolo_conf": None,
                    "vlm_defect": True, "vlm_category": cat, "vlm_conf": 0.7,
                    "vlm_reason": "全图扫描(重试)", "auto_keep": False,
                })
                added += 1
            except Exception:
                continue
        if added:
            total_fixed_scan += 1
        time.sleep(args.sleep)

        if (di + 1) % 30 == 0:
            CAND.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[{di + 1}/{len(data)}] 已重试，修复候选 {total_fixed_cand}，补全图扫描 {total_fixed_scan}")

    CAND.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n完成：修复候选判定 {total_fixed_cand} 个，补全图扫描 {total_fixed_scan} 张")
    still_fail = sum(1 for d in data for c in d["candidates"] if c.get("vlm_defect") is None)
    print(f"仍失败候选: {still_fail} 个（若有，再跑一次本脚本直到归零）")


if __name__ == "__main__":
    main()
