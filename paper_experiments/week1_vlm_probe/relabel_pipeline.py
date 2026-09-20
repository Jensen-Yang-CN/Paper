# -*- coding: utf-8 -*-
"""
重标注 · 第一步：VLM 辅助预筛（生成候选框清单）
================================================
对 val 集每张图：
  1) YOLO 低阈值(0.1)检出所有候选框
  2) 每个候选框裁剪 -> 千问判定(有/无 + 类别 + 置信度)
  3) 千问全图扫描，补出 YOLO 漏掉的框（VLM 输出归一化坐标）
  4) 候选框 IoU 去重，输出 relabel_candidates.json

输出字段（每张图）：
  image, width, height,
  candidates: [{id, source: yolo|vlm, box:[x1,y1,x2,y2], vlm_defect, vlm_category, vlm_conf}]

运行：E:\Miniconda\envs\label-studio\python.exe relabel_pipeline.py
"""
import argparse
import base64
import json
import os
import time
from pathlib import Path

import cv2
import requests
from ultralytics import YOLO

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week1_vlm_probe")
DATASET_ROOT = Path(r"E:\CodeBase_YangJunJie\ClothingInspection\Final_Merged_Dataset_v3")
MODEL_PATH = r"E:\CodeBase_YangJunJie\ClothingInspection\runs\detect\Final_Model_V6\weights\best.pt"
OUT = BASE / "relabel" / "relabel_candidates.json"
CLASS_NAMES = {0: "hole", 1: "stain", 2: "wrinkle"}

CONF_INFER = 0.10
CROP_MARGIN = 0.30
MIN_CROP = 96
IOU_DUP = 0.3          # VLM 框与 YOLO 候选重叠超过此值则视为重复
AUTO_ACCEPT_CONF = 0.85  # vlm 判定有瑕疵且置信度>=此值的 yolo 候选自动保留

API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = os.environ.get("VLM_MODEL", "qwen3-vl-plus")

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


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_vlm(images, prompt, api_key, timeout=60):
    """images: list of (b64_or_path, is_path)"""
    content = []
    for img, is_path in images:
        if is_path:
            b64 = encode_image(img)
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        else:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
    content.append({"type": "text", "text": prompt})
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": content}],
               "temperature": 0.2}
    r = requests.post(API_URL,
                      headers={"Authorization": f"Bearer {api_key}"},
                      json=payload, timeout=timeout)
    r.raise_for_status()
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
    crop = img[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", help="数据划分（默认 val）")
    ap.add_argument("--img-dir", default=None, help="图片目录（默认 DATASET/images/<split>；test 图在 train 里时传 train 目录）")
    ap.add_argument("--list", default=None, help="图片名清单 txt（每行一个文件名），只处理清单内的图")
    ap.add_argument("--out", default=None, help="输出 json 路径（默认 relabel/relabel_candidates.json）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（测试用）")
    ap.add_argument("--start", type=int, default=0, help="从第几张开始（断点续跑）")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("[error] 请设置环境变量 DASHSCOPE_API_KEY")

    out_path = Path(args.out) if args.out else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.img_dir:
        img_dir = Path(args.img_dir)
    else:
        img_dir = DATASET_ROOT / "images" / args.split
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if args.list:
        names = [l.strip() for l in Path(args.list).read_text(encoding="utf-8").splitlines() if l.strip()]
        images = [img_dir / n for n in names if (img_dir / n).exists()]
        missing = len(names) - len(images)
        if missing:
            print(f"[warn] 清单中 {missing} 张在 {img_dir} 不存在，已跳过")
    else:
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)
    if args.limit:
        images = images[:args.limit]
    images = images[args.start:]
    print(f"[info] 待处理 {len(images)} 张（来源目录: {img_dir}）")

    model = YOLO(MODEL_PATH)

    all_data = []
    if out_path.exists():
        all_data = json.loads(out_path.read_text(encoding="utf-8"))
        done_ids = {d["image"] for d in all_data}
    else:
        done_ids = set()

    for idx, img_path in enumerate(images):
        if img_path.name in done_ids:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] 读图失败 {img_path.name}")
            continue
        h, w = img.shape[:2]

        # ---- 1) YOLO 候选 ----
        res = model.predict(source=img, conf=CONF_INFER, iou=0.5, imgsz=640,
                            device="0", verbose=False)
        yolo_dets = []
        if res and res[0].boxes is not None:
            for b in res[0].boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                yolo_dets.append({"box": [x1, y1, x2, y2],
                                  "cls": int(b.cls[0]),
                                  "conf": float(b.conf[0])})

        candidates = []
        # ---- 2) 每个 YOLO 候选裁剪 + 千问判定 ----
        for i, d in enumerate(yolo_dets):
            crop, _ = crop_with_margin(img, d["box"])
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            if crop.shape[0] < MIN_CROP or crop.shape[1] < MIN_CROP:
                s = MIN_CROP / min(crop.shape[0], crop.shape[1])
                crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            tmp = out_path.parent / f"_tmp_{idx:04d}_{i}.jpg"  # 短名，避免长文件名 MAX_PATH
            cv2.imwrite(str(tmp), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            try:
                content = call_vlm([(str(tmp), True)], PROMPT_VERIFY, api_key)
                pred = parse_json(content) or {}
            except Exception as e:
                pred = {"error": str(e)[:80]}
            tmp.unlink(missing_ok=True)
            candidates.append({
                "id": f"{img_path.stem}_y{i}",
                "source": "yolo",
                "box": [int(v) for v in d["box"]],
                "yolo_cls": CLASS_NAMES.get(d["cls"], "?"),
                "yolo_conf": round(d["conf"], 3),
                "vlm_defect": pred.get("is_defect"),
                "vlm_category": pred.get("category"),
                "vlm_conf": pred.get("confidence"),
                "vlm_reason": pred.get("reason", ""),
            })
            time.sleep(args.sleep)

        # ---- 3) 千问全图补漏 ----
        try:
            content = call_vlm([(str(img_path), True)], PROMPT_BOXES, api_key, timeout=90)
            pred = parse_json(content) or {}
            vlm_boxes = pred.get("defects", []) or []
        except Exception as e:
            vlm_boxes = []
            print(f"[warn] 全图扫描失败 {img_path.name}: {str(e)[:80]}")

        for vb in vlm_boxes:
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
                dup = any(iou([x1, y1, x2, y2], c["box"]) > IOU_DUP for c in candidates)
                if dup:
                    continue
                candidates.append({
                    "id": f"{img_path.stem}_v{len([c for c in candidates if c['source']=='vlm'])}",
                    "source": "vlm",
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "yolo_cls": "-",
                    "yolo_conf": None,
                    "vlm_defect": True,
                    "vlm_category": cat,
                    "vlm_conf": 0.7,
                    "vlm_reason": "全图扫描",
                })
            except Exception:
                continue

        # ---- 4) 自动保留标记 ----
        for c in candidates:
            c["auto_keep"] = bool(c["source"] == "yolo" and c["vlm_defect"] is True
                                  and (c["vlm_conf"] or 0) >= AUTO_ACCEPT_CONF)

        all_data.append({
            "image": img_path.name,
            "width": w, "height": h,
            "has_yolo_det": len(yolo_dets) > 0,
            "candidates": candidates,
        })
        if (idx + 1) % 20 == 0 or (idx + 1) == len(images):
            out_path.write_text(json.dumps(all_data, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[{idx + 1}/{len(images)}] 已处理 {img_path.name}，候选 {len(candidates)} 个，自动保留 {sum(1 for c in candidates if c['auto_keep'])} 个")

    out_path.write_text(json.dumps(all_data, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(len(d["candidates"]) for d in all_data)
    auto = sum(1 for d in all_data for c in d["candidates"] if c["auto_keep"])
    print(f"\n完成：{len(all_data)} 张图，候选框 {total} 个，自动保留 {auto} 个，待人工确认 {total - auto} 个")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
