# -*- coding: utf-8 -*-
# 第三周 · 方法V1：YOLO + 置信度路由 + VLM复核 + 结果融合（推理时协同）
# 在重标 test 上跑，对比 YOLO@0.4 / YOLO@0.25(控制) / Proposed，输出框级P/R/F1 + 调用率 + 延迟
import base64
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests
from ultralytics import YOLO

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week3_method")
DATA = Path(r"E:\CodeBase_YangJunJie\Paper\实验数据\relabeled_test")
MODEL_PATH = r"E:\CodeBase_YangJunJie\Paper\模型权重\Final_Model_V6_best.pt"
CONF_HIGH = 0.60   # 路由阈值：>=此置信度的检测直接接受
CONF_LOW = 0.25    # 推理阈值：低于此的一律不输出
CONF0 = 0.40       # YOLO 单独基线阈值
IOU_THR = 0.5
CROP_MARGIN = 0.30
MIN_CROP = 96
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = os.environ.get("VLM_MODEL", "qwen3-vl-plus")

PROMPT = """你是服装与面料质检流水线的 AI 质检员。这张图是从质检图裁剪出的局部区域，请判断该区域是否存在瑕疵（hole破洞/stain污渍/wrinkle褶皱，质检场景中褶皱属于瑕疵）。有则 is_defect=true，完全没有才 false。只输出 JSON：{"is_defect": true或false, "category": "hole|stain|wrinkle|none", "confidence": 0到1, "reason": "一句话"}"""


def imread_unicode(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def load_gt(lbl):
    boxes = []
    if lbl.exists():
        for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split()
            if len(p) == 5:
                boxes.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return boxes


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def crop(img, box):
    h, w = img.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    px, py = bw * CROP_MARGIN, bh * CROP_MARGIN
    x1 = max(0, int(box[0] - px)); y1 = max(0, int(box[1] - py))
    x2 = min(w, int(box[2] + px)); y2 = min(h, int(box[3] + py))
    if x2 - x1 < MIN_CROP or y2 - y1 < MIN_CROP:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = MIN_CROP // 2
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
    return img[y1:y2, x1:x2]


def vlm_verify(crop_img, api_key, scale=""):
    ok, buf = cv2.imencode(".jpg", crop_img)
    b64 = base64.b64encode(buf.tobytes()).decode()
    prompt = scale + PROMPT
    payload = {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": prompt}]}], "temperature": 0.2}
    r = requests.post(API_URL, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=60)
    if r.status_code != 200:
        return {"is_defect": None, "error": f"HTTP{r.status_code}"}
    t = r.json()["choices"][0]["message"]["content"]
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return {"is_defect": None, "error": "json"}


def match(dets, gt_px, class_mode="strict"):
    tp = fp = 0
    matched_gt = [False] * len(gt_px)
    for d in dets:
        cls, box = d[0], d[1]
        best_j, best_v = -1, IOU_THR
        for j, (gcls, gbox) in enumerate(gt_px):
            if matched_gt[j]:
                continue
            if class_mode == "strict" and gcls != cls:
                continue
            v = iou(box, gbox)
            if v > best_v:
                best_v, best_j = v, j
        if best_j >= 0:
            matched_gt[best_j] = True
            tp += 1
        else:
            fp += 1
    return tp, fp, len(gt_px) - sum(matched_gt)


def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("set DASHSCOPE_API_KEY")
    (BASE / "error_cases_week3").mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_PATH)
    img_dir = DATA / "images"; lbl_dir = DATA / "labels"

    stats = defaultdict(lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}))
    calls = 0
    t_yolo = t_vlm = 0.0
    n_img = 0
    img_tp = img_fp = img_fn = 0

    for img_path in sorted(img_dir.iterdir()):
        img = imread_unicode(img_path)
        if img is None:
            continue
        n_img += 1
        h, w = img.shape[:2]
        gts = load_gt(lbl_dir / (img_path.stem + ".txt"))
        gt_px = [(cls, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
                 for cls, x, y, bw, bh in gts]

        t0 = time.time()
        res = model.predict(source=img, conf=CONF_LOW, iou=0.5, imgsz=640, device="0", verbose=False)
        t_yolo += time.time() - t0
        dets = []
        if res and res[0].boxes is not None:
            for b in res[0].boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                dets.append((int(b.cls[0]), [x1, y1, x2, y2], float(b.conf[0])))

        # 路由：高置信直接接受，低置信交给 VLM 复核（类别以 VLM 为准做纠错）
        VCLS = {"hole": 0, "stain": 1, "wrinkle": 2}
        final = []
        for cls, box, conf in dets:
            if conf >= CONF_HIGH:
                final.append((cls, box, conf))
            else:
                t0 = time.time()
                crop_img = crop(img, box)
                ch, cw = crop_img.shape[:2]
                bw = box[2] - box[0]; bh = box[3] - box[1]
                frac = (bw * bh) / (h * w)
                scale = (f"[尺度提示] 这是原始图（{w}x{h}px）中一块区域放大后的裁剪图，"
                         f"裁剪图本身{cw}x{ch}px，候选缺陷框占原图面积约{frac * 100:.1f}%。"
                         f"瑕疵可能很小，请据此判断该区域是否存在瑕疵。\n")
                v = vlm_verify(crop_img, api_key, scale)
                t_vlm += time.time() - t0
                calls += 1
                if v.get("is_defect"):
                    eff = VCLS.get(v.get("category"), cls)
                    final.append((eff, box, conf))

        # 记录各方法框级指标
        t0 = time.time()
        res40 = model.predict(source=img, conf=CONF0, iou=0.5, imgsz=640, device="0", verbose=False)
        t_yolo += time.time() - t0
        dets40 = []
        if res40 and res40[0].boxes is not None:
            for b in res40[0].boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                dets40.append((int(b.cls[0]), [x1, y1, x2, y2], float(b.conf[0])))

        for name, ds in [("yolo40", dets40), ("yolo25", dets), ("proposed", final)]:
            for mode in ("strict", "agnostic"):
                t, f, n = match(ds, gt_px, mode)
                stats[mode][name]["tp"] += t; stats[mode][name]["fp"] += f; stats[mode][name]["fn"] += n
            if name == "proposed":
                if len(gt_px) and len(ds):
                    img_tp += 1
                elif len(gt_px):
                    img_fn += 1
                elif len(ds):
                    img_fp += 1

        if n_img % 40 == 0:
            print(f"[{n_img}] 已处理 {img_path.name}，VLM累计调用 {calls}")

    def report(mode):
        print(f"\n===== 框级 P/R/F1（IoU=0.5，{'类别严格' if mode == 'strict' else '类别无关'}）=====")
        print(f"{'方法':<12}{'TP':>6}{'FP':>6}{'FN':>6}{'Precision':>12}{'Recall':>10}{'F1':>8}")
        for name in ["yolo40", "yolo25", "proposed"]:
            s = stats[mode][name]
            p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0
            r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0
            f1 = 2 * p * r / (p + r) if (p + r) else 0
            print(f"{name:<12}{s['tp']:>6}{s['fp']:>6}{s['fn']:>6}{p:>12.4f}{r:>10.4f}{f1:>8.4f}")

    report("strict")
    report("agnostic")

    print(f"\n图像级(Proposed, 图有/无缺陷): TP={img_tp} FP={img_fp} FN={img_fn}")
    ip = img_tp / (img_tp + img_fp) if (img_tp + img_fp) else 0
    ir = img_tp / (img_tp + img_fn) if (img_tp + img_fn) else 0
    print(f"  图像级F1 = {2 * ip * ir / (ip + ir) if (ip + ir) else 0:.4f}")

    print(f"\nVLM 调用次数: {calls}（图片 {n_img} 张，平均 {calls / n_img:.1f} 次/张）")
    print(f"VLM 平均单次延迟: {t_vlm / calls:.2f}s   YOLO 单图延迟: {t_yolo / n_img:.3f}s")
    out = BASE / "collab_results.json"
    out.write_text(json.dumps({
        "n_img": n_img, "calls": calls, "avg_calls_per_img": calls / n_img,
        "t_vlm_avg": t_vlm / calls, "t_yolo_per_img": t_yolo / n_img,
        "box": {mode: {k: dict(v) for k, v in stats[mode].items()} for mode in ("strict", "agnostic")},
        "image_level": {"tp": img_tp, "fp": img_fp, "fn": img_fn,
                        "f1": round(2 * ip * ir / (ip + ir), 4) if (ip + ir) else 0}
    }, ensure_ascii=False, indent=2))
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
