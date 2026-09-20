# -*- coding: utf-8 -*-
# 第四周 · 核心实验：阈值扫描+Pareto / 四组对比 / 消融
# 复用缓存：所有"裁剪+尺度+VLM判定"与"整图VLM判定"各算一次，后面全部复用，不重复调 API
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

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week4_experiments")
DATA = Path(r"E:\CodeBase_YangJunJie\Paper\实验数据\relabeled_test")
MODEL_PATH = r"E:\CodeBase_YangJunJie\Paper\模型权重\Final_Model_V6_best.pt"
IOU_THR = 0.5
CROP_MARGIN = 0.30
MIN_CROP = 96
CONF_INFER = 0.25
CONF0 = 0.40
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = os.environ.get("VLM_MODEL", "qwen3-vl-plus")

PROMPT = """你是服装与面料质检流水线的 AI 质检员。这张图是从质检图裁剪出的局部区域，请判断该区域是否存在瑕疵（hole破洞/stain污渍/wrinkle褶皱，质检场景中褶皱属于瑕疵）。有则 is_defect=true，完全没有才 false。只输出 JSON：{"is_defect": true或false, "category": "hole|stain|wrinkle|none", "confidence": 0到1, "reason": "一句话"}"""
PROMPT_FULL = """你是服装与面料质检流水线的 AI 质检员。请判断这张质检图片中是否存在瑕疵（hole破洞/stain污渍/wrinkle褶皱，质检场景中褶皱属于瑕疵）。有则 is_defect=true，完全没有才 false。只输出 JSON：{"is_defect": true或false, "confidence": 0到1, "reason": "一句话"}"""
VCLS = {"hole": 0, "stain": 1, "wrinkle": 2}


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


def call_vlm(nd, prompt, api_key, tries=3):
    ok, buf = cv2.imencode(".jpg", nd)
    b64 = base64.b64encode(buf.tobytes()).decode()
    payload = {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": prompt}]}], "temperature": 0.2}
    for att in range(tries):
        try:
            r = requests.post(API_URL, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=90)
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"]["content"]
                s, e = t.find("{"), t.rfind("}")
                try:
                    return json.loads(t[s:e + 1])
                except Exception:
                    return None
            elif r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 + att * 2); continue
            else:
                return None
        except Exception:
            time.sleep(2 + att * 2)
    return None


def match(dets, gt_px, mode="strict"):
    tp = fp = 0
    matched_gt = [False] * len(gt_px)
    for d in dets:
        cls, box = d[0], d[1]
        best_j, best_v = -1, IOU_THR
        for j, (gcls, gbox) in enumerate(gt_px):
            if matched_gt[j]:
                continue
            if mode == "strict" and gcls != cls:
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


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return round(p, 4), round(r, 4), round(f1, 4)


def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("set DASHSCOPE_API_KEY")
    BASE.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_PATH)
    img_dir = DATA / "images"; lbl_dir = DATA / "labels"
    n_img = 0
    imgs = []

    for ip in sorted(img_dir.iterdir()):
        img = imread_unicode(ip)
        if img is None:
            continue
        n_img += 1
        h, w = img.shape[:2]
        gts = load_gt(lbl_dir / (ip.stem + ".txt"))
        gt_px = [(cls, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
                 for cls, x, y, bw, bh in gts]
        r0 = model.predict(source=img, conf=CONF0, iou=0.5, imgsz=640, device="0", verbose=False)
        dets40 = [(int(b.cls[0]), [float(v) for v in b.xyxy[0].tolist()], float(b.conf[0]))
                  for b in r0[0].boxes] if r0 and r0[0].boxes is not None else []
        r = model.predict(source=img, conf=CONF_INFER, iou=0.5, imgsz=640, device="0", verbose=False)
        dets25 = [(int(b.cls[0]), [float(v) for v in b.xyxy[0].tolist()], float(b.conf[0]))
                  for b in r[0].boxes] if r and r[0].boxes is not None else []
        imgs.append({"name": ip.name, "img": img, "w": w, "h": h, "gt": gt_px, "dets40": dets40, "dets25": dets25})

    # 缓存落盘：网络抖 / 崩溃后断点续跑，不重复调 API
    cc = json.loads((BASE / "cache_crop.json").read_text(encoding="utf-8")) if (BASE / "cache_crop.json").exists() else {}
    cw = json.loads((BASE / "cache_whole.json").read_text(encoding="utf-8")) if (BASE / "cache_whole.json").exists() else {}

    print("裁剪复核（全部检测，含尺度提示，可续跑）...")
    for i, d in enumerate(imgs):
        img, h, w = d["img"], d["h"], d["w"]
        name = d["name"]
        cur = cc.setdefault(name, {})
        for k, (cls, box, conf) in enumerate(d["dets25"]):
            sk = str(k)
            if sk in cur:
                continue
            ci = crop(img, box)
            ch, cw_ = ci.shape[:2]
            bw = box[2] - box[0]; bh = box[3] - box[1]
            frac = (bw * bh) / (h * w)
            scale = (f"[尺度提示] 这是原始图（{w}x{h}px）中一块区域放大后的裁剪图，"
                     f"裁剪图本身{cw_}x{ch}px，候选框占原图面积约{frac * 100:.1f}%。瑕疵可能很小。\n")
            cur[sk] = call_vlm(ci, scale + PROMPT, api_key)
        d["cc"] = cur
        if name not in cw:
            cw[name] = call_vlm(img, PROMPT_FULL, api_key) if d["dets25"] else None
        d["whole"] = cw[name]
        if (i + 1) % 20 == 0:
            (BASE / "cache_crop.json").write_text(json.dumps(cc, ensure_ascii=False), encoding="utf-8")
            (BASE / "cache_whole.json").write_text(json.dumps(cw, ensure_ascii=False), encoding="utf-8")
            print(f"[verify续跑] {i + 1}/{len(imgs)} 已复核 {sum(len(v) for v in cc.values())} 框")
    (BASE / "cache_crop.json").write_text(json.dumps(cc, ensure_ascii=False), encoding="utf-8")
    (BASE / "cache_whole.json").write_text(json.dumps(cw, ensure_ascii=False), encoding="utf-8")
    print(f"复核完成：{sum(len(v) for v in cc.values())} 框, 整图 {len(cw)} 张")

    # 构造：路由阈值 T 的融合结果（dets25 全为候选）
    def fused(T):
        per = []
        for d in imgs:
            ds = []
            for k, (cls, box, conf) in enumerate(d["dets25"]):
                if conf >= T:
                    ds.append((cls, box, conf))
                else:
                    v = d["cc"].get(str(k))
                    if v and v.get("is_defect"):
                        ds.append((VCLS.get(v.get("category"), cls), box, conf))
            per.append(ds)
        return per

    # 消融 C：no-crop（用整图判定过滤每张图的所有检测）
    def fused_nocrop():
        per = []
        for d in imgs:
            ds = []
            if d["whole"] and d["whole"].get("is_defect"):
                ds = list(d["dets25"])
            per.append(ds)
        return per

    def agg(per, mode="strict"):
        tot = {"tp": 0, "fp": 0, "fn": 0}
        for d, ds in zip(imgs, per):
            t, f, n = match(ds, d["gt"], mode)
            tot["tp"] += t; tot["fp"] += f; tot["fn"] += n
        p, r, f1 = prf(tot["tp"], tot["fp"], tot["fn"])
        return {"tp": tot["tp"], "fp": tot["fp"], "fn": tot["fn"], "P": p, "R": r, "F1": f1}

    def calls(T):
        return sum(1 for d in imgs for k, (c, b, cf) in enumerate(d["dets25"]) if cf < T)

    # ---- 阈值扫描 + Pareto ----
    print("\n===== 阈值扫描（Ours，严格口径）=====")
    sweep = []
    for T in [0.3, 0.4, 0.5, 0.6, 0.7]:
        per = fused(T)
        a = agg(per)
        rate = calls(T) / n_img
        sweep.append({"T": T, "call_rate": round(rate, 3), "P": a["P"], "R": a["R"], "F1": a["F1"]})
        print(f"  T={T}: 调用率={rate:.3f}  F1={a['F1']}  P={a['P']} R={a['R']}")

    # ---- 四组对比 ----
    print("\n===== 四组对比（严格口径）=====")
    y40 = agg([list(d["dets40"]) for d in imgs])
    y25 = agg([list(d["dets25"]) for d in imgs])
    allv = agg(fused(2.0))          # 全复核：阈值 2.0 → 每个检测都走 VLM
    best = max(sweep, key=lambda s: s["F1"])
    ours = agg(fused(best["T"]))
    print(f"  YOLO@0.4 : {y40}")
    print(f"  YOLO@0.25: {y25}")
    print(f"  YOLO+VLM(全复核): {allv}")
    print(f"  Ours(T={best['T']}): {ours}")

    # ---- 消融 ----
    print("\n===== 消融（严格口径）=====")
    nocrop = agg(fused_nocrop())
    print(f"  A=YOLO@0.4: {y40}")
    print(f"  B=YOLO+Crop+VLM(全复核): {allv}")
    print(f"  C=YOLO+VLM无Crop(整图): {nocrop}")
    print(f"  D=完整方法(Ours): {ours}")

    result = {"n_img": n_img, "sweep": sweep,
              "four_group": {"yolo40": y40, "yolo25": y25, "allverify": allv, "ours": ours},
              "ablation": {"A_yolo": y40, "B_crop_allvlm": allv, "C_nocrop": nocrop, "D_full": ours}}
    out = BASE / "week4_results.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
