# -*- coding: utf-8 -*-
"""
论文方法保真度抽检
==================
用系统自带的协同推理流水线，在重标 test 集的一个子集上复算框级指标，
与论文定版结果对比 —— 用来验证"系统里跑的"和"论文里写的"是同一套方法。

    python scripts/validate_subset.py --n 80 --threshold 0.6
    python scripts/validate_subset.py --n 80 --no-vlm      # 关掉复核做对照

评测口径与论文一致：框级匹配 IoU=0.5；严格=须命中同类别 GT；无关=忽略类别。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config                                    # noqa: E402
from app.core.agent_tools import build_registry           # noqa: E402
from app.core.detector import YoloDetector, imread_unicode  # noqa: E402
from app.core.pipeline import Pipeline                    # noqa: E402
from app.core.vlm_client import verifier_info             # noqa: E402


def load_gt(lbl: Path) -> list[tuple[int, list[float]]]:
    out = []
    if lbl.exists():
        for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split()
            if len(p) == 5:
                out.append((int(float(p[0])), [float(v) for v in p[1:]]))
    return out


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def match(dets, gt, strict: bool) -> tuple[int, int, int]:
    """返回 (TP, FP, FN)。dets 元素需含 cls_id 与 box(像素)。"""
    tp = fp = 0
    used = [False] * len(gt)
    for d in dets:
        best_j, best_v = -1, config.IOU_THRESHOLD
        for j, (gcls, gbox) in enumerate(gt):
            if used[j]:
                continue
            if strict and gcls != d["cls_id"]:
                continue
            v = iou(d["box"], gbox)
            if v > best_v:
                best_v, best_j = v, j
        if best_j >= 0:
            used[best_j] = True
            tp += 1
        else:
            fp += 1
    return tp, fp, len(gt) - sum(used)


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"P": round(p, 4), "R": round(r, 4), "F1": round(f1, 4),
            "TP": tp, "FP": fp, "FN": fn}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="抽检图片数")
    ap.add_argument("--threshold", type=float, default=config.DEFAULT_ROUTER_THRESHOLD)
    ap.add_argument("--infer-conf", type=float, default=config.DEFAULT_INFER_CONF)
    ap.add_argument("--no-vlm", action="store_true", help="关闭大模型复核（做对照）")
    ap.add_argument("--sample-mode", choices=("head", "random"), default="head",
                    help="head=取排序后前 N 张；random=固定种子随机抽样（更能代表全集）")
    ap.add_argument("--seed", type=int, default=20260922, help="随机抽样种子")
    ap.add_argument("--out", default="", help="结果 JSON 输出路径")
    args = ap.parse_args()

    if config.SAMPLE_DIR is None:
        print("[错误] 未找到 test 数据集目录（可用 SAMPLE_DIR 环境变量指定）")
        return 1
    img_dir = config.SAMPLE_DIR
    lbl_dir = img_dir.parent / "labels"

    all_images = sorted(p for p in img_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if args.sample_mode == "random":
        import random
        images = random.Random(args.seed).sample(all_images, min(args.n, len(all_images)))
        images.sort()
    else:
        images = all_images[:args.n]

    info = verifier_info()
    print("=" * 72)
    print(f"论文方法保真度抽检：{len(images)} / {len(all_images)} 张 test 图"
          f"（抽样方式：{'随机种子 ' + str(args.seed) if args.sample_mode == 'random' else '排序前 N 张'}）")
    print(f"  路由阈值 T   : {args.threshold}")
    print(f"  推理阈值     : {args.infer_conf}")
    print(f"  大模型复核   : {'关闭' if args.no_vlm else info['effective_mode']}"
          f"{'' if args.no_vlm else ' / ' + info['model']}")
    print("=" * 72)

    det = YoloDetector.instance()
    det.warmup()

    agg = {"strict": [0, 0, 0], "agnostic": [0, 0, 0]}
    baseline_agg = {"strict": [0, 0, 0], "agnostic": [0, 0, 0]}
    calls = 0
    t_vlm = 0.0
    t_start = time.perf_counter()

    for i, img_path in enumerate(images, 1):
        img = imread_unicode(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = []
        for cls, (x, y, bw, bh) in load_gt(lbl_dir / f"{img_path.stem}.txt"):
            gt.append((cls, [(x - bw / 2) * w, (y - bh / 2) * h,
                             (x + bw / 2) * w, (y + bh / 2) * h]))

        pipe = Pipeline(threshold=args.threshold, infer_conf=args.infer_conf,
                        enable_vlm=not args.no_vlm, save_crops=False,
                        registry=build_registry(), detector=det)
        res = pipe.run(img)
        calls += res["vlm"]["calls"]
        t_vlm += res["vlm"]["total_latency_ms"]

        final = [{"cls_id": b["cls_id"], "box": b["box"]} for b in res["final_boxes"]]
        base = [{"cls_id": d["cls_id"], "box": d["box"]}
                for d in res["baseline_detections"]]

        for mode, strict in (("strict", True), ("agnostic", False)):
            for k, v in zip(("tp", "fp", "fn"), match(final, gt, strict)):
                agg[mode][{"tp": 0, "fp": 1, "fn": 2}[k]] += v
            for k, v in zip(("tp", "fp", "fn"), match(base, gt, strict)):
                baseline_agg[mode][{"tp": 0, "fp": 1, "fn": 2}[k]] += v

        if i % 10 == 0 or i == len(images):
            el = time.perf_counter() - t_start
            print(f"  [{i}/{len(images)}] 大模型调用 {calls} 次，"
                  f"已用 {el:.0f}s")

    print("\n" + "=" * 72)
    print("结果（框级，IoU=0.5）")
    print("=" * 72)
    print(f"{'方法':<26}{'Precision':>11}{'Recall':>10}{'F1':>9}")
    for mode, label in (("strict", "类别严格（主指标）"), ("agnostic", "类别无关")):
        b = prf(*baseline_agg[mode])
        o = prf(*agg[mode])
        print(f"  [仅 YOLO 基线] {label:<14}{b['P']:>11.4f}{b['R']:>10.4f}{b['F1']:>9.4f}")
        print(f"  [协同推理  ] {label:<14}{o['P']:>11.4f}{o['R']:>10.4f}{o['F1']:>9.4f}")
        if mode == "strict":
            print(f"  -> F1 变化：{b['F1']:.4f} -> {o['F1']:.4f} "
                  f"({o['F1'] - b['F1']:+.4f})\n")

    n = len(images)
    print(f"大模型调用：{calls} 次 / {n} 张 = {calls / n:.2f} 次/张")
    if calls:
        print(f"大模型平均单次延迟：{t_vlm / calls:.0f} ms")
    print(f"总耗时：{time.perf_counter() - t_start:.0f} s")

    print("\n论文定版参考值（全量 300 张，T=0.6）：")
    print("  仅 YOLO 基线 F1(严格) = 0.515 ；协同推理 F1(严格) = 0.599")
    print("  大模型调用 = 0.75 次/张 ；大模型单次延迟 ≈ 1.8~2.0 s")
    print("  ⚠ 抽检子集样本量小，指标会有波动，仅用于验证方法一致性。")

    out = Path(args.out) if args.out else \
        config.BACKEND_DIR / "data" / \
        f"validate_subset_{'novlm' if args.no_vlm else 'real'}_{args.sample_mode}{args.n}.json"
    out.write_text(json.dumps({
        "n_images": n, "n_total": len(all_images), "sample_mode": args.sample_mode,
        "seed": args.seed if args.sample_mode == "random" else None,
        "threshold": args.threshold, "infer_conf": args.infer_conf,
        "vlm_mode": "off" if args.no_vlm else info["effective_mode"],
        "vlm_model": info["model"], "vlm_calls": calls,
        "avg_calls_per_img": round(calls / n, 4) if n else 0,
        "avg_vlm_latency_ms": round(t_vlm / calls) if calls else 0,
        "collab": {m: prf(*agg[m]) for m in agg},
        "baseline": {m: prf(*baseline_agg[m]) for m in baseline_agg},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
