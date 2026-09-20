# -*- coding: utf-8 -*-
"""
后端冒烟测试
============
不启动服务，直接跑一遍核心链路，用于快速自检环境是否配好。

    python scripts/smoke_test.py                # 用内置样例跑一（默认阈值 0.6）
    python scripts/smoke_test.py --sample v00001.jpg --threshold 0.5
    python scripts/smoke_test.py --list-samples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config                                   # noqa: E402
from app.core.agent_tools import build_registry          # noqa: E402
from app.core.detector import YoloDetector, imread_unicode  # noqa: E402
from app.core.pipeline import Pipeline                   # noqa: E402
from app.core.vlm_client import verifier_info            # noqa: E402
from app.api.system import build_sample_library           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="", help="样例文件名；留空则自动选一张含缺陷的")
    ap.add_argument("--image", default="", help="直接指定图片绝对路径")
    ap.add_argument("--threshold", type=float, default=config.DEFAULT_ROUTER_THRESHOLD)
    ap.add_argument("--infer-conf", type=float, default=config.DEFAULT_INFER_CONF)
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--list-samples", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("环境自检")
    print("=" * 70)
    print(f"  权重路径   : {config.YOLO_WEIGHTS}")
    print(f"  样例目录   : {config.SAMPLE_DIR}")
    info = verifier_info()
    print(f"  复核后端   : {info['effective_mode']}  ({info['model']})")
    print(f"  说明       : {info['note']}")
    print(f"  设备       : {config.DEVICE}")

    if args.list_samples:
        lib = build_sample_library()
        print(f"\n内置样例 {len(lib)} 张：")
        for it in lib:
            print(f"   {it['file']:<16} {it['label']:<12} {it['label_zh']}")
        return 0

    # ---------------- 选图 ----------------
    if args.image:
        img_path = Path(args.image)
    else:
        if config.SAMPLE_DIR is None:
            print("[错误] 未配置样例图库，请用 --image 指定图片")
            return 1
        name = args.sample
        if not name:
            lib = build_sample_library()
            cand = [x for x in lib if x["n_defect"] > 0] or lib
            if not cand:
                print("[错误] 样例库为空")
                return 1
            name = cand[0]["file"]
        img_path = config.SAMPLE_DIR / name

    if not img_path.exists():
        print(f"[错误] 图片不存在: {img_path}")
        return 1

    image = imread_unicode(img_path)
    if image is None:
        print(f"[错误] 图片解码失败: {img_path}")
        return 1
    print(f"\n测试图片   : {img_path.name}  {image.shape[1]}x{image.shape[0]}")

    # ---------------- 检测器 ----------------
    print("\n" + "=" * 70)
    print("加载 YOLOv11 ...")
    det = YoloDetector.instance()
    det.warmup()
    print("  加载完成")

    # ---------------- 流水线 ----------------
    print("\n" + "=" * 70)
    print(f"协同推理（T={args.threshold}, 推理阈值={args.infer_conf}）")
    print("=" * 70)
    pipe = Pipeline(threshold=args.threshold, infer_conf=args.infer_conf,
                    enable_vlm=not args.no_vlm, registry=build_registry(),
                    detector=det)
    result = pipe.run(image)

    r = result["routing"]
    print(f"\n[路由] 候选 {r['total']} 框 → 直接接受 {r['accepted']}，"
          f"转复核 {r['verified']}（调用比 {r['vlm_call_ratio']:.0%}）")
    print(f"[检测] YOLO 输出 {len(result['detections'])} 框，耗时 {result['timing']['yolo_ms']} ms")
    print(f"[复核] 调用 {result['vlm']['calls']} 次，"
          f"平均 {result['vlm']['avg_latency_ms']} ms，模式 {result['vlm']['mode']}")
    f = result["fusion"]
    print(f"[融合] 最终 {len(result['final_boxes'])} 框，"
          f"剔除误检 {f['dropped_by_vlm']}，类别纠错 {f['class_corrected']}")
    print(f"[总耗时] {result['timing']['total_ms']} ms")

    print("\n--- 最终缺陷框 ---")
    if not result["final_boxes"]:
        print("   （未检出缺陷）")
    for i, b in enumerate(result["final_boxes"], 1):
        tag = ""
        if b["class_corrected"]:
            tag = f"  ← 类别纠错（YOLO 原判 {b['yolo_cls_name']}）"
        print(f"   #{i} {config.CLASS_LABELS_ZH.get(b['cls_name'], b['cls_name'])} "
              f"conf={b['conf']:.3f} box={[round(v) for v in b['box']]} "
              f"[{b['source']}]{tag}")

    print("\n--- 推理轨迹 ---")
    for t in result["trace"]:
        flag = {"ok": "✓", "error": "✗", "skipped": "-"}.get(t["status"], "?")
        print(f"   {t['step']}. {flag} [{t['label']}] {t['summary']} ({t['latency_ms']} ms)")

    print("\n冒烟测试通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
