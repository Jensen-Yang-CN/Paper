# -*- coding: utf-8 -*-
# 重画错误案例图为黑白版：实线=真实标注，虚线=检测输出；每幅局部放大到错误区域
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from ultralytics import YOLO

matplotlib.rcParams["font.sans-serif"] = ["SimSun", "宋体", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 6.5
matplotlib.rcParams["text.color"] = "black"
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["xtick.color"] = matplotlib.rcParams["ytick.color"] = "black"

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week2_test_relabel")
CSV = BASE / "error_cases" / "error_cases.csv"
DATA = Path(r"E:\CodeBase_YangJunJie\Paper\实验数据\relabeled_test")
MODEL = r"E:\CodeBase_YangJunJie\ClothingInspection\runs\detect\Final_Model_V6\weights\best.pt"
OUT = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week5_mvp\fig4_error_cases_bw.png")
CONF, IOU_THR = 0.5, 0.5
NAME = {0: "hole", 1: "stain", 2: "wrinkle"}
TARGETS = [("missed_hole", "v00146.jpg"), ("missed_stain", "v00012.jpg"),
           ("missed_wrinkle", "v00003.jpg"), ("false_positive", "v00004.jpg")]
LAB = {"missed_hole": "(a) 漏检破洞", "missed_stain": "(b) 漏检污渍",
       "missed_wrinkle": "(c) 漏检褶皱", "false_positive": "(d) 误检"}


def imread_unicode(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def load_gt(lbl):
    out = []
    if lbl.exists():
        for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split()
            if len(p) == 5:
                out.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def analyse(model, img_name):
    """返回 (原图, 真值框列表, 检测框列表, 目标框, 类型)"""
    img = imread_unicode(DATA / "images" / img_name)
    h, w = img.shape[:2]
    gts = []
    for cls, x, y, bw, bh in load_gt(DATA / "labels" / (Path(img_name).stem + ".txt")):
        gts.append((cls, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
    res = model.predict(source=img, conf=CONF, iou=0.5, imgsz=640, device="0", verbose=False)
    dets = []
    if res and res[0].boxes is not None:
        for b in res[0].boxes:
            dets.append((int(b.cls[0]), [float(v) for v in b.xyxy[0].tolist()], float(b.conf[0])))
    matched_gt = [False] * len(gts)
    matched_det = [False] * len(dets)
    for i, (_, box, _) in enumerate(dets):
        bj, bv = -1, IOU_THR
        for j, (_, gbox) in enumerate(gts):
            if matched_gt[j]:
                continue
            v = iou(box, gbox)
            if v > bv:
                bv, bj = v, j
        if bj >= 0:
            matched_gt[bj] = True
            matched_det[i] = True
    return img, gts, dets, matched_gt, matched_det


def pick_target(kind, img_name, model):
    img, gts, dets, m_gt, m_det = analyse(model, img_name)
    if kind.startswith("missed"):
        want = NAME_CLS = kind.split("_", 1)[1]
        for j, (gcls, gbox) in enumerate(gts):
            if not m_gt[j] and NAME[gcls] == want:
                return img, gts, dets, gbox, j
    else:
        for i, (cls, box, conf) in enumerate(dets):
            if not m_det[i] and not any(iou(box, gb) >= 0.3 for _, gb in gts):
                return img, gts, dets, box, -1
    return None


def crop_around(img, box, scale=2.6, min_side=200):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    side = max(side, min_side)
    h, w = img.shape[:2]
    ax1, ay1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    ax2, ay2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    return img[ay1:ay2, ax1:ax2], ax1, ay1


def main():
    model = YOLO(MODEL)
    fig, axes = plt.subplots(2, 2, figsize=(3.15, 3.05))
    axes = axes.ravel()
    for k, (kind, img_name) in enumerate(TARGETS):
        got = pick_target(kind, img_name, model)
        ax = axes[k]
        if got is None:
            ax.axis("off")
            continue
        img, gts, dets, tbox, tidx = got
        patch, ox, oy = crop_around(img, tbox)
        g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        ax.imshow(g, cmap="gray", vmin=0, vmax=255)

        def rect(box, **kw):
            x1, y1, x2, y2 = box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, **kw))

        for j, (_, gb) in enumerate(gts):
            if j == tidx:
                continue
            rect(gb, ec="black", lw=0.45, ls="-")
        for _, db, _ in dets:
            rect(db, ec="black", lw=0.7, ls="--")
        if tidx >= 0:
            rect(gts[tidx][1], ec="black", lw=1.5, ls="-")
        else:
            for _, db, _ in dets:
                if iou(db, tbox) > 0.99:
                    rect(db, ec="black", lw=1.5, ls="--")

        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5)
        ax.set_title(LAB[kind], fontsize=6.5, pad=2)
    handles = [Line2D([], [], color="black", lw=1.0, ls="-", label="真实标注"),
               Line2D([], [], color="black", lw=1.0, ls="--", label="检测输出")]
    axes[0].legend(handles=handles, fontsize=5.8, loc="upper left", framealpha=0.85,
                   borderpad=0.25, handlelength=1.5, labelspacing=0.2)
    plt.tight_layout(pad=0.25)
    plt.savefig(OUT, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    from PIL import Image
    with Image.open(OUT) as im:
        print("图4 尺寸:", im.size, round(im.size[0] / 600 * 2.54, 2), "cm")


if __name__ == "__main__":
    main()
