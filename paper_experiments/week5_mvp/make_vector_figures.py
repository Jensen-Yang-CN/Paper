# -*- coding: utf-8 -*-
"""导出《计算机工程与应用》要求的图包：坐标图/框架图给可编辑矢量图（PDF/EPS），
图像类图给不小于 300 dpi 的位图（TIF/PNG）。

绘图代码复用 make_bw_figures.py（不重写、不改动原脚本），
只把 plt.savefig 重定向到本目录下的“图包_矢量图”文件夹。

用法：
    python make_vector_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import make_bw_figures as M

HERE = Path(__file__).resolve().parent
PKG = HERE / "图包_矢量图"
PKG.mkdir(exist_ok=True)

VECTOR = {"fig1_framework_bw", "fig2_h1_bw", "fig3_pareto_bw"}
_saved = []


def _savefig(path, **kw):
    """替换 plt.savefig：矢量图存 PDF+EPS（同时保留 600 dpi PNG）。"""
    stem = Path(path).stem
    fig = plt.gcf()
    fig.savefig(PKG / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    if stem in VECTOR:
        for ext in ("pdf", "eps"):
            fig.savefig(PKG / f"{stem}.{ext}", format=ext,
                        bbox_inches="tight", pad_inches=0.02)
    _saved.append(stem)


plt.savefig = _savefig

if __name__ == "__main__":
    M.fig_framework()
    M.fig_h1()
    M.fig_pareto()
    print("矢量图已导出:", ", ".join(_saved))

    # 图 4 为图像类插图：整幅图直接来自 600 dpi 位图，另存 TIF 供排版使用
    src = M.OUT / "fig4_error_cases_bw.png"
    im = Image.open(src)
    tif = PKG / "fig4_error_cases_bw.tif"
    im.save(tif, dpi=(600, 600), compression="tiff_lzw")
    im.save(PKG / "fig4_error_cases_bw.png", dpi=(600, 600))
    print("位图已导出: fig4_error_cases_bw.tif  %.0f x %.0f px, dpi=%s"
          % (im.size[0], im.size[1], im.info.get("dpi")))
    print("\n图包目录:", PKG)
    for f in sorted(PKG.iterdir()):
        print("   %-28s %8.1f KB" % (f.name, f.stat().st_size / 1024))
