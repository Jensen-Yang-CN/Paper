# -*- coding: utf-8 -*-
# 画 Pareto 曲线：VLM 调用率 vs 严格口径 F1（阈值扫描）
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week4_experiments")
data = json.loads((BASE / "week4_results.json").read_text(encoding="utf-8"))
sweep = data["sweep"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：调用率 vs F1
x = [s["call_rate"] for s in sweep]
y = [s["F1"] for s in sweep]
axes[0].plot(x, y, "o-", color="#4c8bf5", lw=2)
for s in sweep:
    axes[0].annotate(f"T={s['T']:.1f}", (s["call_rate"], s["F1"]), textcoords="offset points", xytext=(0, 8), fontsize=9)
axes[0].set_xlabel("VLM Calls per Image")
axes[0].set_ylabel("Strict F1")
axes[0].set_title("(a) Cost-Performance Pareto (routing)")
axes[0].grid(alpha=0.3)

# 右：调用率 vs 各指标
r = [s["R"] for s in sweep]
p = [s["P"] for s in sweep]
axes[1].plot(x, y, "o-", label="F1", color="#4c8bf5")
axes[1].plot(x, r, "s--", label="Recall", color="#27ae60")
axes[1].plot(x, p, "^--", label="Precision", color="#e74c3c")
axes[1].set_xlabel("VLM Calls per Image")
axes[1].set_ylabel("Strict Metric")
axes[1].set_title("(b) Routing: metrics vs call rate")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
out = BASE / "pareto_curve.png"
plt.savefig(out, dpi=200)
print("已保存:", out)
print("Pareto 点:", [(s["T"], s["call_rate"], s["F1"]) for s in sweep])
