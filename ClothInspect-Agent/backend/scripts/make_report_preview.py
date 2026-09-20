# -*- coding: utf-8 -*-
"""从数据库里挑一条最丰富的记录，重新渲染报告预览图（开发/文档用）。"""
import io
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image                       # noqa: E402

from app import config                      # noqa: E402
from app.core.detector import imread_unicode  # noqa: E402
from app.report import render_report        # noqa: E402

conn = sqlite3.connect(str(config.DB_PATH))
conn.row_factory = sqlite3.Row
row = conn.execute(
    "SELECT id, result_json, image_url, source_name FROM detections "
    "ORDER BY (n_dropped + n_corrected) DESC, n_vlm_calls DESC LIMIT 1"
).fetchone()

if row is None:
    print("数据库里还没有检测记录，先跑一次检测再执行本脚本。")
    raise SystemExit(1)

res = json.loads(row["result_json"])
kind, _, fname = row["image_url"][len("/media/"):].partition("/")
img = imread_unicode(config.DATA_DIR / kind / fname)
png = render_report(img, res, row["source_name"])

out = config.PROJECT_ROOT / "docs" / "_preview_report.png"
out.write_bytes(png)
size = Image.open(io.BytesIO(png)).size
print("report from record {} ({}): {} bytes {}".format(row["id"], row["source_name"], len(png), size))
