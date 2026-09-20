# -*- coding: utf-8 -*-
"""
检测记录模块（结果历史）
========================
软著功能模块之一：检测记录。使用 SQLite 单文件库，零运维依赖。

每条记录保存：原图路径、参数、结论摘要、完整结果 JSON、耗时与时间戳。
列表查询只取摘要字段（避免把大 JSON 全量拉回前端），详情按 id 取全量。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from typing import Any

from . import config

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    NOT NULL,
    created_at    TEXT    NOT NULL,
    source_name   TEXT    NOT NULL,
    image_url     TEXT    NOT NULL,
    image_width   INTEGER NOT NULL DEFAULT 0,
    image_height  INTEGER NOT NULL DEFAULT 0,
    threshold     REAL    NOT NULL DEFAULT 0.6,
    infer_conf    REAL    NOT NULL DEFAULT 0.25,
    vlm_mode      TEXT    NOT NULL DEFAULT 'heuristic',
    vlm_model     TEXT    NOT NULL DEFAULT '',
    n_detections  INTEGER NOT NULL DEFAULT 0,
    n_final       INTEGER NOT NULL DEFAULT 0,
    n_vlm_calls   INTEGER NOT NULL DEFAULT 0,
    n_dropped     INTEGER NOT NULL DEFAULT 0,
    n_corrected   INTEGER NOT NULL DEFAULT 0,
    class_counts  TEXT    NOT NULL DEFAULT '{}',
    elapsed_ms    INTEGER NOT NULL DEFAULT 0,
    result_json   TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_detections_created ON detections(created_at DESC);
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.DB_PATH), timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _LOCK, _conn() as conn:
        conn.executescript(SCHEMA)


def save_record(result: dict, source_name: str, image_url: str) -> int:
    """写入一条检测记录，返回记录 id。"""
    counts: dict[str, int] = {}
    for b in result.get("final_boxes", []):
        counts[b["cls_name"]] = counts.get(b["cls_name"], 0) + 1

    row = {
        "run_id": result.get("run_id", ""),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_name": source_name,
        "image_url": image_url,
        "image_width": result.get("image", {}).get("width", 0),
        "image_height": result.get("image", {}).get("height", 0),
        "threshold": result.get("params", {}).get("threshold", 0.0),
        "infer_conf": result.get("params", {}).get("infer_conf", 0.0),
        "vlm_mode": result.get("vlm", {}).get("mode", ""),
        "vlm_model": result.get("vlm", {}).get("model", ""),
        "n_detections": len(result.get("detections", [])),
        "n_final": len(result.get("final_boxes", [])),
        "n_vlm_calls": result.get("vlm", {}).get("calls", 0),
        "n_dropped": result.get("fusion", {}).get("dropped_by_vlm", 0),
        "n_corrected": result.get("fusion", {}).get("class_corrected", 0),
        "class_counts": json.dumps(counts, ensure_ascii=False),
        "elapsed_ms": result.get("timing", {}).get("total_ms", 0),
        "result_json": json.dumps(result, ensure_ascii=False),
    }
    cols = ", ".join(row)
    ph = ", ".join("?" for _ in row)
    with _LOCK, _conn() as conn:
        cur = conn.execute(f"INSERT INTO detections ({cols}) VALUES ({ph})",
                           list(row.values()))
        return int(cur.lastrowid)


_LIST_COLS = (
    "id, run_id, created_at, source_name, image_url, image_width, image_height, "
    "threshold, infer_conf, vlm_mode, vlm_model, n_detections, n_final, "
    "n_vlm_calls, n_dropped, n_corrected, class_counts, elapsed_ms"
)


def list_records(page: int = 1, size: int = 20, keyword: str = "") -> dict:
    """分页查询记录摘要。"""
    page = max(1, int(page))
    size = max(1, min(100, int(size)))
    where, args = "", []
    if keyword:
        where = "WHERE source_name LIKE ?"
        args.append(f"%{keyword}%")

    with _LOCK, _conn() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM detections {where}", args).fetchone()[0]
        rows = conn.execute(
            f"SELECT {_LIST_COLS} FROM detections {where} "
            f"ORDER BY id DESC LIMIT ? OFFSET ?",
            [*args, size, (page - 1) * size]).fetchall()

    items = []
    for r in rows:
        d = dict(r)
        try:
            d["class_counts"] = json.loads(d.get("class_counts") or "{}")
        except Exception:
            d["class_counts"] = {}
        items.append(d)
    return {"total": total, "page": page, "size": size, "items": items}


def get_record(record_id: int) -> dict | None:
    """取单条记录全量（含结果 JSON）。"""
    with _LOCK, _conn() as conn:
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (record_id,)).fetchone()
    if row is None:
        return None
    d: dict[str, Any] = dict(row)
    for k in ("result_json", "class_counts"):
        try:
            d[k] = json.loads(d.get(k) or "{}")
        except Exception:
            d[k] = {}
    return d


def delete_record(record_id: int) -> bool:
    with _LOCK, _conn() as conn:
        cur = conn.execute("DELETE FROM detections WHERE id = ?", (record_id,))
        return cur.rowcount > 0


def clear_records() -> int:
    with _LOCK, _conn() as conn:
        cur = conn.execute("DELETE FROM detections")
        return cur.rowcount


def statistics() -> dict:
    """记录总览统计，用于历史页顶部卡片。"""
    with _LOCK, _conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n, "
            "COALESCE(SUM(n_final),0) AS boxes, "
            "COALESCE(SUM(n_vlm_calls),0) AS calls, "
            "COALESCE(SUM(n_dropped),0) AS dropped, "
            "COALESCE(SUM(n_corrected),0) AS corrected, "
            "COALESCE(AVG(elapsed_ms),0) AS avg_ms "
            "FROM detections").fetchone()
        by_class = conn.execute(
            "SELECT class_counts FROM detections").fetchall()
    merged: dict[str, int] = {}
    for r in by_class:
        try:
            for k, v in json.loads(r["class_counts"] or "{}").items():
                merged[k] = merged.get(k, 0) + int(v)
        except Exception:
            pass
    return {
        "total_records": row["n"],
        "total_boxes": row["boxes"],
        "total_vlm_calls": row["calls"],
        "total_dropped": row["dropped"],
        "total_corrected": row["corrected"],
        "avg_elapsed_ms": int(row["avg_ms"] or 0),
        "class_counts": merged,
    }
