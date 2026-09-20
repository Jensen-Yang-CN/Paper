# -*- coding: utf-8 -*-
"""检测记录接口（结果历史）。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .. import db

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("")
def list_history(page: int = Query(1, ge=1), size: int = Query(20, ge=1, le=100),
                 keyword: str = ""):
    """分页查询检测记录摘要。"""
    return db.list_records(page=page, size=size, keyword=keyword)


@router.get("/statistics")
def history_statistics():
    """历史总览统计。"""
    return db.statistics()


@router.get("/{record_id}")
def get_history(record_id: int):
    """取单条记录全量结果（含检测框与推理轨迹）。"""
    rec = db.get_record(record_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="记录不存在")
    return rec


@router.delete("/{record_id}")
def delete_history(record_id: int):
    ok = db.delete_record(record_id)
    if not ok:
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"ok": True, "deleted": record_id}


@router.delete("")
def clear_history():
    n = db.clear_records()
    return {"ok": True, "deleted": n}
