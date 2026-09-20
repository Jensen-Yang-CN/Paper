# -*- coding: utf-8 -*-
"""检测接口：图片上传 / 样例检测 / 本地路径检测。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from .. import config, db, storage
from ..core.detector import DetectorError, imread_unicode
from ..core.pipeline import Pipeline
from ..report import render_report
from ..schemas import DetectParams

router = APIRouter(prefix="/api", tags=["detect"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _params_from_form(
    threshold: float, infer_conf: float, baseline_conf: float,
    enable_vlm: bool, enable_enhance: bool, save_record: bool,
) -> DetectParams:
    try:
        return DetectParams(
            threshold=threshold, infer_conf=infer_conf, baseline_conf=baseline_conf,
            enable_vlm=enable_vlm, enable_enhance=enable_enhance, save_record=save_record,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"参数不合法：{e}")


def _run_and_store(image: np.ndarray, source_name: str, image_url: str,
                   params: DetectParams) -> dict:
    """执行流水线并按需落库，返回完整结果。"""
    try:
        pipe = Pipeline(
            threshold=params.threshold,
            infer_conf=params.infer_conf,
            baseline_conf=params.baseline_conf,
            enable_vlm=params.enable_vlm,
            enable_enhance=params.enable_enhance,
        )
        result = pipe.run(image)
    except DetectorError as e:
        raise HTTPException(status_code=503, detail=str(e))

    result["source_name"] = source_name
    result["image_url"] = image_url
    if params.save_record:
        try:
            result["record_id"] = db.save_record(result, source_name, image_url)
        except Exception as e:      # 落库失败不应影响检测结果返回
            result["record_id"] = None
            result["record_error"] = str(e)[:200]
    return result


@router.post("/detect/upload")
async def detect_upload(
    file: UploadFile = File(..., description="待检测的服装质检图片"),
    threshold: float = Form(0.60),
    infer_conf: float = Form(0.25),
    baseline_conf: float = Form(0.40),
    enable_vlm: bool = Form(True),
    enable_enhance: bool = Form(False),
    save_record: bool = Form(True),
):
    """上传图片并执行协同推理检测。"""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传内容为空")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="图片过大（上限 20 MB）")

    params = _params_from_form(threshold, infer_conf, baseline_conf,
                               enable_vlm, enable_enhance, save_record)

    # 用临时文件解码，兼容中文名与各种格式
    suffix = Path(file.filename or "x.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(data)
        tmp = Path(tf.name)
    try:
        image = imread_unicode(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    if image is None:
        raise HTTPException(status_code=400,
                            detail="无法解析该图片，请上传 JPG / PNG / BMP / WEBP 格式")

    _, name = storage.save_upload(data, file.filename or "upload.jpg")
    return _run_and_store(image, file.filename or name,
                          storage.media_url("uploads", name), params)


@router.post("/detect/sample")
async def detect_sample(
    sample: str = Form(..., description="内置样例图片文件名"),
    threshold: float = Form(0.60),
    infer_conf: float = Form(0.25),
    baseline_conf: float = Form(0.40),
    enable_vlm: bool = Form(True),
    enable_enhance: bool = Form(False),
    save_record: bool = Form(True),
):
    """对内置样例图库中的图片执行检测（新用户零上传体验）。"""
    if config.SAMPLE_DIR is None:
        raise HTTPException(status_code=404, detail="未配置样例图库目录")
    src = (config.SAMPLE_DIR / sample).resolve()
    try:
        src.relative_to(config.SAMPLE_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="非法的样例文件名")
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"样例不存在：{sample}")

    image = imread_unicode(src)
    if image is None:
        raise HTTPException(status_code=400, detail="样例图片解码失败")

    params = _params_from_form(threshold, infer_conf, baseline_conf,
                               enable_vlm, enable_enhance, save_record)
    _, name = storage.copy_to_uploads(src)
    return _run_and_store(image, src.name, storage.media_url("uploads", name), params)


@router.post("/detect/report")
async def export_report(payload: dict):
    """根据已有检测结果导出可视化报告 PNG。"""
    record_id = payload.get("record_id")
    result = payload.get("result")

    if record_id and not result:
        rec = db.get_record(int(record_id))
        if rec is None:
            raise HTTPException(status_code=404, detail="记录不存在")
        result = rec["result_json"]
        image_url = rec["image_url"]
        source_name = rec["source_name"]
    elif result:
        image_url = result.get("image_url", "")
        source_name = result.get("source_name", "")
    else:
        raise HTTPException(status_code=400, detail="请提供 record_id 或 result")

    if not image_url.startswith("/media/"):
        raise HTTPException(status_code=400, detail="图片路径无效")
    kind, _, fname = image_url[len("/media/"):].partition("/")
    img_path = config.DATA_DIR / kind / Path(fname).name
    image = imread_unicode(img_path)
    if image is None:
        raise HTTPException(status_code=404, detail="原图已丢失，无法导出报告")

    png = render_report(image, result, source_name)
    filename = f"clothinspect_report_{result.get('run_id', 'x')}.png"
    return Response(
        content=png, media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
