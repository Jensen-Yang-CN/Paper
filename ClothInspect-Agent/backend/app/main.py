# -*- coding: utf-8 -*-
"""
ClothInspect-Agent 服装智能质检系统 V1.0 —— 后端服务入口
=========================================================
启动：
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
或直接：
    python run.py

接口文档：http://127.0.0.1:8000/docs
前端页面：http://127.0.0.1:8000/  （已构建 dist 时自动托管）
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, db
from .api import detect, experiments, history, system
from .core.detector import DetectorError, YoloDetector
from .core.vlm_client import verifier_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("clothinspect")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动 / 关闭钩子。"""
    db.init_db()
    log.info("数据库就绪: %s", config.DB_PATH)

    info = verifier_info()
    if info["is_real"]:
        log.info("多模态复核后端: 真实调用 %s", info["model"])
    else:
        log.warning("多模态复核后端: 离线启发式（%s）", info["note"])

    st = YoloDetector.status()
    if st["weights_found"]:
        log.info("YOLOv11 权重: %s", st["weights_path"])
        try:
            YoloDetector.instance().warmup()
            log.info("YOLOv11 预热完成，推理设备: %s", config.DEVICE)
        except DetectorError as e:
            log.error("YOLOv11 加载失败: %s", e)
        except Exception as e:                       # noqa: BLE001
            log.error("YOLOv11 预热异常: %s", e)
    else:
        log.error("未找到 YOLOv11 权重，检测接口将返回 503。"
                  "请把 Final_Model_V6_best.pt 放入 backend/assets/weights/ "
                  "或设置环境变量 YOLO_WEIGHTS")

    if config.SAMPLE_DIR:
        log.info("内置样例图库: %s", config.SAMPLE_DIR)
    else:
        log.warning("未配置样例图库（可用 SAMPLE_DIR 环境变量指定）")

    log.info("%s %s 已就绪 → http://%s:%d",
             config.SYSTEM_NAME, config.SYSTEM_VERSION, config.HOST, config.PORT)
    yield
    log.info("服务已停止")


app = FastAPI(
    title=f"{config.SYSTEM_NAME} {config.SYSTEM_NAME_ZH} {config.SYSTEM_VERSION}",
    description=(
        "视觉模型与多模态大模型协同推理的服装表面微痕检测系统。\n\n"
        "核心流程：YOLOv11 检测 → 置信度路由 → 局部裁剪/增强 → "
        "多模态大模型复核 → 结果融合。"
    ),
    version=config.SYSTEM_VERSION,
    lifespan=lifespan,
)

# 开发时前端跑在 5173，允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(detect.router)
app.include_router(history.router)
app.include_router(experiments.router)

# 媒体文件：上传原图 / 局部裁剪图 / 导出报告
app.mount("/media/uploads", StaticFiles(directory=str(config.UPLOAD_DIR)), name="uploads")
app.mount("/media/crops", StaticFiles(directory=str(config.CROP_DIR)), name="crops")
app.mount("/media/reports", StaticFiles(directory=str(config.REPORT_DIR)), name="reports")


# ---------------------------------------------------------------- 异常处理
@app.exception_handler(DetectorError)
async def detector_error_handler(request: Request, exc: DetectorError):
    return JSONResponse(status_code=503, content={
        "detail": str(exc),
        "hint": "请在 backend/assets/weights/ 放置 YOLOv11 权重，或设置 YOLO_WEIGHTS 环境变量。",
    })


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={
        "detail": "请求参数校验失败",
        "errors": exc.errors()[:8],
    })


# ---------------------------------------------------------------- 前端托管
if config.FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(config.FRONTEND_DIST / "assets")),
              name="frontend-assets")

    @app.get("/", include_in_schema=False)
    def spa_root():
        return FileResponse(str(config.FRONTEND_DIST / "index.html"))

    @app.get("/{path:path}", include_in_schema=False)
    def spa_fallback(path: str):
        """Vue Router history 模式回退。"""
        if path.startswith(("api/", "media/", "docs", "openapi.json", "redoc")):
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        candidate = config.FRONTEND_DIST / path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(config.FRONTEND_DIST / "index.html"))
else:
    @app.get("/", include_in_schema=False)
    def dev_hint():
        return {
            "message": f"{config.SYSTEM_NAME} 后端已启动，但前端尚未构建。",
            "frontend_build": "cd frontend && npm install && npm run build",
            "frontend_dev": "cd frontend && npm run dev  （开发模式，访问 http://localhost:5173）",
            "api_docs": "/docs",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, reload=False)
