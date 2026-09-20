# -*- coding: utf-8 -*-
"""
一键启动后端服务
================
    python run.py                 # 使用 .env / 环境变量中的配置
    python run.py --port 8001     # 指定端口
    python run.py --reload        # 开发热重载
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="ClothInspect-Agent 后端服务")
    ap.add_argument("--host", default=config.HOST)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = ap.parse_args()

    # 让 app 内的日志与文档使用真实的监听地址（config 在导入时已读过默认值）
    config.HOST = args.host
    config.PORT = args.port

    try:
        import uvicorn
    except ImportError:
        print("[错误] 缺少 uvicorn，请先执行：pip install -r requirements.txt")
        return 1

    print("=" * 66)
    print(f"  {config.SYSTEM_NAME} {config.SYSTEM_NAME_ZH} {config.SYSTEM_VERSION}")
    print("=" * 66)
    print(f"  YOLOv11 权重 : {config.YOLO_WEIGHTS or '【未找到】'}")
    print(f"  样例图库     : {config.SAMPLE_DIR or '【未配置】'}")
    print(f"  复核后端     : {config.vlm_effective_mode()} "
          f"({config.VLM_MODEL if config.vlm_effective_mode() == 'real' else '离线启发式'})")
    print(f"  服务地址     : http://{args.host}:{args.port}")
    print(f"  接口文档     : http://{args.host}:{args.port}/docs")
    print("=" * 66)

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
