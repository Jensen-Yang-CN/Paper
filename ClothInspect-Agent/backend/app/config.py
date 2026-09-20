# -*- coding: utf-8 -*-
"""
全局配置
========
ClothInspect-Agent 服装智能质检系统 V1.0

配置来源优先级：环境变量 / .env 文件 > 本文件默认值。
所有路径均提供自动探测，保证工程换机器后仍能启动。
"""
from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------- .env 读取
def _load_dotenv() -> None:
    """极简 .env 加载器（不引入额外依赖）。已存在的环境变量不会被覆盖。"""
    for candidate in (PROJECT_ROOT / ".env", PROJECT_ROOT / "backend" / ".env"):
        if not candidate.exists():
            continue
        try:
            for raw in candidate.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
        except Exception:  # .env 损坏不应阻塞启动
            pass


# ---------------------------------------------------------------- 目录常量
BACKEND_DIR = Path(__file__).resolve().parent.parent          # backend/
PROJECT_ROOT = BACKEND_DIR.parent                             # ClothInspect-Agent/
PAPER_ROOT = PROJECT_ROOT.parent                              # Paper/

_load_dotenv()

DATA_DIR = BACKEND_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CROP_DIR = DATA_DIR / "crops"
REPORT_DIR = DATA_DIR / "reports"
DB_PATH = Path(os.environ.get("DB_PATH", DATA_DIR / "clothinspect.db"))

ASSETS_DIR = BACKEND_DIR / "assets"
PAPER_ASSETS_DIR = ASSETS_DIR / "paper"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

for _d in (DATA_DIR, UPLOAD_DIR, CROP_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ultralytics 会往用户目录写配置，受限环境下会报权限错——固定到工程内可写目录
_YOLO_CFG_DIR = DATA_DIR / "ultralytics"
_YOLO_CFG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(_YOLO_CFG_DIR)


# ---------------------------------------------------------------- 模型权重
def _resolve_weights() -> Path | None:
    """按优先级探测 YOLOv11 权重文件。"""
    candidates: list[Path] = []
    if os.environ.get("YOLO_WEIGHTS"):
        candidates.append(Path(os.environ["YOLO_WEIGHTS"]))
    candidates += [
        PAPER_ROOT / "模型权重" / "Final_Model_V6_best.pt",
        BACKEND_DIR / "assets" / "weights" / "Final_Model_V6_best.pt",
        BACKEND_DIR / "assets" / "weights" / "best.pt",
    ]
    candidates += sorted((BACKEND_DIR / "assets" / "weights").glob("*.pt")) \
        if (BACKEND_DIR / "assets" / "weights").exists() else []
    for c in candidates:
        if c.exists():
            return c
    return None


YOLO_WEIGHTS: Path | None = _resolve_weights()


# ---------------------------------------------------------------- 样例图库
def _resolve_sample_dir() -> Path | None:
    """内置样例图库（重标 test 集），用于新用户零上传体验。"""
    if os.environ.get("SAMPLE_DIR"):
        p = Path(os.environ["SAMPLE_DIR"])
        if p.exists():
            return p
    for c in (
        PAPER_ROOT / "实验数据" / "relabeled_test" / "images",
        BACKEND_DIR / "assets" / "samples",
    ):
        if c.exists():
            return c
    return None


SAMPLE_DIR: Path | None = _resolve_sample_dir()


# ---------------------------------------------------------------- 算法超参
# 类别定义（与论文数据集严格一致，顺序即索引）
CLASS_NAMES = ["hole", "stain", "wrinkle"]
CLASS_LABELS_ZH = {"hole": "破洞", "stain": "污渍", "wrinkle": "褶皱"}

# 路由阈值 T：置信度 >= T 的检测框直接接受；< T 的送 VLM 复核
DEFAULT_ROUTER_THRESHOLD = float(os.environ.get("ROUTER_THRESHOLD", 0.60))
# YOLO 推理下限：低于此置信度的候选直接丢弃（不进入后续流程）
DEFAULT_INFER_CONF = float(os.environ.get("INFER_CONF", 0.25))
# 论文基线阈值：仅用于"仅 YOLO"对照展示
BASELINE_CONF = float(os.environ.get("BASELINE_CONF", 0.40))

# 局部裁剪：外扩比例 & 最小边尺寸
CROP_MARGIN = float(os.environ.get("CROP_MARGIN", 0.30))
MIN_CROP = int(os.environ.get("MIN_CROP", 96))

# 框级匹配 IoU 阈值
IOU_THRESHOLD = 0.5

IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))
DEVICE = os.environ.get("YOLO_DEVICE", "0")   # "0" = 第一块 GPU；"cpu" = CPU


# ---------------------------------------------------------------- VLM 配置
VLM_PROVIDER = os.environ.get("VLM_PROVIDER", "dashscope")
VLM_API_URL = os.environ.get(
    "VLM_API_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
)
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen3-vl-plus")
VLM_API_KEY = (
    os.environ.get("DASHSCOPE_API_KEY")
    or os.environ.get("VLM_API_KEY")
    or ""
).strip()
VLM_TIMEOUT = int(os.environ.get("VLM_TIMEOUT", 60))
VLM_TEMPERATURE = float(os.environ.get("VLM_TEMPERATURE", 0.2))

# 复核模式：
#   auto      —— 有密钥走真实 qwen3-vl-plus，无密钥自动降级为离线启发式复核（推荐）
#   real      —— 强制真实调用，无密钥时报错
#   heuristic —— 强制离线启发式（无网络演示 / 答辩彩排）
VLM_MODE = os.environ.get("VLM_MODE", "auto").strip().lower()


def vlm_effective_mode() -> str:
    """解析 VLM_MODE，返回实际生效模式：real | heuristic。"""
    if VLM_MODE == "real":
        return "real"
    if VLM_MODE == "heuristic":
        return "heuristic"
    return "real" if VLM_API_KEY else "heuristic"


# ---------------------------------------------------------------- 服务配置
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 8000))

SYSTEM_NAME = "ClothInspect-Agent"
SYSTEM_NAME_ZH = "服装智能质检系统"
SYSTEM_VERSION = "V1.0"
