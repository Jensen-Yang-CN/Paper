# 服装微痕检测的视觉与大模型协同推理方法

**Collaborative Inference between a Vision Model and a Multimodal Large Model for Garment Micro-defect Detection**

针对服装表面破洞、污渍、褶皱三类微痕检测中"专用检测模型定位准但疑难样本漏检多、多模态大模型语义判断强但定位粗"的能力互补性，本仓库给出一套**推理期协同**方案及其完整实验证据与可运行系统。

> 核心主张：**不重训练检测器**，在系统层面以检测置信度为信号，把有限的大模型复核预算分配到最值得复核的候选框上。

---

## 1. 核心结果

主实验在标注修正后的 300 幅测试图像（849 个缺陷框）上进行，框级匹配 IoU=0.5，以**类别严格**口径为主指标。

| 方法 | P(严格) | R(严格) | **F1(严格)** | F1(无关) | 图像级 F1 | 大模型调用/幅 |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv11s 基线（conf=0.40） | 0.563 | 0.475 | 0.515 | 0.787 | 0.908 | 0 |
| 低阈值控制组（conf=0.25） | 0.553 | 0.528 | 0.540 | 0.826 | 0.907 | 0 |
| 全复核（逐框裁剪复核） | **0.815** | **0.703** | **0.755** | **0.858** | **0.954** | 2.70 |
| **本文方法（T=0.60，推荐操作点）** | 0.627 | 0.572 | **0.599** | 0.845 | 0.923 | **0.75** |
| 本文方法（T=0.70） | 0.660 | 0.590 | 0.623 | 0.846 | 0.924 | 1.30 |

**诚实结论（请勿误读）**：全复核 0.755 是性能上限；置信度路由不是"超过上限"，而是在**调用预算约束下逼近上限**的 Pareto 权衡。相比"直接降低检测阈值"的控制组（0.540），本文方法的提升（0.599）来自大模型语义复核，而非阈值下移。

### 阈值扫描（成本-性能 Pareto）

| 路由阈值 T | 0.30 | 0.40 | 0.50 | **0.60** | 0.70 |
|---|---:|---:|---:|---:|---:|
| 大模型调用/幅 | 0.12 | 0.31 | 0.48 | **0.75** | 1.30 |
| F1(严格) | 0.549 | 0.562 | 0.576 | **0.599** | 0.623 |
| 边际收益（F1/次调用） | 0.068 | 0.082 | 0.085 | **0.085** | 0.044 |

边际收益在 T=0.60 达峰后明显下降，故取其为默认操作点。

### 假设验证

**H1 置信度可作为路由信号**（图 3）：置信度 < 0.5 的检测平均误检率约 25%，≥ 0.6 的约 12%，且在 0.6~0.9 区间单调下降。

**H2 局部裁剪是有效视觉证据**（表 5）：同一批 810 个检测框，仅改变大模型输入形式：

| 大模型输入 | 准确率 | 精确率 | 召回率 | F1 | 真负例 |
|---|---:|---:|---:|---:|---:|
| 裁剪图（含尺度提示） | **0.643** | **0.609** | 0.996 | **0.755** | **75** |
| 整图 | 0.598 | 0.579 | 0.993 | 0.732 | 39 |

提升主要来自**拒绝假缺陷**，召回基本不变。

**按类别难度**：召回增益集中在最难的褶皱类别。

| 方法 | 破洞 | 污渍 | 褶皱 |
|---|---:|---:|---:|
| YOLO（conf=0.40） | 0.605 | 0.621 | 0.278 |
| 控制组（conf=0.25） | 0.699 | 0.629 | 0.304 |
| **本文方法（T=0.60）** | 0.691 | 0.644 | **0.417** |

### 效率

| 项 | 数值 |
|---|---|
| YOLO 单幅推理 | 约 0.027 s |
| qwen3-vl-plus 单次复核 | 约 1.82 s |
| 本文方法平均调用 | 0.75 次/幅 |
| 本文方法平均大模型耗时 | 约 1.37 s/幅 |

---

## 2. 方法

```
输入图像
   │
   ▼
YOLOv11s 检测（权重冻结，conf=0.25）──► 候选框 {类别, 边框, 置信度 C}
   │
   ▼
置信度路由（阈值 T）        C ≥ T ──► 直接接受 YOLO 结果
   │
   │ C < T
   ▼
局部裁剪（外扩 30%，最小 96 px）+ 尺度提示（原图尺寸 / 裁剪尺寸 / 面积占比）
   │
   ▼
qwen3-vl-plus 复核 ──► {"is_defect": bool, "category": str, "confidence": float, "reason": str}
   │
   ▼
结果融合：保留 / 剔除；类别以复核结果为准（类别纠错），边框沿用 YOLO 的精确定位
   │
   ▼
最终缺陷框
```

设计要点：

- **检测器权重全程冻结**，只改推理流程，反映真实部署条件下的改进路径；
- **类别来自大模型、坐标来自检测器**，各取所长；
- 路由阈值 T 是唯一的成本旋钮，可直接映射到算力预算。

方法渊源与级联推理、learning-to-defer、选择性分类的思想一致，详见论文第 1.3 节。

---

## 3. 目录结构

```
Paper/
├── 论文初稿_v6.md                  ★ 论文（可编辑源文件）
├── 论文初稿_v6_模板版.docx          ★ 论文（《计算机工程与应用》官方模板版）
├── 项目相关文献检索汇总_40篇.docx     引用文献汇总
│
├── md2docx.py                   ★ Markdown → DOCX 转换器（三线表 / 图题表题中英对照 / 图片按真实 DPI 限宽 8 cm）
├── md2docx_cea.py / _cea2.py / _cea3.py  期刊模板专用转换脚本（cea3：首页脚注 + 变量斜体）
├── make_software_copyright_doc.py 软著源程序文档生成脚本
│
├── paper_experiments/           按周组织的实验代码与结果
│   ├── week1_vlm_probe/         大模型能力探针、数据清洗、重新标注管线
│   ├── week2_test_relabel/      test 集重标注、基线评测、置信度-错误率（H1）
│   ├── week3_method/            协同推理首轮实现
│   ├── week4_experiments/       ★ 阈值扫描 / 对比 / 消融全量数据
│   ├── week5_mvp/               ★ 最终方法定型、H2 直接证据、黑白投稿图
│   └── data_cleanup_backup/     数据清洗前的标签备份
│
└── ClothInspect-Agent/          服装智能质检系统 Demo（Vue3 + FastAPI）
    ├── backend/app/core/        detector / router / cropper / vlm_client / fusion / pipeline
    ├── frontend/src/views/      首页 / 检测工作台 / 检测记录 / 实验结果 / 关于
    ├── docs/软件功能说明书.md
    └── docs/screenshots/        11 张功能截图（软著申请用）
```

### 关键结果文件

| 文件 | 内容 |
|---|---|
| `paper_experiments/week4_experiments/week4_results.json` | **阈值扫描 sweep**（T=0.3~0.7 的调用率与 P/R/F1）、四组对比、消融 |
| `paper_experiments/week4_experiments/week4_results_full.json` | 各方法的严格/无关双口径完整指标 |
| `paper_experiments/week5_mvp/h2_crop_vs_full.json` | **H2 直接证据**：同一批 810 框的裁剪图 vs 整图对照 |
| `paper_experiments/week5_mvp/hard_set_experiment.json` | 难样本子集与按类别召回 |
| `paper_experiments/week2_test_relabel/baseline_test_results.json` | V6 基线 mAP50 与逐类 P/R/F1 |
| `paper_experiments/week2_test_relabel/vlm_baseline_test.json` | 大模型整图独立判断（图像级 F1 0.9315） |

---

## 4. 环境与复现

### 环境

实验在 Windows + Conda 环境下完成：

| 项 | 版本 |
|---|---|
| Python | 3.9.23 |
| PyTorch | 2.1.2+cu118 |
| Ultralytics | 8.3.217 |
| OpenCV | 4.12.0 |
| NumPy | 1.26.4 |

```bash
conda create -n label-studio python=3.9
conda activate label-studio
pip install ultralytics==8.3.217 opencv-python numpy requests
```

### 大模型调用

复核使用阿里云百炼 DashScope 的 `qwen3-vl-plus`，需设置环境变量：

```bat
set DASHSCOPE_API_KEY=sk-xxxx
```

`ClothInspect-Agent` 的完整配置项见 `ClothInspect-Agent/.env.example`（复制为 `.env` 后填写）。

### 复现主要实验

```bat
set DASHSCOPE_API_KEY=sk-xxxx

:: 基线评测（无需大模型）
python paper_experiments\week2_test_relabel\verify_v6_on_test.py

:: 阈值扫描 + 对比 + 消融（需要大模型，结果带增量缓存）
python paper_experiments\week4_experiments\week4_experiments.py

:: H2：裁剪图 vs 整图 对照（复用 week4 缓存，零新增 API 调用）
python paper_experiments\week5_mvp\h2_crop_vs_full.py

:: 难样本与按类别召回
python paper_experiments\week5_mvp\hard_set_experiment.py

:: 重绘投稿版黑白插图（宽度 < 8 cm、图字小 6 号宋体）
python paper_experiments\week5_mvp\make_bw_figures.py
python paper_experiments\week5_mvp\make_error_cases_bw.py
```

> 脚本内的权重与数据路径需按本机实际位置调整。

---

## 5. 数据集说明

| 划分 | 图像数 | 说明 |
|---|---:|---|
| 训练集 | 2155 | 已移出 test 图并去重 |
| 验证集 | 271 | 已重新标注 |
| 测试集 | 300 | 已重新标注，849 个缺陷框（破洞 372 / 污渍 132 / 褶皱 345） |

三个划分按**图像内容（MD5）去重**后完全隔离。

原始公开数据存在明显的标注质量问题（见 `paper_experiments/week1_vlm_probe/baseline_v1.md`）：

| 问题 | 实测结果 |
|---|---|
| 漏标 | 被标注为"无标注"的图像中，约 **92%** 实际含缺陷 |
| train/val 泄漏 | **162/271** 幅验证图像与训练集完全相同（MD5） |
| 孤儿标签 | 277 个（有标签无对应图像） |
| 类别失衡 | 原始训练标注中仅 12 幅图含破洞标注，而重标测试集含 372 个破洞框 |

因此本文对验证集与测试集执行"**大模型预筛 + 人工逐框确认**"的重新标注（预筛 1148 个候选 / 自动保留 849 / 人工确认 299）。

**口径影响**：同一批 300 幅测试图像，使用**原始（漏标）标签**时 V6 的 mAP50 为 **0.953**，使用**修正标签**时为 **0.387**。比赛时期声称的 0.865~0.877 因受训练/验证泄漏与漏标影响而虚高，本文不予采用。这一差距说明数据标注质量是制约该任务评测的主要因素，也是本文选择改进数据与推理流程、而非改动模型结构的原因。

> 受仓库体积限制，**数据集与模型权重未纳入版本库**，详见第 7 节。

---

## 6. 系统 Demo：ClothInspect-Agent

`ClothInspect-Agent` 是论文方法的工程实现，用于软件著作权申请与演示。

**功能**：样例库选取 / 上传图像 → YOLO 检测 → 协同推理（含路由轨迹可视化）→ 仅 YOLO 对照 → 检测记录与统计 → 实验结果看板（内嵌论文实验数据）→ 导出检测报告。

**Agent 工具链**：`detect_defect → crop_region → verify_with_vlm → fuse_results`，可通过 `GET /api/tools` 导出 Function Calling 声明。

**启动**：

```bat
cd ClothInspect-Agent
start.bat
```

后端 FastAPI（默认 `http://127.0.0.1:8000`）+ 前端 Vue3/Vite。未配置 API Key 时自动降级为离线启发式复核，界面会明确标注，全流程仍可演示。

**一致性校验**：`ClothInspect-Agent/backend/scripts/validate_subset.py` 可在 Demo 数据上重算指标。关闭大模型的模式在全部 300 幅图像上复现 **F1 = 0.5401**，与论文低阈值控制组的 0.540 一致；开启真实大模型复核的结果存于 `ClothInspect-Agent/backend/data/validate_subset_real.json`。

---

## 7. 未纳入版本库的内容

因体积原因（总计约 983 MB）以下内容未上传：

| 内容 | 体积 | 说明 |
|---|---:|---|
| `实验数据/` | 605.6 MB | 训练/验证/测试图像与标签 |
| `模型权重/` | 109.8 MB | 6 个 `best.pt`（含主模型 `Final_Model_V6_best.pt`，18.3 MB） |
| `ClothInspect-Agent/node_modules`、`.npm-cache` | ~150 MB | 依赖与缓存 |
| `paper_experiments/**/test_originals_backup/` | 70.2 MB | 测试集原图备份 |
| `paper_experiments/**/hard_samples/` | 20.7 MB | 中间产物图像 |
| 个人简历 PDF | — | 含个人联系方式 |

数据集与权重保留在本地，脚本通过路径引用。若需公开数据集，请依据原始数据来源的许可协议另行确认。

此外，**投稿信**属个人隐私材料，已由 `.gitignore` 排除，不在本仓库内；投稿流程与证明材料（执行方案、期刊模板与格式要求、学位规定、软著文档等）同样只保留在本地。

排除规则见 [`.gitignore`](.gitignore)。所有实验脚本均可重跑以重新生成被排除的中间产物。

---

## 8. 论文与投稿

- **目标期刊**：《计算机工程与应用》（北大核心 / CCF T2）
- **稿件**：`论文初稿_v6.md` → `论文初稿_v6_模板版.docx`（投稿版），4 幅插图、7 张表格，黑白、宽度 < 8 cm
- **图包**：`paper_experiments/week5_mvp/图包_矢量图/` —— 图 1~3 为可编辑矢量图（PDF/EPS），图 4 为 600 dpi 位图（TIF/PNG）
- **作者**：杨骏杰，武汉工程大学 计算机科学与工程学院
- **状态**：已按期刊须知补齐中文文献英文对照、卷期页码、首页脚注（基金项目/作者简介）与文后联系方式，待投稿
- **说明**：投稿信不在本仓库内

---

## 9. 引用

若本仓库对你的研究有帮助：

```bibtex
@article{yang2026garment,
  title   = {服装微痕检测的视觉与大模型协同推理方法},
  author  = {杨骏杰},
  journal = {计算机工程与应用},
  year    = {2026},
  note    = {投稿中}
}
```

## 10. 说明

本仓库为个人研究项目，用于记录研究过程与投稿配套材料（论文稿件正文与投稿信不在仓库内）。论文尚在投稿阶段，结论以最终发表版本为准。

- 已核查仓库内**不含任何 API 密钥**；`.env` 与个人简历已由 `.gitignore` 排除。
- 复现实验需要自行准备 DashScope API Key，并注意调用成本。
