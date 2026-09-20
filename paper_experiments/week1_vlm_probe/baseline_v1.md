# Baseline V1 报告（2026-08-22，第一周收官）

> 本文件冻结第一周的基线结论。**重要：本报告的数字基于修正后的标注（relabeled_val），
> 与比赛时期的数字（mAP50=86.5%）口径不同，后者已被证明受训练/验证泄漏影响。**

## 一、数据集审计结论（本周最重要的发现）

| 项 | 结果 | 影响 |
|---|---|---|
| 漏标率 | "无标注"图中 ~92% 实际有缺陷（用户目测 3/3 确认） | 原 GT 不可信，已重标 |
| train/val 泄漏 | **162/271 张 val 图与 train 完全相同（MD5）** | 原 mAP50=87.7% 被高估 |
| 孤儿标签 | 277 个（有标签无图），已移入备份 | 已清理 |
| train 内部重复 | 5 组 | 已去重统计 |
| 类别失衡 | train 仅 12 张图含 hole 标注（原始标注）；修正后 val 有 152 个 hole 框 | 论文必须如实报告 |
| 长文件名 | Roboflow 文件名最长 176 字符，超 Windows MAX_PATH | relabeled_val 用短名 v00001-v00271 + 映射表 |

## 二、修正后的标注（relabeled_val，val 集 271 张）

- 修正后框数：hole 152 / stain 201 / wrinkle 176（共 529）
- 相比原始标注：73 张"无标注"图补上真实缺陷框；22 张"有标注"图被判定为干净
- 方法：YOLO 低阈值候选 + qwen3-vl-plus 预筛 + 人工逐框确认（394 个需确认，已全部完成）

## 三、V6 模型在修正后 val 上的真实指标（baseline_results.json）

### mAP（ultralytics val，全 271 张，修正 GT）

| 类 | mAP50 | mAP50-95 |
|---|---:|---:|
| hole | 0.563 | - |
| stain | 0.397 | - |
| wrinkle | **0.101** | - |
| **ALL** | **0.406** | **0.354** |

（比赛/简历声称 0.865-0.877，受泄漏+旧标注影响，**不可用**）

### P/R/F1（conf=0.4，IoU=0.5，框级）

| 集 | 类 | Precision | Recall | F1 |
|---|---|---|---|---|
| 全量 271 | hole | 0.412 | 0.665 | 0.509 |
| 全量 271 | stain | 0.517 | 0.751 | 0.613 |
| 全量 271 | wrinkle | 0.576 | **0.108** | 0.182 |
| 全量 271 | ALL | 0.475 | 0.512 | 0.493 |
| 无泄漏 109 | ALL | 0.386 | 0.466 | **0.422** |

### 关键结论

1. **wrinkle 是最大短板**：修正 GT 下召回仅 0.11（176 个框只检出 19 个）——模型对"未见过"的褶皱几乎失效；
2. **泄漏影响巨大**：无泄漏子集（109 张）总体 F1 仅 0.42；stain 召回从 0.75 掉到 0.21；
3. **这对论文是利好的事实基础**：YOLO 在难样本（新标注的褶皱/污渍）上确实大量漏检，
   而千问对"漏检 GT"找回率 90%——协同推理的动机被实验完全坐实；
4. 论文正式指标以**重标后的 test 集**为准（test 已切 300 张，标签待重标，第二周做）。

## 四、第一周完成清单

- [x] VLM 可行性探针：难样本一致率 90.2%（qwen3-vl-plus + 质检提示词）
- [x] 置信度-错误率曲线（H1 证据）：conf 0.1 错误率 84% → conf 0.9 为 0%
- [x] 标注审计：漏标率 ~92%，千问精度 94% / 漏报 0%（44 张人工目测）
- [x] val 全量重标（271 张，VLM 预筛 + 人工确认）→ relabeled_val
- [x] 孤儿标签清理（277 个）
- [x] 泄漏检测（162 张）与无泄漏子集切分（clean_val 109 张）
- [x] test 集分层切分（300 张：hole 12 + stain 120 + wrinkle 120 + 负样本，标签待重标）
- [x] 基线评估脚本（eval_baseline.py，可复现）
- [x] 本报告

## 五、第二周（8/29-9/4）计划

1. **test 集重标**（复用 relabel 流水线，300 张，约 1-2 天）
2. 在修正 test 上重算最终基线（论文第一张正式表）
3. 置信度分布与错误率分析细化（H1 正式图）
4. Hard Set 定义（按标注客观属性：框面积/对比度，**不用模型置信度**）
5. VLM 复核 Baseline 正式化（固定 prompt v2 + qwen3-vl-plus）

## 六、遗留问题

- test 集原始标签有漏标/误标，重标前不可用于正式指标
- clean_val（109 张）类别失衡（hole 偏多），仅作参考，不作为论文正式验证集
- 数据来源许可需核对（天池布匹 + Roboflow，投稿前确认可发表）

---

# 八、第二周更新：test 重标完成 + 论文正式基线（2026-08-29）

## 8.1 test 集重标完成

- 300 张全部确认（VLM 预筛 1148 候选 / 自动保留 849 / 人工确认 299）
- 修正后框数：hole 372 / stain 132 / wrinkle 345（共 849）
- 相比原始 train 标注：48 张"无标注"图补上框；49 张"有标注"图判定为干净
- **重大发现：原始 train 的 hole 标注只有 12 张图，重标 test 发现 372 个 hole 框——hole 漏标极其严重**

## 8.2 数据结构定版

- train：Final_Merged_Dataset_v3/images/train（**已移出 300 张 test 图，剩 2155 张**）
- val：`relabeled_val/`（271 张，修正标签）
- test：`relabeled_test/`（300 张，修正标签）
- 最终划分：`final_data.yaml`

## 8.3 论文正式基线（baseline_test_results.json，2026-08-29）

### mAP（test 300 张，重标 GT，ultralytics val）

| 类 | mAP50 | mAP50-95 |
|---|---:|---:|
| hole | 0.471 | - |
| stain | 0.265 | - |
| wrinkle | 0.198 | - |
| **ALL** | **0.387** | **0.311** |

### P/R/F1（conf=0.4，IoU=0.5，框级，test 300 张）

| 类 | Precision | Recall | F1 |
|---|---:|---:|---:|
| hole | 0.629 | 0.605 | 0.616 |
| stain | 0.443 | 0.621 | 0.517 |
| wrinkle | 0.555 | **0.278** | 0.371 |
| ALL | 0.563 | 0.475 | **0.515** |

### 结论

1. **这是论文第一张正式结果表**，口径为"重标 test + 统一 conf=0.4"；
2. wrinkle 召回 0.278（最弱）→ 协同机制的主攻方向：VLM 找回 YOLO 漏检的褶皱；
3. hole 最强（F1 0.616）→ 基线本身对破洞有一定能力，协同提升空间主要看 wrinkle/stain；
4. 论文中禁止使用比赛口径 0.865（泄漏 + 旧标注）。

## 8.4 第二周剩余任务

- [x] test 重标 + 最终基线
- [ ] 置信度分布与错误率正式图（H1）
- [ ] Hard Set 定义（标注客观属性：框面积分位/对比度，不用模型置信度）
- [ ] VLM 复核 Baseline 正式化（固定 prompt v2 + qwen3-vl-plus，在 test 上评估）
- [ ] 错误案例收集（error_cases/）

---

# 七、第一周完整工作日志（2026-08-22 ～ 08-28 过程记录）

## 7.1 逐日/逐环节记录

| 环节 | 做了什么 | 结果 |
|---|---|---|
| 1. 环境确认 | 确认 label-studio 环境（ultralytics/torch/cv2/requests 齐全），确认 V6 模型与数据路径 | 可复现基线 |
| 2. 难样本提取 | 编写 `extract_hard_samples.py`，val 271 张 YOLO(conf=0.1) 推理 + GT 匹配，抽 6 类难样本 | 291 个裁剪样本 |
| 3. VLM 探针 v1 | qwen-vl-plus + 简版提示词 | 把褶皱当"正常"，检出率低 → 发现问题 |
| 4. 提示词迭代 | v2（质检场景，褶皱=瑕疵）→ 高置信度TP 66.7%；v3（负例压制）→ 误报降但召回归零 | 确认提示词敏感性 |
| 5. 模型对比 | qwen-vl-max 无明显提升；qwen3-vl-plus 显著更好 | 确定用 qwen3-vl-plus |
| 6. 全量探针 | qwen3-vl-plus + v2 全量 291 样本 | **难样本一致率 90.2%，方向成立** |
| 7. 置信度曲线 | `confidence_stats.json`：conf 0.1→错误率 84%，0.9→0% | H1 证据 |
| 8. 标注审计 | 抽样 44 张，本地网页点选；漏标率 ~92%，千问精度 94%、漏报 0% | 实锤漏标 |
| 9. 重标预筛 | `relabel_pipeline.py`：YOLO 候选 + 千问逐框判定 + 千问全图补漏 | 923 候选，529 自动保留 |
| 10. 人工确认 | `relabel_server.py` v1（卡顿）→ v2（画布直绘+自动保存+断点续传） | 271 张全部确认 |
| 11. 生成修正标签 | `gen_relabeled_labels.py`（修 MAX_PATH：短名+映射表） | relabeled_val：529 框 |
| 12. 数据整理 | `data_cleanup.py`：孤儿标签 277 个清理；MD5 查重 | 发现 162 张 train/val 泄漏 |
| 13. 无泄漏切分 | `make_splits.py`：clean_val 109 张；test 300 张分层切分 | 修复泄漏影响 |
| 14. 基线评估 | `eval_baseline.py`：修正 val 上 mAP50=0.406；conf=0.4 P/R/F1 | baseline_results.json |

## 7.2 脚本清单（全部在 `paper_experiments/week1_vlm_probe/`，用 label-studio 环境）

| 脚本 | 用途 | 状态 |
|---|---|---|
| `extract_hard_samples.py` | 难样本提取（低置信度TP/漏检/误检/正常区域） | ✅ 复用 |
| `test_vlm.py` | VLM 复核探针（含提示词 v2/v3 选择） | ✅ 复用 |
| `relabel_pipeline.py` | 重标预筛（YOLO+千问候选生成） | ✅ 复用（test 重标用） |
| `relabel_server.py` | 重标人工确认界面 v2 | ✅ 复用 |
| `gen_relabeled_labels.py` | 生成修正 YOLO 标签（短名+映射） | ✅ 复用 |
| `data_cleanup.py` | 孤儿标签清理 + MD5 查重 | ✅ 复用 |
| `make_splits.py` | 无泄漏 val/test 切分 | ✅ 复用 |
| `eval_baseline.py` | 基线评估（mAP + 框级 P/R/F1） | ✅ 复用 |
| `build_audit_samples.py` / `analyze_audit.py` | 审计抽样/分析 | ✅ 复用 |

## 7.3 关键决策记录

1. **VLM 模型**：qwen3-vl-plus（qwen-vl-plus/max 感知不足；qwen2.5-vl-72b 无权限）
2. **提示词**：v2 质检场景版（褶皱=瑕疵 + 尽量不漏报），标注在 `test_vlm.py` 中可切换
3. **评测 GT**：以重标后的 relabeled_val / 未来重标 test 为准；**禁用**比赛时期旧指标
4. **Hard Set 原则**：按标注客观属性定义（框面积/对比度），不用模型置信度（避免循环论证）
5. **长文件名**：重标输出统一短名 v00001-v00271 + `name_mapping.csv`

## 7.4 产出文件一览

```
paper_experiments/week1_vlm_probe/
├── probe_results.md            # 探针与审计完整结论（含迭代记录）
├── baseline_v1.md              # 本文件
├── baseline_results.json       # 修正口径基线指标
├── hard_samples/               # 难样本裁剪 + meta + VLM 结果 + 审计结果
├── relabel/                    # 重标候选与人工确认结果
├── relabeled_val/              # 修正标签数据集（images + labels + 映射表 + clean_val_list）
├── test/test_images.txt        # test 切分清单（300 张，待重标）
└── *.py                        # 上述脚本
```
