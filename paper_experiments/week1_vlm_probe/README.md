# 第一周任务包：VLM 可行性探针（2026-08-22 ~ 08-28）

> 目标：在完整整理数据集**之前**，先用 2~3 天证明"Qwen-VL 能否在局部裁剪图上
> 确认/找回/拒绝瑕疵"。这一步是整篇论文方向的**生死实验**——VLM 不成立，
> 后面的协同推理、动态路由、消融全部没有意义。

## 一、今天（Day 1）干什么

1. **拿 API Key**：dashscope 控制台 → 创建 API-KEY → 设置环境变量
   ```
   set DASHSCOPE_API_KEY=sk-xxxx
   ```
2. **提取难样本**（用 label-studio 环境）：
   ```
   E:\Miniconda\envs\label-studio\python.exe extract_hard_samples.py --split val
   ```
   输出在 `hard_samples/`：crops 裁剪图 + meta csv + confidence_stats.json
3. **人工抽查**：打开 `hard_samples/crops/` 的 `missed_gt`（漏检）和 `false_positive`（误检）
   两类裁剪图，确认：(a) 漏检框里**确实有**瑕疵；(b) 误检框里**确实没**瑕疵。
   如果标注本身有问题，先修标注再测，否则 VLM 判对了也会被算成错。
4. **试通 VLM**（先 10 张）：
   ```
   E:\Miniconda\envs\label-studio\python.exe test_vlm.py --meta hard_samples\hard_samples_meta.csv --limit 10
   ```
   目的只是打通调用 + 输出 JSON 解析。如果解析经常失败，改 `test_vlm.py` 里的 PROMPT。

## 二、Day 2~3：正式跑探针

- 跑全量难样本（几十到一百多张，成本几块钱内）：
  ```
  E:\Miniconda\envs\label-studio\python.exe test_vlm.py --meta hard_samples\hard_samples_meta.csv
  ```
- 看 `vlm_results.csv`，按 case_type 统计一致率。
- **判定标准（方向是否跑通）**：
  | 指标 | 目标值 | 含义 |
  |---|---|---|
  | 难样本一致率（low_conf_tp + missed_gt） | ≥ 70% | VLM 能确认/找回多数难样本 |
  | 误检拒绝率（false_positive） | ≥ 60% | VLM 不乱报，能拒绝 YOLO 误检 |
  | 正常区域误报率（normal_region） | 越低越好 | VLM 不会凭空造瑕疵 |

- 未达标时的**调整顺序**（每次改完重测同一批样本）：
  1. 改 PROMPT（更明确的判定标准、强调微小瑕疵、放大倍数提示）
  2. 换模型：`set VLM_MODEL=qwen-vl-max`
  3. 换输入：裁剪图太小就加大 `MIN_CROP_SIZE` / 裁剪前先放大
  4. 以上都不行 → **停下，回来讨论 pivot**，不要恋战

## 三、Day 4~6：数据整理 + 冻结基线（无论探针结果如何都要做）

1. **清孤儿标签**：`Final_Merged_Dataset_v3/labels/` 有约 3003 个 txt，但图片只有 2726 张
   → 删除没有对应图片的 txt
2. **建测试集**：从 train 按类别分层切出 ~300 张作 test，更新 data.yaml
3. **查跨源重复**：天池 + 多个 Roboflow 数据合并时可能混入重复图（做一遍文件 hash 查重）
4. **写评估脚本**：固定 conf、固定 test，输出每类 P/R/mAP50/mAP50-95 → `baseline_results.json`
   （注意：简历里的 86.5% 是旧口径，论文里所有数字必须用新协议重算）

## 四、环境与依赖

- 运行环境：`E:\Miniconda\envs\label-studio\python.exe`（有 ultralytics/torch/cv2/requests）
- 依赖：脚本只用 cv2 / numpy / ultralytics / requests，无新增安装
- 模型：`Final_Model_V6\weights\best.pt`（mAP50≈87.7%）
- 脚本位置：`E:\CodeBase_YangJunJie\Paper\paper_experiments\week1_vlm_probe\`
  （`paper_experiments/` 是论文实验专用目录，与根目录的比赛代码区分）

## 五、产物清单（本周结束应产出）

- [ ] `hard_samples/`：难样本裁剪图 + meta + confidence_stats（H1 的置信度-错误率证据雏形）
- [ ] `vlm_results.csv` + 一致率汇总（方向成立性结论）
- [ ] 孤儿标签清理后的数据集 + 新 test 划分
- [ ] 统一协议的 `baseline_results.json`（V6 在 test 上的真实数字）
- [ ] 本周结论写进 `baseline_v1.md`：方向是否成立 + 下一步计划
