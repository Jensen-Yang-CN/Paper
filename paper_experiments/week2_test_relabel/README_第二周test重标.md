# 第二周任务 · test 集重标操作说明（2026-08-22 预生成）

> 预筛已在 2026-08-22 后台启动，预计当天跑完（300 张）。
> 下周回来按下面 4 步操作即可，全程约 30-60 分钟。

## 第 0 步：确认预筛完成

检查文件是否已生成且完整：

```
E:\CodeBase_YangJunJie\Paper\paper_experiments\week1_vlm_probe\relabel\relabel_test_candidates.json
```

预期：300 张图记录。可打开查看（或数一下 JSON 里的对象个数，应为 300）。
若不足 300（中途断电等），重新跑一遍命令即可断点续传（已处理的会自动跳过）：

```
set DASHSCOPE_API_KEY=sk-xxxx
E:\Miniconda\envs\label-studio\python.exe relabel_pipeline.py ^
  --img-dir "E:\CodeBase_YangJunJie\ClothingInspection\Final_Merged_Dataset_v3\images\train" ^
  --list "E:\CodeBase_YangJunJie\Paper\paper_experiments\week1_vlm_probe\test\test_images.txt" ^
  --out "E:\CodeBase_YangJunJie\Paper\paper_experiments\week1_vlm_probe\relabel\relabel_test_candidates.json"
```

## 第 1 步：启动人工确认界面（test 版）

用环境变量切换到 test 数据（和上次 val 版不同，图片在 train 目录、候选文件不同、结果文件不同、端口不同）：

```
set RELABEL_CAND=E:\CodeBase_YangJunJie\Paper\paper_experiments\week1_vlm_probe\relabel\relabel_test_candidates.json
set RELABEL_RESULTS=E:\CodeBase_YangJunJie\Paper\paper_experiments\week1_vlm_probe\relabel\relabel_test_results.json
set RELABEL_IMG_DIR=E:\CodeBase_YangJunJie\ClothingInspection\Final_Merged_Dataset_v3\images\train
set RELABEL_PORT=8767
E:\Miniconda\envs\label-studio\python.exe relabel_server.py
```

浏览器打开 **http://127.0.0.1:8767**，逐张确认（操作和上次 val 版完全一样）：
- 虚线框 = 自动保留，不用管
- 实线框 = 点选后 保留/删除/改类别
- 漏了 → ＋补框拖拽
- 干净的图 → 整图无瑕疵
- 每点一下自动保存；右上角进度 300/300 即完成

## 第 2 步：生成 test 修正标签

确认全部完成后，运行（用 test 版生成脚本——val 版脚本已参数化，直接传参）：

```
E:\Miniconda\envs\label-studio\python.exe gen_relabeled_labels.py --mode test
```

> 注：如果 gen_relabeled_labels.py 还没加 --mode test 参数，届时告诉我，我来改（1 分钟）。
> 预期输出：relabeled_test/（短名 v 文件 + 映射表）。

## 第 3 步：收尾数据结构（关键，防反向泄漏）

1. 把 300 张 test 图从 train **物理移出**到 test 目录（train 剩 ~2155 张）；
2. 更新 data.yaml：train / val(relabeled_val) / test(relabeled_test) 三套分开；
3. 在重标 test 上重算 V6 最终基线（eval_baseline.py 传 test 参数）→ 论文第一张正式表；
4. 继续第二周剩余任务（置信度分析正式图、Hard Set 定义、VLM Baseline 正式化）。

## 提示

- API Key 用同一个：sk-xxxx（如已轮换，替换上面命令）
- 预筛产物位置：`relabel/relabel_test_candidates.json`（候选框 + 千问判定 + auto_keep 标记）
- 一切脚本都在 `paper_experiments/week1_vlm_probe/`，用 label-studio 环境
