# Event-TimeSeries_model
This is a graduation project which is used to predict SP500index by fusing event and time-series

# 毕业设计实现说明

## 落地方案

- 任务定义：把问题收缩成“事件发布后未来若干个 bar 的涨跌方向预测（二分类）”。
- 文本输入：只使用 `*_full_summary.txt` 作为真实文本。
- 时序输入：使用 OHLC 表结构，至少包含 `date/Open/High/Low/Close`。
- 融合方式：`TF-IDF + 文本统计特征 + 事件前窗口时序特征 -> Logistic Regression`
- 对比实验：保留三组模型：
  - `text`：文本分支
  - `ts`：时序分支
  - `fusion`：融合分支
- 可解释性：输出全局特征重要性，以及单条预测的关键词和关键技术指标贡献。

这个版本的优点是：

1. CPU 就能跑。
2. 可复现性较好，训练后会保存 `aligned_samples.csv / metrics.json / model_bundle.joblib / global_importance.csv`。

---

## 为什么不用 counterfactual 文本训练

- 只训练真实的 `*_full_summary.txt`
- 把 counterfactual 留作后续扩展（稳健性分析 / 敏感性分析 / 数据增强对比实验）

---

## 适配数据结构

### 1）事件文本目录

脚本默认读取这样的目录：

```text
processed_events_and_counterfactuals/
├── 1/
├── 2/
├── 3/
├── 4/
├── 5/
└── 6/
    └── 20021017/
        └── 20021017.txt_full_summary.txt
```

事件类型映射：

- `1`：Unemployment Insurance Claims
- `2`：Employment Situation Reports
- `3`：GDP Advance Releases
- `4`：FOMC Minutes
- `5`：CPI Reports
- `6`：PPI Reports

### 2）行情表结构

脚本支持：

- `csv`
- `xlsx`
- `parquet`

最少需要这些列：

- `date` 或 `datetime` 或 `timestamp`
- `Open`
- `High`
- `Low`
- `Close`

---

## 默认工程假设

事件目录里主要是“日期”，没有统一给出具体发布时间，所以我做了一个默认值：

- Claims / Employment / GDP / CPI / PPI：默认 `08:30`
- FOMC Minutes：默认 `14:00`


---

## 推荐参数

- `window_bars = 24`：事件前 24 个 bar（约 2 小时）
- `future_bars = 12`：事件后 12 个 bar（约 1 小时）
- `max_features = 1000`

这是一个比较稳妥、硬件压力很小的毕业设计参数组合。

---

## 快速开始

### A. 先生成一份演示数据，验证脚本可运行

```bash
python sp500_event_fusion_minimal.py demo-data --out-dir ./demo_run
```

### B. 用真实数据训练

```bash
python sp500_event_fusion_minimal.py train \
  --events-root "processed_events_and_counterfactuals 路径" \
  --market-file "标普500行情文件.csv" \
  --out-dir "./artifacts" \
  --window-bars 24 \
  --future-bars 12 \
  --max-features 1000
```

### C. 只做对齐、导出样本

```bash
python sp500_event_fusion_minimal.py build-samples \
  --events-root "processed_events_and_counterfactuals 路径" \
  --market-file "标普500行情文件.csv" \
  --out-csv "./aligned_samples.csv"
```

### D. 做单条预测

```bash
python sp500_event_fusion_minimal.py predict \
  --bundle "./artifacts/model_bundle.joblib" \
  --market-file "标普500行情文件.csv" \
  --event-type 5 \
  --publish-time "2024-03-12 08:30:00" \
  --event-text "Inflation cooled more than expected and price pressures eased." \
  --out-json "./pred.json"
```

---

## 训练后会产生的文件

训练完成后，`out_dir` 下会出现：

- `aligned_samples.csv`：对齐后的训练样本
- `metrics.json`：Text / TS / Fusion 三组模型指标
- `model_bundle.joblib`：模型与特征器打包文件
- `global_importance.csv`：全局特征重要性

---

## 为什么这么简化


> 为降低实现复杂度与硬件成本，
> 系统未采用高显存依赖的深层多模态神经网络，而是构建了一个轻量级事件驱动预测框架。
> 文本侧使用 TF-IDF 与轻量文本统计特征，时序侧使用事件前窗口收益率、波动率、动量、RSI、ATR 等统计特征，
> 在融合层采用 Logistic Regression 完成方向预测。
> 该方案支持 CPU 环境下训练与推理，并保留 Text-only、TS-only 与 Fusion 三组实验结果，
> 便于从毕业设计角度完成对比评估与可解释展示。

---


1. 为什么不用大模型微调？
   - 因为毕业设计更强调可复现、可落地、硬件可承受。CPU 版更稳。

2. 为什么只做方向分类？
   - 因为方向预测更稳定，样本规模有限时比直接回归收益率更适合作为主任务。

3. 为什么先不用 counterfactual？
   - 因为它不是真实发生的文本，直接继承原市场标签会污染监督信号。

4. 为什么这个实现是合理的？
   - 因为它保留了事件文本、历史行情、时间对齐、融合建模、基线对比和解释输出，完整覆盖了毕业设计的核心环节。

---

## 后续......

- 用 counterfactual 做稳健性分析
- 增加 LightGBM 作为第二个分类器
- 增加年份分段评估
- 增加一个最小 Flask 页面做演示

当前版本已经足够作为一个能训练、能评估、能预测、能解释、能答辩的最小毕业设计实现。
