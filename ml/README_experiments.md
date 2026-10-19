# 实验总结 / Experiment Summary

## 1. 核心指标

- **优化目标：F1-score**（正类 = 正面）。
- 理由：情感分析需要在 Precision（不把好评误判成差评）和 Recall（不漏掉真实差评）之间平衡。只看 Accuracy 会在类别不平衡时产生误导，而舆情场景里「漏掉差评」和「错杀好评」的代价都很高。

## 2. 生产模型

**`baseline-v1`**（`models/model_v1.pkl`）是当前生产模型：

- 训练数据：`data-v1`（`data/imdb_v2.csv`，2000 条干净样本）
- 超参：`TF-IDF(max_features=10000, ngram_range=(1,2))` + `LogisticRegression(C=1.0)`
- **test F1 = 0.8328**
- 训练脚本 commit：`e73dd4d3`，数据 md5：`48978ce7a8aa6e804962a2f4af700159`
- 复现命令：`python -m ml.train --config ml/configs/baseline.yaml`

## 3. 实验对比

| run 名称 | 数据 | C | max_features | min_df | sublinear_tf | test F1 | 该 run 要回答的问题 |
|----------|------|---|--------------|--------|--------------|---------|---------------------|
| `baseline-v1` | data-v1 | 1.0 | 10,000 | 1 | false | **0.8328** | 干净数据上的基线能到多少？（→ 生产模型） |
| `improved-tuned` | data-v1 | 2.0 | 30,000 | 2 | true | 0.8212 | 在**同一份数据**上，只靠调超参能不能超过基线？ |
| `improved-v2-cleaned` | data-v2 | 2.0 | 20,000 | 1 | false | 0.7525 | 换成**规则清洗后的 data-v2**，模型会变好还是变差？ |

> **这三条 run 不是"同一目标的逐次优化"，而是三个独立假设的验证。**
> `improved-v2-cleaned` 沿用了 MLflow 记录时的历史 run 名，容易被误读成"改进版"；它的真实目的是**验证数据清洗管线是否有效**，而不是刷高 F1。三者合起来覆盖了"模型侧（超参）"和"数据侧（清洗）"两条路径，这正是设置多个实验的意义。

### 实验设计说明：三条 run 各自的假设

| run | 改动维度 | 假设 | 结果 |
|-----|----------|------|------|
| `baseline-v1` | — | 固定基线，作为其他实验的参照系 | 建立参照：F1 = 0.8328 |
| `improved-tuned` | 仅超参（词表 / 正则 / 加权） | "模型侧还有提升空间" | **假设不成立**，0.8212 ≤ 0.8328 |
| `improved-v2-cleaned` | 仅数据（v1 → v2 清洗链） | "清洗数据能让模型更好" | **假设不成立**，0.7525 < 0.8328，且原因可解释 |

复现命令：

```bash
python -m ml.train --config ml/configs/baseline.yaml      # 0.8328
python -m ml.train --config ml/configs/improved.yaml      # 0.8212
python -m ml.train --config ml/configs/cleaned-v2.yaml    # 0.7525
```

### 补充：C 值扫描（都在 data-v1 上，max_features=20000）

| C | 0.5 | 1.0 | 3.0 | 10.0 |
|---|-----|-----|-----|------|
| test F1 | 0.8054 | 0.8328 | 0.8185 | 0.8212 |

扫描用的是同一份数据、同一个随机种子，说明在这份 2000 条的数据上，C=1.0 附近就是最优点；继续增大或减小都会掉点。

## 4. 结论与解释

### 4.1 调参方向：已经打到瓶颈

3 组调参配置（更大的词表、`min_df=2`、`sublinear_tf`、不同 C）都低于基线。测试集只有 300 条，F1 的标准误约 ±0.02，这些差异基本落在噪声范围内。**结论：在 2000 条这个数据量级上，TF-IDF + LR 的超参已接近上限，继续调参的边际收益极低。**

### 4.2 数据方向：data-v2 的目标是验证清洗管线，不是提升 F1

`data-v2` 是刻意设计的"数据灾难演练"，完整链路见 [`README_data.md`](README_data.md)：

```text
data-v1（2000 条干净）
   │ inject_noise.py：+200 条乱码 / +100 条空文本 / +200 条重复水军，并翻转 5% 标签
   ▼
imdb_v3.csv（2500 条，中间态，不打版本标签）
   │ clean.py：去空文本 / 去乱码 / 去重 / 截断
   ▼
data-v2（2001 条）
```

关键在于两类问题的性质完全不同：

- **格式异常**（空文本、乱码、重复样本）→ 有明确规则可依，`clean.py` 全部清掉；
- **标签错误**（被翻转的 5%）→ 文本本身完全正常，只是标签被改错，规则清洗**没有任何依据**去发现它们。

因此 `data-v2` 的标签质量实际上低于 `data-v1`（残留约 5% 翻转噪声），F1 从 0.8328 掉到 0.7525 是这条数据链的必然结果，而不是清洗脚本写错了。

**这个实验的价值正是一个明确的负面结论：面对标签噪声，规则清洗无效，继续调参同样无效。** 若目标是继续提分，正确方向是标签层面的处理（人工复核、`cleanlab` 之类的置信学习），而不是在特征或超参上加码。该方向本次未执行，仅作为后续路线记录，不作为已完成的实验结果。

### 4.3 因此生产模型保持 `baseline-v1`

它在干净数据（`data-v1`）上取得最高 F1 0.8328，同时词表只有 10k、推理延迟最低。**数据正确性优先于指标好看**——这一取舍同步记录在 [`registry/current_model.md`](registry/current_model.md)。

## 5. 每条 run 都记录了什么

- `git_commit_sha`：代码版本（`get_git_commit()`）
- `dvc_dataset_hash`：训练数据的 DVC md5（`get_dvc_hash()` 直接读 `.dvc` 指针）
- `data_path`、`max_features`、`ngram_range`、`C`、`min_df`、`sublinear_tf`、`max_text_length`
- `test_f1`
- 模型 artifact：`mlflow.sklearn.log_model(...)` 与 `.pkl` 文件

> 历史遗留：早期两条 run（`dapper-fox-483` F1=0.7483、`nosy-mole-266` F1=0.7550）没有记录 git / data 版本信息；`secretive-shark-163` 记录的是旧指针哈希 `58e00f98`（当时 `.dvc` 指针尚未刷新）。以上 run 保留作历史，不作为生产依据。

## 6. 查看方式

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
