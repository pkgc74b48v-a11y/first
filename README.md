# IMDb 影评情感分类器 (MLOps Pipeline)

> 一个集成了 DVC 数据版本管理、MLflow 实验追踪、Docker 容器化与 GitHub Actions CI/CD 的轻量级 NLP 分类服务。

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Task](https://img.shields.io/badge/Task-Text%20Classification-green.svg)
![Data](https://img.shields.io/badge/Dataset-IMDb-orange.svg)
![DVC](https://img.shields.io/badge/Data%20Version-DVC-red.svg)
![MLflow](https://img.shields.io/badge/Experiment%20Tracking-MLflow-lightgrey.svg)
![Docker](https://img.shields.io/badge/Deploy-Docker-blueviolet.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-brightgreen.svg)

---

## 📑 目录

- [1. 任务定义](#1-任务定义)
- [2. 数据与版本管理 (DVC)](#2-数据与版本管理-dvc)
- [3. 模型与实验追踪 (MLflow)](#3-模型与实验追踪-mlflow)
- [4. CI/CD 自动化流水线](#4-cicd-自动化流水线)
- [5. 本地运行方式](#5-本地运行方式)

---

## 1. 任务定义

| 项目 | 内容 |
|------|------|
| **任务类型** | 文本二分类（轻量 NLP） |
| **输入** | 一条英文影评文本（`string`，截断至 1000 字符） |
| **输出** | 正面（`1`）/ 负面（`0`） |
| **主指标** | **F1-score**（正类 = 正面） |

### 💡 业务解释
> F1 同时约束「漏放」和「错杀」，比 Accuracy 更贴合客服 / 舆情场景。如果 F1 低，意味着差评可能被漏判为好评，导致舆情监控失效。

---

## 2. 数据与版本管理 (DVC)

- **数据集**：`stanfordnlp/imdb`（Hugging Face，公开数据集）。
- **规模**：**2000 条**（正负各 1000 条）。
- **切分**：70/15/15 分层划分（Train 1400 / Val 300 / Test 300）。
- **DVC 版本链**：
  - `data-v1`：基础干净数据集（2000条），对应文件 `imdb_v2.csv`。
  - `data-v2`：经过重度数据灾难注入（乱码、水军、20%标签翻转）后，使用 `clean.py` 清洗修复的数据集，对应文件 `imdb_v4.csv`。
- **复现数据**：详见 `ml/README_data.md` 和 `ml/PROVENANCE.md`。

---

## 3. 模型与实验追踪 (MLflow)

- **模型**：TF-IDF (1-2 gram, `max_features=10000`) + LogisticRegression。
- **MLflow 追踪**：记录超参数、F1 指标、代码 Commit SHA、数据 DVC Hash 以及最终的模型 Artifact (`.pkl`)。
- **实验成果**：
  - 基线模型 F1 约 0.78。
  - 清洗后模型 F1 稳定在 **0.8328** 左右。
  - 详见 `ml/README_experiments.md`。

---

## 4. CI/CD 自动化流水线

本项目采用标准的 `feature -> dev -> staging -> main` 四层分支模型，并由 GitHub Actions 提供全自动的校验与交付。

### 流程图
```text
[feature/*] 
    │ (Open PR)
    ▼
[ dev ] ──▶ Triggers CI: Lint (ruff/black) + Pytest + Docker Build
    │ (Merge PR)
    ▼
[ staging ] ──▶ Triggers CD: Reads STAGING_ENV_FILE -> Builds `ml-app:staging` Docker image
    │ (Merge PR)
    ▼
[ main ] ──▶ Triggers CD: Reads PRODUCTION_ENV_FILE -> Builds `ml-app:production` Docker image- [x] test: verify staging to main deployment
