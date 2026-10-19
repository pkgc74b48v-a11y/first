# IMDb 影评情感分类器（MLOps Pipeline）

> 一个集成了 DVC 数据版本管理、MLflow 实验追踪、Docker 容器化与 GitHub Actions CI/CD 的轻量级 NLP 分类服务。

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Task](https://img.shields.io/badge/Task-Text%20Classification-green.svg)
![Data](https://img.shields.io/badge/Dataset-IMDb-orange.svg)
![DVC](https://img.shields.io/badge/Data%20Version-DVC-red.svg)
![MLflow](https://img.shields.io/badge/Experiment%20Tracking-MLflow-lightgrey.svg)
![Docker](https://img.shields.io/badge/Deploy-Docker-blueviolet.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-brightgreen.svg)

---

## 目录

- [1. 任务定义](#1-任务定义)
- [2. 仓库结构](#2-仓库结构)
- [3. 数据与版本管理 (DVC)](#3-数据与版本管理-dvc)
- [4. 模型与实验追踪 (MLflow)](#4-模型与实验追踪-mlflow)
- [5. CI/CD 自动化流水线](#5-cicd-自动化流水线)
- [6. 本地运行](#6-本地运行)
- [7. Docker](#7-docker)
- [8. 已知限制](#8-已知限制)

---

## 1. 任务定义

| 项目 | 内容 |
|------|------|
| **任务类型** | 文本二分类（轻量 NLP） |
| **输入** | 一条英文影评文本（`string`，服务端截断至 1000 字符） |
| **输出** | `sentiment`：`positive` / `negative`；`confidence`：置信度（0~1） |
| **主指标** | **F1-score**（正类 = 正面） |
| **模型** | TF-IDF (1–2 gram) + LogisticRegression |

### 业务解释

> F1 同时约束「漏放」和「错杀」，比 Accuracy 更贴合客服 / 舆情场景。如果 F1 低，意味着差评可能被漏判为好评，导致舆情监控失效。

---

## 2. 仓库结构

```text
.
├── app/
│   └── main.py                 # FastAPI 服务入口（/health、/predict）
├── ml/
│   ├── train.py                # 训练脚本（调用 MLflow，支持 YAML 配置）
│   ├── data_pipeline.py        # 数据下载 / 分层抽样 / 70-15-15 切分
│   ├── inject_noise.py         # 数据灾难演练：注入乱码/空文本/水军/标签翻转
│   ├── clean.py                # 数据清洗：去空、去重、截断、去乱码
│   ├── configs/                # 超参配置（baseline / improved / cleaned-v2）
│   ├── registry/current_model.md   # 当前生产模型说明
│   ├── README_data.md          # 数据来源与版本说明
│   ├── README_experiments.md   # 实验对比与生产模型选择
│   └── PROVENANCE.md           # 代码 → 数据 → 模型 溯源表
├── models/
│   └── model_v1.pkl            # 生产模型（入库，保证 docker build 可复现）
├── data/                       # 原始数据目录（只提交 .dvc 指针，不提交数据）
├── tests/                      # 单元测试（49 个用例）
├── .github/workflows/          # ci.yml / deploy-staging.yml / deploy-production.yml
├── Dockerfile
├── .dockerignore
├── requirements.txt            # 推理运行时依赖
├── requirements-dev.txt        # 测试 / 训练 / CI 依赖
├── pyproject.toml              # black / ruff / pytest 配置
└── DEPLOYMENT.md               # 部署说明
```

---

## 3. 数据与版本管理 (DVC)

- **数据集**：Hugging Face `stanfordnlp/imdb`（公开数据集）。
- **规模**：2000 条（正负各 1000），按 70/15/15 分层切分为 train 1400 / val 300 / test 300。
- **远端存储**：课程提供的 Azure Blob Storage，DVC 配置为 `azure://dvc`（`account_name = wut2026`）。
  - SAS token 只写在 `.dvc/config.local`（已被 `.dvc/.gitignore` 忽略，**不会提交**）。
- **版本链**：

| 版本 | 文件 | 行数 | DVC md5 | Git tag |
|------|------|------|---------|---------|
| `data-v1` | `data/imdb_v2.csv` | 2000（1400/300/300） | `48978ce7a8aa6e804962a2f4af700159` | `data-v1` |
| 中间态（未打 tag） | `data/imdb_v3.csv` | 2500（注入噪声后） | `c6cdef524357df9e75fd6c8279c2a8a5` | — |
| `data-v2` | `data/imdb_v4.csv` | 2001（清洗后） | `1dbc6282ccf17c82dc5eaedb684e45fb` | `data-v2` |

- **复现某个数据版本**：

```bash
dvc pull                       # 从 Azure 拉取当前 commit 需要的所有数据
git checkout data-v1
dvc checkout                   # 把工作区数据切换到 data-v1
```

- 详细说明见 [`ml/README_data.md`](ml/README_data.md)，代码 → 数据 → 模型的对应关系见 [`ml/PROVENANCE.md`](ml/PROVENANCE.md)。

> 说明：`.dvc/config` 里只放远端地址（不含密钥），SAS token 放在 `config.local`；CI 不执行 `dvc pull`，因为仓库里没有长期有效的 Azure 凭证。

---

## 4. 模型与实验追踪 (MLflow)

- **追踪后端**：本地 SQLite，`sqlite:///mlflow.db`（可由 `MLFLOW_TRACKING_URI` 覆盖）。
- **每个 run 记录**：`git_commit_sha`、`dvc_dataset_hash`、`data_path`、`max_features`、`ngram_range`、`C`、`min_df`、`sublinear_tf`、`test_f1`，以及模型 artifact（`mlflow.sklearn.log_model` + `.pkl`）。
- **当前生产模型**：`models/model_v1.pkl`，在 `data-v1` 上训练，**test F1 = 0.8328**。
- **实验对比**：

| run | 数据 | C | max_features | F1 |
|-----|------|---|--------------|-----|
| `baseline-v1`（生产） | data-v1 | 1.0 | 10,000 | **0.8328** |
| `improved-tuned` | data-v1 | 2.0 | 30,000 | 0.8212 |
| `improved-v2-cleaned` | data-v2 | 2.0 | 20,000 | 0.7525 |

- 查看实验：`mlflow ui --backend-store-uri sqlite:///mlflow.db`
- 选择理由与调参结论见 [`ml/README_experiments.md`](ml/README_experiments.md)。

---

## 5. CI/CD 自动化流水线

采用 `feature -> dev -> staging -> main` 四层分支模型，`dev` 与 `main` 已开启分支保护（PR 合并前必须通过 CI）。

```text
[feature/*]
    │ Open PR
    ▼
[ dev ]  ──▶ CI：ruff + black + pytest（49 个用例）
    │            + docker build + 容器冒烟（/health 断言 model_loaded、/predict）
    │ Merge PR
    ▼
[ staging ] ──▶ Deploy Staging：读取 STAGING_ENV_FILE → 生成 .env → build ml-app:staging
    │ Merge PR
    ▼
[ main ] ──▶ Deploy Production：读取 PRODUCTION_ENV_FILE → 生成 .env → build ml-app:production
```

| 触发条件 | 工作流 | 做什么 |
|----------|--------|--------|
| 向 `dev` / `staging` / `main` 提 PR | `ci.yml` | lint（ruff）、格式检查（black）、全量 pytest、构建镜像并**启动容器冒烟测试** |
| push 到 `staging` | `deploy-staging.yml` | 用 `staging` environment 的 Secrets 生成 `.env`，构建 `ml-app:staging` |
| push 到 `main` | `deploy-production.yml` | 用 `production` environment 的 Secrets 生成 `.env`，构建 `ml-app:production` |

CI 失败时 PR 无法合并（分支保护）。镜像冒烟测试会真正启动容器并断言 `model_loaded=true`，避免出现「镜像能 build、但里面没有模型」的情况。

---

## 6. 本地运行

```bash
# 1. 安装依赖
python -m venv .venv
.venv\Scripts\activate            # Windows；macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

# 2. 拉取数据（需要 Azure SAS，见 DEPLOYMENT.md）
dvc pull

# 3. 训练（可选，仓库已带 models/model_v1.pkl）
python -m ml.train --config ml/configs/baseline.yaml
python -m ml.train --config ml/configs/improved.yaml

# 4. 启动服务
uvicorn app.main:app --reload --port 8000

# 5. 调用
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"A wonderful, brilliant film.\"}"

# 6. 测试与静态检查
pytest -v
ruff check .
black --check .
```

---

## 7. Docker

```bash
docker build -t ml-app:local .
docker run -d -p 8000:8000 --name ml-app ml-app:local
curl http://127.0.0.1:8000/health          # {"status":"ok","model_loaded":true}
```

镜像基于 `python:3.11-slim`，只拷贝 `app/` 与 `models/`（不拷数据、测试与训练脚本），以非 root 用户运行，并内置 `HEALTHCHECK`。

---

## 8. 已知限制

- 测试集只有 300 条，F1 的波动约 ±0.02，超参调优的收益可能落在噪声范围内。
- `data-v2`（清洗后数据）仍保留了约 5% 的标签翻转噪声，规则清洗无法修复，因此 F1 反而低于使用干净数据的 `data-v1`。
- Azure SAS token 有有效期（当前至 2026-12-12），过期后需要重新生成并更新 `.dvc/config.local`。
- 训练数据与模型都较小，属于课程作业规模；生产化还需要数据校验、模型监控与灰度发布。
- [x] init: CI first run
