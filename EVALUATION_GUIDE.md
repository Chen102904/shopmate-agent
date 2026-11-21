# 评测运行说明

## 1. 环境要求

| 项 | 要求 |
|---|---|
| Python | 3.12（项目 `.venv`） |
| 依赖 | `requirements.txt` 已安装 |
| 密钥 | `.env` 中 `DEEPSEEK_API_KEY` 必填；`ZHIPU_API_KEY` / `QWEN_API_KEY` 用于降级测试 |
| 模型 | `models/bge-small-zh-v1.5/` 本地存在（约 95MB，不进 git） |
| 网络 | 需访问模型 API，延迟约 0.5–3s/次 |

## 2. 运行方式

### 全跑

```bash
.venv/Scripts/python.exe scripts/run_all_evaluations.py
```

### 只跑某几个模块

```bash
.venv/Scripts/python.exe scripts/run_all_evaluations.py intent rag_retrieval
```

可选模块名：

```
intent  tool_selection  tool_arguments  rag_retrieval
rag_answer  agent_e2e  ticket_flow  adversarial  safety  resilience
```

### 单独跑一个

```bash
.venv/Scripts/python.exe scripts/evaluate_intent.py
```

## 3. 预计耗时

| 模块 | 样本 | 模型调用次数 | 约耗时 |
|---|---:|---:|---|
| 意图识别 | 46 | 46 | 45–70s |
| 工具选择 | 30 | 60 | 100–150s |
| 工具参数 | 30 | 60 | 100–150s |
| RAG 检索 | 30 | 0（本地向量） | 30–60s |
| RAG 回答 | 30 | 25 | 60–120s |
| Agent 端到端 | 30 | 约 90 | 200–400s |
| 售后申请写操作 | 14 | 约 25 | 90–150s |
| 安全探测 | 11 | 9 | 40–70s |
| 容错 | 8 | 3（其余 mock） | 20–40s |

> 耗时随模型服务响应波动，以上为实测区间参考。

## 4. 产物位置

```
results/<run_id>/
├── intent.json
├── tool_selection.json
├── tool_arguments.json
├── rag_retrieval.json
├── rag_answer.json
├── agent_e2e.json
├── safety.json
├── resilience.json
└── summary.json
EVALUATION_REPORT.md        # 总报告（每次运行覆盖写）
```

`run_id` 格式：`20261120T114322-dfea28`（时间戳 + 随机后缀）。

每个 JSON 的固定字段：

```
run_id / timestamp / evaluation_name / dataset_version / environment
total_cases / passed_cases / failed_cases
metrics / case_results / error_details / limitations
```

## 5. 数据口径

### 三层计分（工具参数）

| 层级 | 分子 | 分母 |
|---|---|---|
| 参数级 | 正确的 (tool, key, value) 三元组 | 预期三元组总数 |
| 调用级 | 完全匹配的 tool call | 预期调用总数 |
| 样本级 | 完全匹配的样本 | 样本总数 |

**不可混用**。报告里出现的「准确率」必须注明是哪一层。

### 检索指标

- **Hit@K**：正确内容是否出现在前 K 条
- **MRR**：正确内容排名的倒数，取平均
- **Precision@3**：前 3 条中相关条目占比
- 命中判定用「内容锚点」（keywords 全部出现），不用硬编码 chunk_id

### 安全与容错

- 安全是「探测现状」，`safe_rate` 低代表缺口存在，不代表回归失败
- 容错是「mock 注入」，不等于真实故障演练

## 6. 新增样本

1. 直接编辑 `scripts/evaluation/datasets/<模块>.jsonl`，一行一条 JSON
2. 以 `#` 开头的行会被跳过，可写注释
3. **改完样本，`dataset_version` 会自动变化**，与旧结果不可直接比较——这是有意设计
4. 不要为了让指标好看而删减困难样本

## 7. 复现性说明

| 项 | 是否可固定 |
|---|---|
| 数据集 | ✅ 有内容哈希版本号 |
| Python 与依赖版本 | ✅ 记录在 `environment` |
| Embedding 模型 | ⚠️ 路径固定，但未记录文件哈希 |
| LLM 模型 | ❌ `deepseek-flash` 是服务端别名，非具体版本 |
| Prompt 版本 | ❌ 未做版本管理（提示词硬编码在源码） |
| 随机性 | ⚠️ 未显式设置 temperature，按服务端默认 |
| 采样次数 | ❌ 单次采样，无重复运行 |

因此：**同一数据集两次运行的指标会有波动**，不能把单次结果当作稳定性能。
