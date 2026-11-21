# 评测运行说明

> 本文只讲「怎么跑、产物在哪、什么不能比」。
> 结果数字以 `EVALUATION_REPORT.md` 为准，本文不重复抄写，避免两处对不上。

## 1. 环境要求

| 项 | 要求 |
|---|---|
| Python | 3.12（项目 `.venv`） |
| 依赖 | `requirements.txt` 已安装 |
| 密钥 | `.env` 中 `DEEPSEEK_API_KEY` 必填；`ZHIPU_API_KEY` / `QWEN_API_KEY` 用于降级测试 |
| 模型 | `models/bge-small-zh-v1.5/` 本地存在（约 95MB，不进 git） |
| 网络 | 需访问模型 API，延迟约 0.5–3s/次 |

## 2. 运行方式

### 全跑（推荐）

```bash
.venv/Scripts/python.exe scripts/run_all_evaluations.py
```

十个模块跑完才会覆盖 `EVALUATION_REPORT.md`。

### 只跑某几个模块

```bash
.venv/Scripts/python.exe scripts/run_all_evaluations.py intent rag_retrieval
```

可选模块名：

```
intent  tool_selection  tool_arguments  rag_retrieval
rag_answer  agent_e2e  ticket_flow  adversarial  safety  resilience
```

**局部运行不会覆盖总报告。** 报告写到 `results/<run_id>/partial_report.md`，
并打印缺了哪几个模块。这条是踩过坑之后加的：

> 之前为了省时间只跑 `rag_retrieval` + `rag_answer`，
> 一份 60 条样本的局部结果把 288 条的总报告整份冲掉，
> 文件名还叫「总报告」，从外面完全看不出它是局部的。

### 汇总已有结果，不重跑

```bash
.venv/Scripts/python.exe scripts/run_all_evaluations.py --from-results
```

从 `results/` 里取每个模块**最新**的一份拼成报告。注意这种方式拼出来的报告可能跨多个批次，
报告第四节「运行批次」会把这件事写明，并核对各批次的知识库版本、温度、Embedding 是否一致。

### 单独跑一个

```bash
.venv/Scripts/python.exe scripts/evaluate_intent.py
```

## 3. 样本数与耗时

| 模块 | 样本 | 是否调用模型 | 约耗时 |
|---|---:|---|---|
| 意图识别 | 49 | 是（每条 1 次） | 60–90s |
| 工具选择 | 30 | 是，工具打桩 | 100–150s |
| 工具参数 | 30 | 是，工具打桩 | 100–150s |
| RAG 检索 | 30 | 否（本地 embedding） | 30–60s |
| RAG 回答 | 30 | 是 | 60–120s |
| Agent 端到端 | 30 | 是，约 90 次 | 200–400s |
| 售后申请写操作 | 16 | 是，约 30 次 | 100–160s |
| 对抗性输入 | 54 | 是 | 250–400s |
| 权限与安全 | 11 | 部分（含 REST 直连探测） | 40–70s |
| 容错 | 8 | 部分（其余 mock 注入） | 20–40s |

全量一次约 15 分钟，随模型服务响应波动。

> **四种验证方式不能混为一谈**：真实模型调用（意图、RAG 回答、端到端、对抗）、
> 桩函数替换（工具选择、工具参数，只验模型「选哪个工具、传什么参数」）、
> 本地检索（RAG 检索，不调模型）、mock 故障注入（容错）。
> 报告里每个模块的这两列是分开写的，不要合并成一句「全部真实调用」。

## 4. 产物位置

```
results/<run_id>/
├── intent.json
├── tool_selection.json
├── tool_arguments.json
├── rag_retrieval.json
├── rag_answer.json
├── agent_e2e.json
├── ticket_flow.json
├── adversarial.json
├── safety.json
├── resilience.json
├── summary.json
└── partial_report.md        # 仅局部运行时生成

EVALUATION_REPORT.md         # 总报告，只有十个模块全跑完才会覆盖
```

`run_id` 格式：`20261121T161816-616013`（时间戳 + 随机后缀）。

每个 JSON 的固定字段：

```
run_id / timestamp / evaluation_name / dataset_version / environment
total_cases / passed_cases / failed_cases
metrics / case_results / error_details / limitations
```

`environment` 里记录了知识库版本、温度、Embedding、模型别名，
报告第四节用它核对跨批次一致性。

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

### 多轮

「多轮」在两个模块里是两条不同的指标，别混：

| 模块 | 字段 | 含义 |
|---|---|---|
| 意图识别 | `multi_turn_accuracy` | 15 条带历史的指代消解样本 |
| Agent 端到端 | `multi_turn_pass_rate` | 3 条多轮任务，看整条链路是否完成 |

## 6. 新增样本

1. 直接编辑 `scripts/evaluation/datasets/<模块>.jsonl`，一行一条 JSON
2. 以 `#` 开头的行会被跳过，可写注释
3. **改完样本，`dataset_version` 会自动变化**，与旧结果不可直接比较——这是有意设计
4. 版本号是**整文件**的 sha256，连 `#` 注释也算。所以只改注释也会让版本变。
   报告第三节用的是「那次运行记录的版本」，不是生成报告时重新算的；
   两者不一致时会标「文件已变更」——意思就是跑完之后样本文件又被改过
5. 不要为了让指标好看而删减困难样本

## 7. 复现性说明

| 项 | 是否可固定 |
|---|---|
| 数据集 | ✅ 有内容哈希版本号 |
| Python 与依赖版本 | ✅ 记录在 `environment` |
| 知识库版本 | ✅ sha256 前 12 位记录在 `environment` |
| Embedding 模型 | ⚠️ 路径固定，但未记录文件哈希 |
| LLM 模型 | ❌ `deepseek-flash` 是服务端别名，非具体版本 |
| Prompt 版本 | ❌ 未做版本管理（提示词硬编码在源码） |
| 温度 | ✅ 显式设为 0（`DEFAULT_TEMPERATURE`，可用 `SHOPMATE_TEMPERATURE` 覆盖） |
| 采样次数 | ❌ 单次采样，无重复运行 |

温度固定为 0 只消除了本地这一侧的随机源，服务端仍可能有非确定性，
所以：**同一数据集两次运行的指标会有波动，不能把单次结果当作稳定性能。**
