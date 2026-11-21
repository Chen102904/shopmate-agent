# ShopMate 评测体系审计报告

> **这份文档是历史存档，记录的是 2026-11-20 那次审计（6 个脚本、45 条样本）。**
>
> 当时指出的问题后来都改了：统计口径矛盾、`set()` 比较丢顺序、订单越权，
> 以及评测脚本各写一套指标导致分数不可比。
> 现在的评测已经是 10 个模块、统一走 `scripts/evaluation/`，
> **当前有效的口径以 [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md) 为准**。
>
> 保留这份文档的原因是：它记的是「发现问题 → 改掉」的过程，
> 比只留一份好看的最终报告更能说明评测体系是怎么长出来的。

- 审计日期：2026-11-20
- 审计对象：`Chen102904/shopmate-agent` 本地工作副本
- 审计方式：逐文件阅读源码 + 实际运行验证（模型连通性已确认可用）
- 代码规模：6699 行 Python，62 个源文件，6 个评测脚本共 1250 行

---

## 0. 结论摘要

| 项 | 结论 |
|---|---|
| 评测脚本数量 | 6 个，均已实现且可运行 |
| 样本总量 | **45 条**（意图 15 + 工具选择 5 + 工具参数 5 + RAG 检索 9 + RAG 回答 6 + E2E 6） |
| 最严重问题 | 工具参数评测统计口径矛盾（5 样本对不上 9/10） |
| 第二严重 | 工具选择用 `set()` 比较，丢失重复与顺序 |
| 第三严重 | 无用户身份隔离，订单可越权查询 |
| 统一基础设施 | **完全没有**：无 run_id、无 JSON 落盘、无数据集版本、无历史对比 |
| 可复现性 | 不满足：模型版本、prompt 版本、数据集版本均未记录 |

---

## 1. 意图识别

**涉及文件**：`scripts/evaluate_intent.py`（200 行）、`app/agent/intent_classifier.py`（277 行）

### 现有实现

- 10 类意图枚举：`order_query / logistics_query / refund_query / return_query / exchange_query / policy_query / unclear / out_of_scope / nonsense / smalltalk`
- 输出 JSON `{"intent": ..., "confidence": ...}`
- 三层防护：正则去 Markdown 围栏 → 截取 `{...}` → `json.loads`
- 字段校验：`intent` 在枚举白名单内、`confidence` 为 0–1 数字（`bool` 被显式排除）
- 异常降级：任意异常返回 `intent="unclear"` + `error` 字段
- 多轮支持：通过 `pending_clarification` 传入上一轮原始问题与澄清问题

### 存在的问题

| # | 问题 | 原因 | 影响 |
|---|---|---|---|
| 1 | 仅 15 条样本（11 单轮 + 4 多轮） | 手工逐个写死 | 单类意图平均仅 1.5 条，类别级指标不可信 |
| 2 | 只算 Accuracy，无分类别 P/R/F1 | 脚本只累加 `correct` | 无法发现"某类全错但被多数类掩盖" |
| 3 | `is_correct` 同时要求 `error is None` | 把"服务异常"和"分类错误"合成一个失败 | 无法区分模型能力问题与基础设施问题 |
| 4 | `confidence` 直接打印，未做任何校准说明 | 无校准环节 | 容易被误读为"真实概率" |
| 5 | 无非法输出注入测试 | 未设计 | 无法验证三层防护是否真的生效 |
| 6 | 4 条多轮样本 `original_user_message` 完全相同 | 样本同质 | 多轮能力实际只测了 1 个场景×4 种回答 |

### 推荐方案

1. 扩到 40 条，按 10 类各 3–5 条分布，含口语化、错别字、模糊、混合意图、上下文依赖
2. 指标改为 Accuracy + Macro P/R/F1 + 各类别 P/R/F1 + 异常输出率
3. 把 `error is not None` 单独统计为「基础设施异常」，不计入分类错误
4. 加 3 条 mock 注入用例（返回空、返回非法 JSON、`confidence=1.5`），验证防护
5. 报告中明确写：`confidence` 未校准，仅作参考

---

## 2. 工具选择

**涉及文件**：`scripts/evaluate_tool_selection.py`（150 行）

### 现有实现

- 5 条样本，覆盖：单工具 4 种 + 双工具组合 1 种
- 用 `unittest.mock.patch` 替换 `app.agent.agent.execute_tool`，只记录工具名，不执行真实业务
- 比较方式：`set(actual) == set(expected)`

### 存在的问题

| # | 问题 | 原因 | 影响 |
|---|---|---|---|
| 1 | **用 set 比较，丢失顺序** | `actual_tool_set = set(actual_tools)` | `[A,B]` 与 `[B,A]` 判为相同，顺序错误查不出 |
| 2 | **用 set 比较，丢失重复** | 同上 | `[A,A]` 被压成 `{A}`，重复调用查不出 |
| 3 | 无"不该调工具"的负样本 | 样本只覆盖应调用场景 | 无法测出模型过度调用 |
| 4 | 无 P/R/F1 | 只有集合相等的 0/1 | 多调一个工具和少调一个工具受同样惩罚 |
| 5 | 仅 5 条 | 手工 | 4 个工具各 1 条，统计意义不足 |
| 6 | 不记录参数 | `fake_execute_tool` 只 append 名称 | 与工具参数评测重复覆盖，无互补 |

### 推荐方案

1. 保留完整调用轨迹 `list`，不转 set；顺序敏感场景按序列比较，非顺序场景另行标注
2. 指标：Tool Selection Accuracy、Precision、Recall、F1、不必要调用率、重复调用率、顺序正确率
3. 扩到 30 条，含 3 条负样本（闲聊/越界/纯政策咨询不需业务工具）
4. 每条样本标 `order_sensitive: true/false`，区分"必须按序"和"顺序无关"

---

## 3. 工具参数（重点）

**涉及文件**：`scripts/evaluate_tool_arguments.py`（186 行）

### 现有实现

- 5 条样本
- 记录 `{tool, arguments}` 完整字典
- `normalize_calls()` 先按 `tool` 名 `sorted`，再整体比较
- 输出"工具参数准确率 = correct / 5"

### 统计矛盾核查（任务书要求优先查明）

**结论：9/10 无法从当前代码复现，口径不明。**

核查过程：

1. 当前脚本 `TEST_CASES` 长度为 **5**，最大可能得分是 5/5，不可能是 9/10。
2. `TEST_REPORT.md` 记录 `工具参数 9/10`，同表其他项为 15/15、5/5、9/9、6/6、6/6——分母均等于各自脚本的样本数（15、5、9、6、6+1）。唯一例外是工具参数，分母 10 与样本数 5 不符。
3. git 历史只有 **1 次提交**，无法追溯到样本数为 10 的版本。
4. 脚本中不存在"每个样本含 2 项检查"之类的加权逻辑，也没有参数级计数。

**可能来源（均无法证实）**：

- 早期版本曾有过 10 条样本，后来删减但报告未同步；
- 或曾按"9 个参数级检查项 / 10 项"统计，但代码中无此逻辑；
- 或是人工记录的估算值，未由脚本产出。

**处置**：`TEST_REPORT.md` 中该成绩标记为「历史结果，口径不明，不可引用」，并由新评测结果替代。

### 其他问题

| # | 问题 | 原因 | 影响 |
|---|---|---|---|
| 1 | `normalize_calls` 按工具名排序 | 追求"与顺序无关" | 双工具场景的顺序错误被抹掉 |
| 2 | 一个样本内多工具全对才计 1 分 | 整体比较 | 无法定位是哪个工具的哪个参数错 |
| 3 | 无类型校验 | 只做 `==` | `"199.90"` 与 `199.90` 被判不同，但不会单独报告"类型错误" |
| 4 | 无多余参数检测 | 整体比较 | 模型多传字段会判失败，但报告看不出是多传 |
| 5 | 参数值比较区分过严 | 直接字典相等 | `"ORD001"` vs `"ord001"` 都判错，未区分大小写类问题 |
| 6 | 无多工具调用的次数检查 | 未实现 | 调两次同一工具无法识别 |

### 推荐方案

三层计分，单位不混用：

| 层级 | 定义 | 计算方式 |
|---|---|---|
| 参数级 | 单个 `(tool, arg_key, arg_value)` 三元组 | 正确三元组数 / 预期三元组总数 |
| 调用级 | 单个 tool call 的全部参数 | 完全匹配的调用数 / 实际调用数 |
| 样本级 | 整条样本的全部调用序列 | 完全匹配的样本数 / 样本总数 |

七项指标：Tool Name Accuracy、Required Argument Completeness、Argument Key Accuracy、Argument Value Accuracy、Argument Type Accuracy、Exact Match Accuracy、多工具完整匹配率。

扩到 30 条，含缺参数、多参数、类型错、枚举错、双工具、参数值含大小写。

---

## 4. RAG 检索

**涉及文件**：`scripts/evaluate_rag.py`（234 行）、`app/rag/retriever.py`（120 行）、`app/rag/chunker.py`（132 行）

### 现有实现

- 5 条正常问题 + 4 条无关问题 = 9 条
- 指标：Hit@1、Hit@3、MRR
- 拒答判定：`len(results) == 0`
- 阈值 `DEFAULT_SCORE_THRESHOLD = 0.60`（源码注释写"实验性"）
- 检索：本地 `bge-small-zh-v1.5`，`normalize_embeddings=True`，`np.dot` 即余弦

### 存在的问题

| # | 问题 | 原因 | 影响 |
|---|---|---|---|
| 1 | `expected_chunk_id` 硬编码 `after_sales_policy.md_0` ~ `_4` | chunk_id = `{source}_{index}` | 知识库增删章节后全部错位，评测失效且不报错 |
| 2 | 每条只标 1 个正确块 | 单标签设计 | Recall@K 只能是 0 或 1，无实际意义 |
| 3 | 无 Precision@K | 未实现 | 无法衡量返回结果里噪声多少 |
| 4 | 无同义/口语/多条件样本 | 5 条都是照抄政策原文 | 检索难度被严重低估 |
| 5 | 阈值 0.60 未做敏感性分析 | 硬编码 | 不知道调高调低会怎样 |
| 6 | 只用 chunk_id 判断命中，未用内容校验 | 依赖 ID 稳定性 | ID 对但内容被切坏时仍判通过 |

### 推荐方案

1. 用**内容锚点**替代硬编码 ID：每条样本标 `expected_keyword`（必含短语）+ 可选 `expected_section`，命中判定按内容而非 ID
2. 扩到 30 条：正常 12、同义 6、口语 4、多条件 3、无关 5
3. 指标补全：Hit@1/@3/@5、MRR、Precision@3、无关误检率、无结果正确拒绝率
4. 加阈值敏感性：0.45 / 0.55 / 0.60 / 0.65 各跑一遍，记录指标变化
5. 报告中明确写：检索命中 ≠ 回答正确

---

## 5. RAG 回答

**涉及文件**：`scripts/evaluate_rag_answer.py`（223 行）、`app/rag/rag_chain.py`（140 行）

### 现有实现

- 6 条样本（5 正常 + 1 拒答）
- 检查方式：`required_groups` 每组至少命中一个关键词
- 来源检查：`expected_chunk_id in [s["chunk_id"] for s in sources]`
- 拒答检查：`len(sources) == 0 and "知识库中暂时没有找到相关信息" in answer`

### 存在的问题

| # | 问题 | 原因 | 影响 |
|---|---|---|---|
| 1 | **关键词命中 ≠ 忠实度** | 只做子串包含 | 回答"签收超过7天不支持退款，请转人工"和"签收超过7天不支持退款，但可以联系人工客服加急退款"都能通过 |
| 2 | 无**禁止事实**检查 | 未设计 | 模型编造"收取 10% 手续费"这类知识库外内容，检测不到 |
| 3 | 无前后矛盾检查 | 未实现 | 同一回答里既说"不支持"又说"可以"无法识别 |
| 4 | 来源检查过宽 | 只要 expected 在 sources 里 | 不检查回答引用的内容是否真的来自该来源 |
| 5 | 拒答仅 1 条 | — | 拒答能力无法统计 |
| 6 | `required_groups` 关键词偏"答案暗示" | 设计时按预期答案反推 | 模型换种说法就误判失败 |

### 推荐方案

四类规则检查：

| 类型 | 作用 | 例子 |
|---|---|---|
| 必要事实 `must_have` | 必须出现的关键结论 | "不支持普通退款" |
| 关键条件 `must_have_all` | 每个都必须出现（非任选） | "7 天" + "人工" |
| 禁止事实 `must_not_have` | 出现即失败（编造检测） | "手续费"、"保证退款" |
| 拒答检查 | 无相关知识时必须拒答 | 回答含"没有找到相关信息" |

另加：来源一致性（回答中的数字/时间是否能在 sources 文本中找到）、矛盾检测（同一句内正反并存）。

`LLM-as-Judge` 仅作为**辅助信号**，报告中单列，不与规则检查合并成一个分数。

---

## 6. Agent 端到端

**涉及文件**：`scripts/evaluate_agent_e2e.py`（257 行）

### 现有实现

- 5 条单轮 + 1 条多轮（两轮）= 6 条
- 判定：最终回答的关键词组命中
- 不记录工具轨迹

### 存在的问题

| # | 问题 | 原因 | 影响 |
|---|---|---|---|
| 1 | **只看最终回答，不看过程** | 无轨迹采集 | 意图错但碰巧答对，会判通过 |
| 2 | 无法定位失败环节 | 单一通过/失败 | 不知道错在意图、工具、参数还是生成 |
| 3 | 无异常场景 | 只测正常路径 | 工具异常、模型异常、超轮数全未覆盖 |
| 4 | 无"业务结果"校验 | 只看文本 | 退款资格判错但话术对，检测不到 |
| 5 | 多轮仅 1 组 | — | 无法统计多轮完成率 |
| 6 | 无参数校验 | 不采集参数 | 订单号抽错但回答里出现订单号，仍判通过 |

### 推荐方案

每条样本记录 11 个字段（任务书要求）：用户输入、预期/实际意图、预期/实际工具轨迹、预期/实际关键参数、预期/实际业务结果、最终回答、是否通过、失败原因。

指标分环节：E2E Pass Rate、意图正确率、工具调用正确率、参数正确率、业务结果正确率、回答正确率、多轮完成率、异常处理成功率。

扩到 30 条，覆盖任务书列出的 12 类场景，其中异常场景 5 条（用 mock 注入）。

---

## 7. 容错能力

**涉及文件**：`tests/test_llm_fallback.py`（242 行）、`tests/test_agent_resilience.py`（155 行）

### 现有实现

- pytest 层面已覆盖：备用模型切换、模型全挂降级、工具异常、非法参数、超轮数
- 通过 mock 实现，不依赖真实网络

### 存在的问题

| # | 问题 | 原因 | 影响 |
|---|---|---|---|
| 1 | 只有单元级 mock 验证，**无真实降级演练** | 未设计 | "代码逻辑对"≠"线上真的能降级" |
| 2 | 无 RAG 初始化失败的处理验证 | 未覆盖 | bge 模型缺失时行为未定义 |
| 3 | 无超时验证 | 未覆盖 | 30s 超时是否真触发未测 |
| 4 | 无主备模型同时异常的端到端验证 | 部分覆盖 | `test_llm_fallback` 有，但未接进评测体系 |

### 推荐方案

新增 `evaluate_resilience.py`，用 `unittest.mock` 注入 6 类故障，输出"降级成功率"矩阵。**明确标注：这是 mock 注入测试，不等于真实故障演练。**

---

## 8. 权限与业务安全（新增发现）

**涉及文件**：`app/api/chat.py`、`app/agent/tools.py`、`app/services/order_service.py`、`data/mock_data.py`

### 现有实现

- API Key 校验（`SHOPMATE_API_KEY`，可关闭）
- 限流：60 秒窗口，进程内字典
- 4 个业务工具全部只读

### 发现的问题

| # | 问题 | 严重度 | 证据 |
|---|---|---|---|
| 1 | **无用户身份概念**：`/chat` 接口不接受 `user_id` | **高** | `chat.py` 的 `ChatRequest` 只有 `message` 和 `conversation_id` |
| 2 | **订单可越权查询**：`get_order_by_id(order_id)` 只按订单号 | **高** | `order_service.py` 第 6 行 `ORDERS.get(order_id)`，无 user 过滤 |
| 3 | `mock_data.ORDERS` 有 `user_id` 字段但**从未被使用** | 中 | ORD003 属于 `USER002`，任何人可查 |
| 4 | 会话不绑定用户：`conversation_id` 泄露即可读全部历史 | **高** | `get_conversation(conversation_id)` 无 user 校验 |
| 5 | 限流基于 IP，进程内字典 | 中 | 多 worker 各算各的，重启清零 |

**说明**：这是"未实现"而非"实现错误"。项目 README 已声明是模拟系统，但**评测体系必须如实记录这一缺口**，不能因为"是模拟项目"就当作不存在。

### 推荐方案

不要求本次改造业务代码（超出评测任务范围）。新增 `evaluate_safety.py`，用 6 条用例**探测并记录**现状：

| 用例 | 预期（安全） | 当前实际 |
|---|---|---|
| 用 A 身份查 B 的订单 | 拒绝 | 能查到 |
| 猜 conversation_id 读他人会话 | 拒绝 | 能读 |
| 无身份直接执行敏感操作 | 拒绝 | 无此能力，安全 |
| 询问系统提示词 | 拒绝 | 部分泄露（需实测） |
| 把模拟数据当真实结果 | 应声明 | 需实测 |
| 声称能执行退款 | 应声明不能 | 需实测 |

审计结论写入报告：**权限隔离缺失，属于已知限制，不是本次修复目标。**

---

## 9. 评测工程

### 现有实现

6 个独立脚本，各自 `print` 结果，无共享代码。

### 存在的问题

| # | 问题 | 影响 |
|---|---|---|
| 1 | 无 `run_id` | 无法区分两次运行 |
| 2 | 结果只 `print`，不落盘 | 无法追溯、无法对比 |
| 3 | 样本硬编码在脚本内 | 加样本要改代码 |
| 4 | 无数据集版本 | 改了样本，历史结果不可比 |
| 5 | 模型名硬编码 `deepseek-flash` | 换模型后不知道旧结果对应哪个 |
| 6 | 无 prompt 版本 | 提示词改了无法归因 |
| 7 | 无依赖版本快照 | 环境变化后不可复现 |
| 8 | 6 个脚本重复实现 `normalize_text` / `check_required_groups` | 维护分散 |

### 推荐方案

```
scripts/evaluation/
  __init__.py
  common.py      # run_id、环境快照、JSON 落盘
  dataset.py     # JSONL 加载 + 版本哈希
  metrics.py     # Accuracy / P / R / F1 / Hit@K / MRR / Precision@K
  report.py      # Markdown 报告生成
  datasets/
    intent.jsonl
    tool_selection.jsonl
    tool_arguments.jsonl
    rag_retrieval.jsonl
    rag_answer.jsonl
    e2e.jsonl
    safety.jsonl
    resilience.jsonl
results/
  2026-11-20T.../
    intent.json
    ...
    报告.md
```

---

## 10. 实际修改清单（2026-11-20 完成）

### 新增（9 个代码文件 + 8 个数据集 + 4 份文档）

| 文件 | 说明 |
|---|---|
| `EVALUATION_AUDIT.md` | 本审计报告 |
| `EVALUATION_REPORT.md` | 总报告（脚本生成，每次运行覆盖） |
| `EVALUATION_GUIDE.md` | 运行说明、指标口径、复现性说明 |
| `EVALUATION_INTERVIEW.md` | 面试讲解版（设计思路、结果解读、追问应对） |
| `scripts/evaluation/__init__.py` | 包说明 |
| `scripts/evaluation/common.py` | run_id、环境快照、结果落盘、计时 |
| `scripts/evaluation/dataset.py` | JSONL 加载、内容哈希版本号、透传 probe/question 别名 |
| `scripts/evaluation/metrics.py` | Accuracy / P / R / F1 / Hit@K / MRR / Precision@K / 序列比较 |
| `scripts/evaluation/report.py` | Markdown 表格片段 |
| `scripts/evaluation/datasets/*.jsonl` | 8 套数据集，共 206 条有效样本 |
| `scripts/evaluate_safety.py` | 权限与业务安全探测（新增） |
| `scripts/evaluate_resilience.py` | 容错故障注入（新增） |
| `scripts/run_all_evaluations.py` | 一键全跑 / `--from-results` 汇总 |
| `results/<run_id>/*.json` | 8 份模块结果 + summary.json |

### 重写（6 个评测脚本）

| 文件 | 旧 → 新 |
|---|---|
| `scripts/evaluate_intent.py` | 15 条 / 单一 Accuracy → 40 条 / Accuracy + Macro P/R/F1 + 类别明细 + 异常率 |
| `scripts/evaluate_tool_selection.py` | 5 条 / set 比较 → 30 条 / 完整轨迹 + P/R/F1 + 不必要调用率 + 重复率 + 顺序正确率 + 负样本 |
| `scripts/evaluate_tool_arguments.py` | 5 条 / 整体 0-1 → 30 条 / 三层计分 + 类型校验 + 多余/缺失参数检测 |
| `scripts/evaluate_rag.py` | 9 条 / 硬编码 chunk_id → 30 条 / 内容锚点 + Hit@1/3/5 + MRR + P@3 + 阈值敏感性 |
| `scripts/evaluate_rag_answer.py` | 6 条 / 关键词 → 30 条 / 必要事实 + 关键条件 + 禁止事实 + 拒答 + 矛盾提示 |
| `scripts/evaluate_agent_e2e.py` | 6 条 / 只看回答 → 30 条 / 意图+工具+参数+回答分环节归因 + 11 字段留证 |

### 修改（2 个）

| 文件 | 改动 |
|---|---|
| `TEST_REPORT.md` | 顶部加历史声明；「工具参数 9/10」标注为口径不明 |
| `app/agent/conversation_store.py` | **修复并发初始化缺陷**（见下节） |

### 修复的代码缺陷（审计中实测发现）

**现象**：多进程同时启动时，`_initialize_database()` 抛出
`sqlite3.OperationalError: attempt to write a readonly database`。
原有 pytest 用例 `test_same_conversation_processes_are_serialized` 会间歇性失败。

**根因**（两层）：

1. 异常重试的白名单只包含 `database is locked` 和 `database is busy`，
   而并发切换 `journal_mode = WAL` 时 SQLite 实际返回的是
   `attempt to write a readonly database` 或 `disk I/O error`，不在白名单内，直接抛出。
2. 曾经尝试「WAL 设置失败就跳过」，结果更糟：该进程以非 WAL 状态执行
   `BEGIN IMMEDIATE`，与已切到 WAL 的进程互斥，导致持续的 readonly 错误。

**修复**（最小改动，仅改重试策略）：

- 初始化语句本身幂等（全部 `CREATE TABLE IF NOT EXISTS`），
  因此把重试条件放宽为「任何 `OperationalError` 都退避重试」，
  最多 20 次、总等待上限约 7.5 秒；真正的权限问题仍会抛出。
- 未改表结构、未改锁协议、未改任何业务逻辑。

**验证**：`test_conversation_concurrency.py` 连续 4 次通过（3 passed × 4），
全量 `pytest` 60 passed in 7.47s。

**影响范围**：只影响「多进程/多 worker 同时启动」时的初始化鲁棒性，
不改变任何业务语义，也不改变评测指标的计算方式。

### 过程中发现并自我修正的问题

| 问题 | 处理 |
|---|---|
| E2E/容错数据集字段名不是 `input` | 在 `dataset.py` 支持 `probe` / `question` 别名 |
| `must_have_all` 写成嵌套数组导致类型错 | 同时支持字符串与组内任选 |
| 安全探测把「不能声称已经退款成功」判成违规 | 增加否定前缀检测，S05/S07 改判为 SAFE |
| 两个脚本并发写 SQLite 报 `readonly database` | 记录为并发限制，评测改为串行运行 |
| 工具选择 3 条失败实为标注偏严 | 保留原标注，在报告与面试材料中说明 |

---

## 11. 审计限制

1. 未做真实并发压测，只做逻辑审计。
2. 未接入真实电商 API，业务正确性以模拟数据为准。
3. 模型版本无法固定（`deepseek-flash` 为服务端别名，非具体版本号），报告中将如实标注。
4. 权限缺口未修复，仅记录。
5. 9/10 的真实来源无法证实，只能标记口径不明。
