# QGCG：实现细节与代码审查指南

QGCG 的全称是 **Question-Guided Core-Group Construction**（问题引导的核心证据组构建）。
代码统一使用 `qgcg`，聊天中写的 `gqcg` 指的是这一实现。
它把“回答问题不可缺少的要求”映射成页面级证据组合，用于评估检索结果是否覆盖完整答案需求。
输入是问题与已经选定的 canonical pages；它本身不检索、不重新解析 PDF，也不生成最终自然语言答案。

本分支以 `d18448b` 为基线，收录工作区当前的 5 个实现脚本和 5 个测试文件。
这是**代码审查包**：完整实验依赖本地冻结数据和模型账号，单独 clone 并不能重现历史模型输出。
本分支没有加入原始语料、运行日志、盲审对应表、账号配置或 `.env.local`。
人工审查尚未完成；`verified` 只表示程序校验和模型核查通过，不代表人工确认的 gold labels。

## 1. 建议阅读顺序

| 文件 | 重点 |
| --- | --- |
| [build_qgcg_groups.py](../scripts/build_qgcg_groups.py) | 全部系统提示词、Pydantic schema、三阶段算法、修复、版本化导出 |
| [demo_qgcg.py](../examples/demo_qgcg.py) | 使用固定模拟响应走通真实算法；无网络、无账号 |
| [test_build_qgcg_groups.py](../tests/test_build_qgcg_groups.py) | 布尔结构、逐字引用、输入保持、定向修复的行为测试 |
| [run_qgcg_pilot.py](../scripts/run_qgcg_pilot.py) | E1/E2 输入选择、调用重试、冻结输入与审计、失败恢复 |
| [qgcg_subscription_transports.py](../scripts/qgcg_subscription_transports.py) | 三个本地账号 CLI 的隔离、请求封装与响应解码 |
| [qgcg_gemini_api.py](../scripts/qgcg_gemini_api.py) | 显式选择的 Gemini API 路径，GPT/Claude 仍走账号 CLI |
| [evaluate_qgcg_pilot.py](../scripts/evaluate_qgcg_pilot.py) | 盲审材料、诊断报告、冻结检索结果的 Boolean recall |

其余测试是 `test_run_qgcg_pilot.py`、`test_qgcg_subscription_transports.py`、
`test_qgcg_gemini_api.py`、`test_evaluate_qgcg_pilot.py`。本分支中的其他 V3 文件来自基线提交。

## 2. 核心流程

```text
question ── A: 提取不可缺少的回答要求
                    │
question + requirements + canonical pages
                    │
              B: 构造证据选项
                    │
      Python: schema / 页面 ID / 逐字引用校验
                    │
question + proposed_groups + canonical pages
                    │
              C: 独立语义核查
                    │
      有可操作问题 → B 定向修复 → Python 校验 → C 再核查
                    │
       导出初始版本、修复历史、剩余问题与 review_status
```

入口 `build_qgcg(question, relevant_pages)` 顺序调用
`model_a_extract_requirements`、`model_b_construct_groups`、`check_page_ids_and_quotes`、
`model_c_verify`、`repair_flagged`、`export_versioned`。

### Pass 1：A 提取要求

A **只接收问题**，不接收页面，以减少“根据现有证据倒推问题要求”的偏差。
采用 omission test：缺少该要求是否会导致问题的主要显式或必要隐含需求没有被回答？
输出非空数组，每项包含 `requirement`、`why_indispensable`。
保留实体、时间、比较维度和限定条件；不把可选背景或单纯的引用格式拆成必答要求。
程序检查字段和重复项；“是否真的不可缺少”仍是语义判断。

### Pass 2：B 构造证据组

B 接收原问题、A 的有序要求和 `page_id -> text` 字典。
必须逐项、逐字、按原顺序保留 A 的两个字段；不能在这一阶段增删要求。
每项增加 `evidence_options`、`source_excerpts`、`status`、`issues`。

`evidence_options` 是析取范式：**外层 OR，内层 AND**。

| JSON | 含义 |
| --- | --- |
| `[["A"], ["B"]]` | A 或 B 中任意一页独立充分 |
| `[["A", "B"]]` | A 与 B 必须一起命中 |
| `[["A"], ["B", "C"]]` | A 独立充分，或者 B 和 C 联合充分 |
| `[]` | 尚无充分证据选项，仅用于 unresolved |

先寻找单页充分证据，再构造互补的多页组合。每个选项要覆盖整个要求，且应是最小充分集合。
相似主题不等于证据等价；时间、对象或条件不兼容的页面不能简单拼起来。
程序能拒绝重复页面、重复选项及已列选项的严格超集，但无法单靠集合关系证明语义上的最小性。

每条 `source_excerpts` 含 `page_id`、`quote`、`explanation`。
`quote` 必须是该 canonical page 原文的连续子串；不做空白规范化，不允许改写或自行插入省略号。
所有证据选项引用的页面都必须有摘录；也允许保留未进入选项的部分支持或矛盾摘录。

### 程序校验与状态

Pydantic v2 使用 strict 类型检查和 `extra="forbid"`，拒绝多余字段。
Pass 1/3 使用 `TypeAdapter` 保持顶层数组。JSON Schema 供模型参考，Python validators 负责实际约束；
导出的 JSON Schema 不能独立执行页面存在性、逐字引用等上下文检查。

- `required`：已有非空充分证据选项；这不是人工认可标记。
- `unresolved`：选项必须是 `[]`，且至少记录一条问题；不能因为证据缺失就删除必答要求。
- `supplementary`：通用 schema 保留此枚举，但本 QGCG 必答要求流程明确拒绝它。

`[[]]` 被拒绝，防止空 AND 被当成自动命中。未知页面、非逐字引用、要求改写或重排都会报错。
程序校验只能证明结构及引用来源完整性，不能证明摘录蕴含要求。

### Pass 3：C 核查与定向修复

C 接收原问题、候选组、组引用 `G0001` 等以及原页面，不接收 B 的聊天历史。
输出问题数组；无问题时返回 `[]`。问题含引用、类型、严重性、依据和推荐动作。
核查包括证据不足、缺少依赖、冗余/过度拆分、遗漏要求、矛盾、布尔逻辑、范围及状态错误。
新增要求须使用 `NEW_REQUIREMENT` + `missing_requirement` + `add`。

默认最多 **1 轮语义修复**（`max_repair_rounds`）。仅 `keep` 问题不会触发修复，
但残留问题仍会使导出状态为 `needs_review`。B 修复后再次执行来源校验并交给 C 独立核查。
未被标记为可编辑的组必须完整保留，且相对顺序不能改变。
合法修复不保证语义已解决；校验失败会保留前一候选并记录错误。
传输/模型调用失败由 runner 记录，不会伪造成功结果。

提示词约束与程序约束的边界也需要审查：例如“只针对遗漏要求增加新需求”和语义最小性，
并非全部都能由 schema 完整证明。模型间可能存在相关偏差，人审仍是必要步骤。

## 3. 输出与可追溯性

核心导出包含 `schema_version`（`qgcg-1.0.0`）、`prompt_version`（`qgcg-prompts-1.0.0`）、
UTC 时间、问题、页面 ID、页面字典的 SHA-256、初始要求/组/问题、最终组/问题、
`repair_history`、`repair_error`、`requirement_references` 和 `review_status`。
任意剩余核查问题、修复错误、非 required 状态或组内问题都会得到 `needs_review`。

实验 runner 进一步冻结输入、实现文件和请求，记录原始响应、解码/校验过程、尝试与运行 manifest。
原始 CLI 输出先落盘再解码；写入采用 flush/fsync，更新文件使用临时文件替换。
已有实验目录受保护。`--resume-failed` 使用冻结输入恢复失败问题，并保留已有成功结果和历史记录。
格式/调用重试（默认 `--max-attempts 3`）与语义修复轮数是不同参数。

## 4. Pilot 与模型配置

默认 pilot 问题为 Q2、Q3；stress checks 使用 Q1、Q6、Q7。

| 组别 | 输入页面来源 | Q2 | Q3 |
| --- | --- | ---: | ---: |
| E0 | 现有 baseline core groups，仅用于离线比较 | — | — |
| E1 | consensus v2 中 `core=true` | 4 | 22 |
| E2 | consensus v2 中 `relevant=true` | 13 | 41 |

62/127 是冻结 pool **全体 query-page membership** 的数量，不是单题页面数，也不是去重页面数。
runner 显式检查这两个全局数量以及 core 是 relevant 的子集，所以不能直接用任意小 CSV 跑 pilot。
任意新数据应调用核心 `build_qgcg`，或另行建立明确版本的新实验配置。

代码默认角色：A=`gpt-5.6-sol`，B=`claude-sonnet-5`，C 的订阅 CLI 参数默认
`gemini-3.1-pro-low`；显式 API 路径使用 `gemini-3.1-pro-preview`。
这些是当前脚本的配置值，不是对账号当下可用模型的保证，也不能仅凭默认常量认定历史运行模型。
实际实验应以各次 manifest、请求和提供方返回的模型身份记录为准。

自动模式依赖本地环境：Codex 在 PATH；Claude 在
`.benchmark-tools/cli/node_modules/@anthropic-ai/claude-code`，账号目录为
`.benchmark-tools/accounts/claude`（JS 入口还需对应 Node runtime）；
Gemini 使用 `.benchmark-tools/antigravity/agy.exe` 与本机 keyring。
这些路径体现原实验的 Windows 环境，不是可移植的通用安装器；账号与二进制不随源码发布。
订阅 transport 过滤 API-key 环境变量，检查账号与模型，不自动切换模型或回退 API。
`--gemini-transport api` 才显式选择 Gemini API，凭据从环境或本地 `.env.local` 读取。
完整 HTTP 内部重试与后端 revision 不一定可见；手动模式的模型选择是操作者声明。

## 5. 无账号运行示例和测试

在本分支仓库根目录执行，Python 3.10+：

```powershell
python -m pip install -r scripts/requirements-qgcg-pilot.txt
python examples/demo_qgcg.py
python -m unittest discover -s tests -p "test_*qgcg*.py" -v
python scripts/build_qgcg_groups.py --schema pass2
```

示例使用虚构页面 A/B/C 和固定模拟模型响应，走真实 schema、三阶段调度及导出。
它展示 `A OR (B AND C)`，**不是一次真实模型实验，也不证明模型判断质量**。
测试也不需要账号或付费模型请求，覆盖核心规则、审计/恢复、传输封装和评估逻辑。

自定义模型接入契约：

```python
from scripts.build_qgcg_groups import configure_qgcg, build_qgcg

def complete(*, role, system_prompt, payload, json_schema):
    # role 为 model_a/model_b/model_c；分别路由到指定模型。
    # 每次使用新请求，不共享对话历史；返回原始 JSON 字符串。
    raise NotImplementedError("接入自己的模型客户端")

configure_qgcg(complete, max_repair_rounds=1)
# result = build_qgcg("问题", [{"page_id": "P1", "text": "canonical 原文"}])
```

核心 CLI 也支持 `--adapter module:function --question-file question.txt --pages pages.json --output result.json`。
输出文件必须尚不存在；adapter 模块需在 Python import path 中。

## 6. 完整实验的复现前提与命令

以下命令需要先取得原冻结数据，且在仓库根目录运行；本分支不附完整数据。
`--data-root` 必须指向真实实验数据根目录。显式传入它也避免旧工作区 `new-V3` 层级差异。

生成阶段需要：

- `benchmark/questions.csv`：`query_id,query`，问题原文保持不变。
- `benchmark/qrels/consensus_qrels_v2.csv`：`query_id,page_id,core,relevant`，完整冻结 pool。
- `benchmark/fair_v2/corpus.jsonl`：canonical 页面，如 `page_id,page_text`；不可用重叠 chunks 临时拼接替代。

```powershell
# 仅冻结输入；manual + prepare-only 不调用模型，也不要求剪贴板/账号可用。
python scripts/run_qgcg_pilot.py --data-root . --transport manual --prepare-only --output-root benchmark/qgcg_prepare_review

# 实际账号调用前检查本机环境。
python scripts/run_qgcg_pilot.py --data-root . --check-accounts

# 调用模型，写入新的实验目录。
python scripts/run_qgcg_pilot.py --data-root . --output-root benchmark/qgcg_rerun_review
```

离线评估还需要原 `benchmark/qgcg_pilot/{E1,E2}` 完整审计目录、
`benchmark/qrels/core_groups.json`、`benchmark/analysis/v2_reviewed_evaluation_manifest.json`、
`benchmark/analysis/core_group_page_comparison_20261008/manifest.json`、
`benchmark/analysis/gpt_v3_comparison_20261007/all_runs_recomputed.csv` 及该清单引用的排名文件。
stress 目录可缺失，但评估会明确报告未完成。manifest/hash 必须与冻结版本一致。
评估器固定读取 `benchmark/qgcg_pilot`，不会自动读取上述示例中的新 rerun 目录。

```powershell
python scripts/evaluate_qgcg_pilot.py --data-root . --output-root benchmark/qgcg_pilot/review
```

评估不会调用模型、重新检索或修改 gold。会生成盲审材料、对照报告和诊断文件。
`blind_key.json`、对照报告（包括 `comparison_report_q2_q3.html`）和诊断会暴露条件，
应由组织者保管；正在做盲审的审查者应先完成盲审再阅读这些材料。
本分支不附这些实验输出。

## 7. CoreGroupRecall 的具体计算

设检索页面集合为 R，某组的证据选项为 O。命中条件为：存在 o∈O，使 o 中每个页面都属于 R。
例：`[["A"],["B","C"]]` 在 R={B} 时不命中，在 R={A} 或 {B,C} 时命中。
这是覆盖一条完整证据路径的指标；不要求取回全部替代选项。

实现同时报告两个分母，避免忽略 unresolved 产生的乐观偏差：

- `CoreGroupRecall_excluding_unresolved` = 命中 required 数 / required 数。
- `CoreGroupRecall_including_unresolved_as_misses` = 命中 required 数 / (required + unresolved 数)。

分母为零时返回 `None`。supplementary 不进入这两个分母。
评估采用预先冻结的 46-run inventory，检查来源 SHA-256、页面/排名重复和排名连续性，
不因分数好坏重新挑选运行。该指标只衡量给定分组下的覆盖，不等同于分组质量或最终答案正确率。

## 8. 人工审查重点

优先确认要求是否完整且必要、每个 OR 分支是否独立充分、AND 页是否确实互补且必需、
摘录是否支持完整范围、unresolved 是否被正确保留，以及修复是否引入语义偏移。
应结合原文判断；通过 schema 与模型 C 不能替代人工证据审查。
