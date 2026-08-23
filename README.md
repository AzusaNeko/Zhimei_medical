# 智美医美智能顾问平台

这是一个个人独立实现的可运行工程项目，聚焦医美知识科普、风险筛查与高风险人工审核，形成一条完整业务链路：

```text
用户咨询
  → 知识科普 Agent（本地 RAG 检索）
  → 风险筛查 Agent（确定性规则）
      ├─ 低/中风险 → 生成有依据的科普回答
      └─ 高风险 → LangGraph interrupt
                    → 医师 approve/reject/edit
                    → Command(resume) 恢复 Workflow
```

> 本仓库不包含任何公司的私有源代码、业务数据或医学规则。它是个人独立实现的工程项目，不是生产医疗系统，不能替代执业医师面诊、诊断或治疗方案。

## 已实现

- FastAPI REST API 与 Swagger UI
- LangGraph `StateGraph`、条件路由和状态管理
- 知识科普 Agent
- 风险筛查 Agent
- 高风险 `interrupt()` 暂停
- 医师审核后 `Command(resume=...)` 恢复
- 本地示例知识库与来源返回
- Dockerfile 和自动测试

为了无需 API Key 即可运行，RAG 默认采用 BM25-like 词法评分和中文字符二元组重合度。生产扩展时可以把 `LocalHybridRetriever` 替换为 BGE-M3 + Milvus + BGE-Reranker。

## 目录

```text
app/
  main.py       # FastAPI 接口
  models.py     # 请求模型
  knowledge.py  # 示例知识库与本地检索器
  workflow.py   # LangGraph Agent Workflow 与 HitL
tests/
  test_workflow.py
```

## 本地运行

要求 Python 3.12+。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

打开 http://127.0.0.1:8000/docs 直接在 Swagger 中测试。

## API

### 低风险咨询

```bash
curl -X POST http://127.0.0.1:8000/consultations \
  -H "Content-Type: application/json" \
  -d '{"question":"透明质酸术后如何护理？","project_type":"注射","profile":{}}'
```

低风险会直接返回 `status=completed`、检索来源和回答。

### 高风险咨询

```bash
curl -X POST http://127.0.0.1:8000/consultations \
  -H "Content-Type: application/json" \
  -d '{"question":"孕期可以做注射项目吗？","project_type":"注射","profile":{"pregnant":true}}'
```

高风险返回 `status=pending_review` 和 `thread_id`。使用该 ID 审核：

```bash
curl -X POST http://127.0.0.1:8000/consultations/替换为thread_id/review \
  -H "Content-Type: application/json" \
  -d '{"action":"reject","note":"建议产后再由医师面诊评估"}'
```

支持 `approve`、`reject`、`edit` 三种审核动作。

## 测试

```bash
pytest -q
```

覆盖检索命中、低风险直达、高风险 interrupt/resume 和中风险流程。

## 面试时应该怎么描述

可以说：

> 我独立实现了一个基于 LangGraph 的医美智能顾问项目，将知识检索、确定性风险规则和 Human-in-the-Loop 串成可运行工作流。高风险状态通过 interrupt 暂停，使用 thread_id 从 checkpoint 恢复，并支持医师批准、拒绝或修改。

面试时应明确这是个人独立实现的公开项目，不要声称它是公司的生产源代码，也不要声称当前版本已经使用 Milvus、BGE-M3、PostgreSQL 或真实医学规则。

## 后续扩展

- BGE-M3 dense/sparse + Milvus Hybrid Search
- BGE-Reranker 和父子块切分
- PostgreSQL checkpointer
- Redis 会话与审核任务缓存
- JWT 角色权限、审核幂等和审计日志
- SSE 流式输出与 RAG 评测集

开源灵感与来源见 [NOTICE.md](NOTICE.md)。
