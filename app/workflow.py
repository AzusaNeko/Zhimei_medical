from __future__ import annotations

from typing import Any, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .knowledge import retriever


class AdvisorState(TypedDict, total=False):
    question: str
    project_type: str
    profile: dict[str, Any]
    sources: list[dict]
    risk_level: str
    risk_reasons: list[str]
    review_decision: dict[str, str]
    answer: str


def knowledge_agent(state: AdvisorState) -> AdvisorState:
    return {"sources": retriever.search(state["question"], top_k=3)}


def risk_agent(state: AdvisorState) -> AdvisorState:
    profile = state["profile"]
    project = state["project_type"]
    reasons: list[str] = []
    high_risk = False

    if profile.get("pregnant") and project in {"注射", "激光", "手术"}:
        reasons.append("孕期信息需要医师确认")
        high_risk = True
    if profile.get("anticoagulants") and project in {"注射", "手术"}:
        reasons.append("正在使用抗凝药物")
        high_risk = True
    if profile.get("photosensitive_medication") and project == "激光":
        reasons.append("正在使用光敏性药物")
        high_risk = True
    if profile.get("allergy_history"):
        reasons.append("存在过敏史，需要进一步确认")

    level = "high" if high_risk else "medium" if reasons else "low"
    return {"risk_level": level, "risk_reasons": reasons}


def route_after_risk(state: AdvisorState) -> str:
    return "doctor_review" if state["risk_level"] == "high" else "answer_agent"


def doctor_review(state: AdvisorState) -> AdvisorState:
    decision = interrupt(
        {
            "type": "doctor_review",
            "question": state["question"],
            "project_type": state["project_type"],
            "risk_level": state["risk_level"],
            "risk_reasons": state["risk_reasons"],
            "allowed_actions": ["approve", "reject", "edit"],
        }
    )
    return {"review_decision": decision}


def answer_agent(state: AdvisorState) -> AdvisorState:
    decision = state.get("review_decision")
    if decision and decision.get("action") == "reject":
        answer = "医师审核未通过。智能顾问不继续提供项目建议，请预约线下面诊。"
    elif decision and decision.get("action") == "edit" and decision.get("note"):
        answer = f"医师审核意见：{decision['note']}"
    else:
        evidence = " ".join(item["content"] for item in state.get("sources", [])[:2])
        prefix = "医师审核已通过。" if decision else ""
        answer = (
            f"{prefix}根据演示知识库：{evidence}"
            if evidence
            else "知识库依据不足，建议转人工咨询。"
        )
    return {"answer": answer}


builder = StateGraph(AdvisorState)
builder.add_node("knowledge_agent", knowledge_agent)
builder.add_node("risk_agent", risk_agent)
builder.add_node("doctor_review", doctor_review)
builder.add_node("answer_agent", answer_agent)
builder.add_edge(START, "knowledge_agent")
builder.add_edge("knowledge_agent", "risk_agent")
builder.add_conditional_edges(
    "risk_agent",
    route_after_risk,
    {"doctor_review": "doctor_review", "answer_agent": "answer_agent"},
)
builder.add_edge("doctor_review", "answer_agent")
builder.add_edge("answer_agent", END)
advisor_graph = builder.compile(checkpointer=InMemorySaver())


class AdvisorService:
    def start(self, question: str, project_type: str, profile: dict[str, Any]) -> dict:
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        result = advisor_graph.invoke(
            {"question": question, "project_type": project_type, "profile": profile},
            config,
        )
        state = advisor_graph.get_state(config).values
        if "__interrupt__" in result:
            return {
                "thread_id": thread_id,
                "status": "pending_review",
                "risk_level": state["risk_level"],
                "risk_reasons": state["risk_reasons"],
                "sources": state.get("sources", []),
            }
        return {"thread_id": thread_id, "status": "completed", **state}

    def review(self, thread_id: str, action: str, note: str = "") -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        if not advisor_graph.get_state(config).values:
            raise KeyError("consultation not found")
        advisor_graph.invoke(Command(resume={"action": action, "note": note}), config)
        return {
            "thread_id": thread_id,
            "status": "completed",
            **advisor_graph.get_state(config).values,
        }


advisor_service = AdvisorService()

