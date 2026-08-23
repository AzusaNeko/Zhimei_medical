from app.knowledge import retriever
from app.workflow import AdvisorService


def test_retrieval_returns_relevant_evidence():
    results = retriever.search("透明质酸术后如何护理？")
    assert results
    assert results[0]["doc_id"] == "kb-002"


def test_low_risk_flow_completes_directly():
    result = AdvisorService().start("透明质酸术后如何护理？", "注射", {})
    assert result["status"] == "completed"
    assert result["risk_level"] == "low"
    assert result["answer"]


def test_high_risk_flow_interrupts_and_resumes():
    service = AdvisorService()
    pending = service.start("孕期可以做注射项目吗？", "注射", {"pregnant": True})
    assert pending["status"] == "pending_review"
    assert pending["risk_level"] == "high"
    completed = service.review(
        pending["thread_id"], "reject", "建议产后再由医师面诊评估"
    )
    assert completed["status"] == "completed"
    assert "审核未通过" in completed["answer"]


def test_medium_risk_does_not_block_demo_flow():
    result = AdvisorService().start(
        "想了解皮肤护理项目", "皮肤护理", {"allergy_history": True}
    )
    assert result["status"] == "completed"
    assert result["risk_level"] == "medium"

