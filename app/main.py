from fastapi import FastAPI, HTTPException

from .models import ConsultationRequest, ReviewRequest
from .workflow import advisor_service


app = FastAPI(
    title="智美医美智能顾问平台",
    version="0.1.0",
    description="LangGraph 知识科普、风险筛查与高风险医师审核最小实现。",
)


@app.get("/")
def root() -> dict:
    return {
        "name": "zhimei-medical-advisor",
        "docs": "/docs",
        "warning": "本系统不提供医疗诊断或治疗建议。",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/consultations")
def create_consultation(payload: ConsultationRequest) -> dict:
    return advisor_service.start(
        payload.question,
        payload.project_type,
        payload.profile.model_dump(),
    )


@app.post("/consultations/{thread_id}/review")
def review_consultation(thread_id: str, payload: ReviewRequest) -> dict:
    try:
        return advisor_service.review(thread_id, payload.action, payload.note)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
