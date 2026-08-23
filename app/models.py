from typing import Literal

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    age: int | None = Field(default=None, ge=0, le=120)
    pregnant: bool = False
    anticoagulants: bool = False
    allergy_history: bool = False
    photosensitive_medication: bool = False


class ConsultationRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    project_type: Literal["注射", "激光", "手术", "皮肤护理", "其他"] = "其他"
    profile: UserProfile = Field(default_factory=UserProfile)


class ReviewRequest(BaseModel):
    action: Literal["approve", "reject", "edit"]
    note: str = Field(default="", max_length=500)

