"""Pydantic request/response schemas for the detection API."""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    code: str = Field(..., description="Source code to analyse")
    language: str = Field(
        ...,
        description="Programming language: c, cpp, csharp, java, javascript, python",
    )
    problem_id: str | None = Field(None, description="Optional problem identifier for context")

    model_config = {"json_schema_extra": {
        "examples": [{
            "code": '#include <iostream>\\nint main() { std::cout << 42; }',
            "language": "cpp",
            "problem_id": "p00001",
        }]
    }}


class ComponentScores(BaseModel):
    statistical: float | None = Field(None, description="XGBoost baseline score")
    codebert: float | None = Field(None, description="CodeBERT classifier score")
    llm_judge: float | None = Field(None, description="LLM-as-judge score (if triggered)")


class AnalyzeResponse(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Ensemble risk score")
    decision: str = Field(..., description="accept, review, or hold")
    component_scores: ComponentScores
    signals: list[str] = Field(default_factory=list, description="Human-readable detection signals")


class BatchRequest(BaseModel):
    submissions: list[AnalyzeRequest]


class BatchResponse(BaseModel):
    results: list[AnalyzeResponse]


class HealthResponse(BaseModel):
    status: str = "ok"
    models_loaded: dict[str, bool] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    code: str
    language: str
    true_label: int = Field(..., ge=0, le=1, description="0=human, 1=AI")
    problem_id: str | None = None
    reviewer_id: str | None = None


class FeedbackResponse(BaseModel):
    status: str = "recorded"
    feedback_id: str
