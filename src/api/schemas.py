"""Pydantic request/response schemas for the detection API."""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    code: str = Field(..., description="Source code to analyse")
    language: str = Field(
        ...,
        description="Programming language: c, cpp, csharp, java, javascript, python",
    )
    problem_id: str | None = Field(None, description="Optional problem identifier for context")
    detection_mode: str = Field(
        "ensemble",
        description=(
            "How to combine detectors: "
            "`ensemble` (default, optional LLM when gate exceeded), "
            "`stylometric`, `randomforest`, `logisticregression`, `codebert`, `graphcodebert`, `unixcoder`, `fusion` (stat + CodeBERT, no LLM), "
            "`llm` (LLM-as-judge only; requires API keys)"
        ),
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "code": '#include <iostream>\\nint main() { std::cout << 42; }',
            "language": "cpp",
            "problem_id": "p00001",
        }]
    }}


class ComponentScores(BaseModel):
    statistical: float | None = Field(None, description="XGBoost baseline score")
    random_forest: float | None = Field(None, description="Random Forest classifier score")
    logistic_regression: float | None = Field(None, description="Logistic Regression classifier score")
    codebert: float | None = Field(None, description="CodeBERT classifier score")
    graphcodebert: float | None = Field(None, description="GraphCodeBERT classifier score")
    unixcoder: float | None = Field(None, description="UniXcoder classifier score")
    llm_judge: float | None = Field(None, description="LLM-as-judge score (if triggered)")


class AnalyzeResponse(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Ensemble risk score")
    decision: str = Field(..., description="accept, review, or hold")
    detection_mode: str = Field(
        "ensemble",
        description="Detector pipeline used for this analysis",
    )
    component_scores: ComponentScores
    signals: list[str] = Field(default_factory=list, description="Human-readable detection signals")


class BatchRequest(BaseModel):
    submissions: list[AnalyzeRequest]


class BatchResponse(BaseModel):
    results: list[AnalyzeResponse]


class HealthResponse(BaseModel):
    status: str = "ok"
    models_loaded: dict[str, bool] = Field(default_factory=dict)
    threshold_auto_accept: float = Field(
        0.4, description="Risk scores below this are accepted as human-written"
    )
    threshold_flag_review: float = Field(
        0.7, description="Risk scores below this (but above accept) are flagged for review"
    )


class FeedbackRequest(BaseModel):
    code: str
    language: str
    true_label: int = Field(..., ge=0, le=1, description="0=human, 1=AI")
    problem_id: str | None = None
    reviewer_id: str | None = None


class FeedbackResponse(BaseModel):
    status: str = "recorded"
    feedback_id: str


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_]+$")
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class GoogleLoginRequest(BaseModel):
    id_token: str = Field(..., min_length=20)


class UserPublic(BaseModel):
    id: str
    username: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic


class HistoryEntryResponse(BaseModel):
    id: str
    language: str
    problem_id: str | None = None
    detection_mode: str
    risk_score: float
    decision: str
    code_preview: str
    created_at: str


class HistoryDetailResponse(HistoryEntryResponse):
    code: str
    component_scores: dict
    signals: list[str]


class HistoryListResponse(BaseModel):
    entries: list[HistoryEntryResponse]


class GoogleConfigResponse(BaseModel):
    enabled: bool
    client_id: str | None = None


class SupabaseConfigResponse(BaseModel):
    enabled: bool
    url: str | None = None
    anon_key: str | None = None
