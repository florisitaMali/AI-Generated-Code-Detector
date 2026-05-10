"""API route handlers for the AI code detection service."""

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchRequest,
    BatchResponse,
    ComponentScores,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
)
from src.ensemble.scorer import EnsembleScorer, make_decision
from config.settings import MODELS_DIR, RAW_DIR, THRESHOLD_AUTO_ACCEPT, THRESHOLD_FLAG_REVIEW, LLM_GATE_THRESHOLD

router = APIRouter()

SUPPORTED_LANGUAGES = frozenset({"cpp", "python", "java", "c", "csharp", "javascript"})

_scorer: EnsembleScorer | None = None
_models_loaded = {
    "statistical_baseline": False,
    "codebert": False,
    "ensemble": False,
}


def _get_scorer() -> EnsembleScorer:
    global _scorer
    if _scorer is None:
        try:
            _scorer = EnsembleScorer.load()
            _models_loaded["ensemble"] = True
        except FileNotFoundError:
            _scorer = EnsembleScorer()
            logger.warning("No trained ensemble found, using default weights")
    return _scorer


def _get_statistical_score(code: str, language: str) -> float | None:
    try:
        from src.models.statistical_baseline import predict
        score = predict(code, language)
        _models_loaded["statistical_baseline"] = True
        return score
    except Exception as e:
        logger.warning(f"Statistical baseline unavailable: {e}")
    # Last-resort: raw heuristic (no model file needed)
    try:
        from src.models.statistical_baseline import predict_heuristic
        logger.info("Using heuristic scorer as fallback")
        return predict_heuristic(code, language)
    except Exception as e2:
        logger.warning(f"Heuristic scorer also failed: {e2}")
        return None


def _get_codebert_score(code: str) -> float | None:
    try:
        from src.models.codebert_classifier import predict
        score = predict(code)
        _models_loaded["codebert"] = True
        return score
    except Exception as e:
        logger.warning(f"CodeBERT model unavailable: {e}")
        return None


# _get_behavioral_score is kept here intentionally (unused) so the git diff stays small.
def _get_behavioral_score(metadata: dict | None) -> float | None:
    if not metadata:
        return None
    try:
        from src.behavioral.timing import extract_features as timing_features
        from src.behavioral.keystroke import extract_features as keystroke_features

        timing = timing_features(metadata)
        keystroke = keystroke_features(metadata)

        scores = [
            timing.get("timing_score", 0),
            timing.get("velocity_score", 0),
            keystroke.get("keystroke_score", 0),
        ]
        non_zero = [s for s in scores if s > 0]
        return sum(non_zero) / len(non_zero) if non_zero else 0.0
    except Exception as e:
        logger.warning(f"Behavioral analysis failed: {e}")
        return None


def _load_human_examples(problem_id: str, language: str, max_examples: int = 3) -> list[str]:
    """Load human reference solutions from data/raw/ for the given problem."""
    lang_ext = {"cpp": ".cpp", "python": ".py", "java": ".java", "c": ".c",
                "csharp": ".cs", "javascript": ".js"}
    ext = lang_ext.get(language, "")
    problem_dir = RAW_DIR / problem_id
    if not problem_dir.exists():
        return []
    examples = []
    for f in sorted(problem_dir.iterdir()):
        if f.suffix == ext:
            try:
                examples.append(f.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                pass
            if len(examples) >= max_examples:
                break
    return examples


async def _get_llm_judge_score(
    code: str,
    problem_id: str | None,
    language: str,
    stat_score: float | None,
    codebert_score: float | None = None,
) -> float | None:
    """Call LLM-as-judge when ANY raw component score exceeds the gate threshold."""
    scores = [s for s in (stat_score, codebert_score) if s is not None]
    if not scores or max(scores) < LLM_GATE_THRESHOLD:
        return None
    try:
        from src.models.llm_judge import judge
        pid = problem_id or "unknown"
        human_examples = _load_human_examples(pid, language) if pid != "unknown" else []
        result = await judge(code, problem_id=pid, human_examples=human_examples)
        score = result.get("score")
        if score is not None:
            logger.info(f"LLM-judge score={score:.2f} (examples={len(human_examples)}), signals={result.get('signals')}")
        return float(score) if score is not None else None
    except Exception as e:
        logger.warning(f"LLM-judge unavailable: {e}")
        return None


async def _analyze_single(req: AnalyzeRequest) -> AnalyzeResponse:
    scorer = _get_scorer()

    stat_score = _get_statistical_score(req.code, req.language)
    codebert_score = _get_codebert_score(req.code)
    llm_score = await _get_llm_judge_score(req.code, req.problem_id, req.language, stat_score, codebert_score)

    component_scores: dict[str, float] = {}
    if stat_score is not None:
        component_scores["statistical"] = stat_score
    if codebert_score is not None:
        component_scores["codebert"] = codebert_score
    if llm_score is not None:
        component_scores["llm_judge"] = llm_score

    if not component_scores:
        return AnalyzeResponse(
            risk_score=0.5,
            decision="review",
            component_scores=ComponentScores(),
            signals=["All detectors unavailable; score defaulted to 0.5 (review)"],
        )

    result = scorer.score(component_scores)

    signals = []
    if stat_score is not None and stat_score > 0.6:
        signals.append(f"Stylometric features suggest AI origin (score={stat_score:.2f})")
    if codebert_score is not None and codebert_score > 0.6:
        signals.append(f"CodeBERT classifier flags as AI (score={codebert_score:.2f})")
    if llm_score is not None and llm_score > 0.6:
        signals.append(f"LLM-judge classifies as AI-generated (score={llm_score:.2f})")

    return AnalyzeResponse(
        risk_score=result["risk_score"],
        decision=result["decision"],
        component_scores=ComponentScores(
            statistical=stat_score,
            codebert=codebert_score,
            llm_judge=llm_score,
        ),
        signals=signals,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Analyse a single code submission for AI generation."""
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Empty code submission")
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{request.language}'. "
                   f"Must be one of: {', '.join(sorted(SUPPORTED_LANGUAGES))}",
        )

    return await _analyze_single(request)


@router.post("/batch", response_model=BatchResponse)
async def batch_analyze(request: BatchRequest):
    """Analyse multiple submissions in batch."""
    if not request.submissions:
        raise HTTPException(status_code=400, detail="No submissions provided")
    if len(request.submissions) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 submissions per batch")
    for sub in request.submissions:
        if sub.language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{sub.language}' in batch. "
                       f"Must be one of: {', '.join(sorted(SUPPORTED_LANGUAGES))}",
            )

    results = await asyncio.gather(*[_analyze_single(sub) for sub in request.submissions])
    return BatchResponse(results=list(results))


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", models_loaded=_models_loaded)


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    """Record admin feedback on a submission for future retraining."""
    feedback_id = str(uuid.uuid4())[:8]
    logger.info(
        f"Feedback {feedback_id}: label={request.true_label}, "
        f"problem={request.problem_id}, reviewer={request.reviewer_id}"
    )
    return FeedbackResponse(status="recorded", feedback_id=feedback_id)
