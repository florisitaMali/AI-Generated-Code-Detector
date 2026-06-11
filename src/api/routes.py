"""API route handlers for the AI code detection service."""

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
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
from src.auth.deps import get_optional_user
from src.auth.store import User, add_history
from src.debug_log import agent_log
from config.settings import MODELS_DIR, RAW_DIR, THRESHOLD_AUTO_ACCEPT, THRESHOLD_FLAG_REVIEW, LLM_GATE_THRESHOLD

router = APIRouter()

SUPPORTED_LANGUAGES = frozenset({"cpp", "python", "java", "c", "csharp", "javascript"})
DETECTION_MODES = frozenset({"ensemble", "stylometric", "randomforest", "logisticregression", "codebert", "graphcodebert", "unixcoder", "fusion", "llm"})

_scorer: EnsembleScorer | None = None
_models_loaded = {
    "statistical_baseline": False,
    "rf_baseline": False,
    "lr_baseline": False,
    "codebert": False,
    "graphcodebert": False,
    "unixcoder": False,
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


def _get_rf_score(code: str, language: str) -> float | None:
    try:
        from src.models.rf_baseline import predict
        score = predict(code, language)
        _models_loaded["rf_baseline"] = True
        return score
    except Exception as e:
        logger.warning(f"Random Forest model unavailable: {e}")
        return None


def _get_lr_score(code: str, language: str) -> float | None:
    try:
        from src.models.lr_baseline import predict
        score = predict(code, language)
        _models_loaded["lr_baseline"] = True
        return score
    except Exception as e:
        logger.warning(f"Logistic Regression model unavailable: {e}")
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


def _get_graphcodebert_score(code: str) -> float | None:
    try:
        from src.models.graphcodebert_classifier import predict
        score = predict(code)
        _models_loaded["graphcodebert"] = True
        return score
    except Exception as e:
        logger.warning(f"GraphCodeBERT model unavailable: {e}")
        return None


def _get_unixcoder_score(code: str) -> float | None:
    try:
        from src.models.unixcoder_classifier import predict
        score = predict(code)
        _models_loaded["unixcoder"] = True
        return score
    except Exception as e:
        logger.warning(f"UniXcoder model unavailable: {e}")
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


def _parse_detection_mode(detection_mode: str | None) -> str:
    mode = (detection_mode or "ensemble").strip().lower()
    if mode not in DETECTION_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported detection_mode '{detection_mode}'. "
            f"Use one of: {', '.join(sorted(DETECTION_MODES))}",
        )
    return mode


async def _get_llm_judge_score(
    code: str,
    problem_id: str | None,
    language: str,
    stat_score: float | None,
    codebert_score: float | None = None,
    *,
    force: bool = False,
) -> float | None:
    """Call LLM-as-judge when gated, or always when ``force`` (LLM-only mode)."""
    if not force:
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
    mode = _parse_detection_mode(req.detection_mode)

    scorer = _get_scorer()

    stat_score: float | None = None
    rf_score: float | None = None
    lr_score: float | None = None
    codebert_score: float | None = None
    graphcodebert_score: float | None = None
    unixcoder_score: float | None = None
    llm_score: float | None = None

    if mode == "stylometric":
        stat_score = _get_statistical_score(req.code, req.language)
    elif mode == "randomforest":
        rf_score = _get_rf_score(req.code, req.language)
    elif mode == "logisticregression":
        lr_score = _get_lr_score(req.code, req.language)
    elif mode == "codebert":
        codebert_score = _get_codebert_score(req.code)
    elif mode == "graphcodebert":
        graphcodebert_score = _get_graphcodebert_score(req.code)
    elif mode == "unixcoder":
        unixcoder_score = _get_unixcoder_score(req.code)
    elif mode == "llm":
        llm_score = await _get_llm_judge_score(
            req.code, req.problem_id, req.language, None, None, force=True
        )
    else:
        stat_score = _get_statistical_score(req.code, req.language)
        codebert_score = _get_codebert_score(req.code)
        if mode == "ensemble":
            llm_score = await _get_llm_judge_score(
                req.code, req.problem_id, req.language, stat_score, codebert_score
            )
        # mode == "fusion": no LLM

    component_scores: dict[str, float] = {}
    if stat_score is not None:
        component_scores["statistical"] = stat_score
    if rf_score is not None:
        component_scores["random_forest"] = rf_score
    if lr_score is not None:
        component_scores["logistic_regression"] = lr_score
    if codebert_score is not None:
        component_scores["codebert"] = codebert_score
    if graphcodebert_score is not None:
        component_scores["graphcodebert"] = graphcodebert_score
    if unixcoder_score is not None:
        component_scores["unixcoder"] = unixcoder_score
    if llm_score is not None:
        component_scores["llm_judge"] = llm_score

    if not component_scores:
        hint = ""
        if mode == "llm":
            hint = " Configure OPENAI_API_KEY / ANTHROPIC_API_KEY (and optional GOOGLE_API_KEY) for LLM mode."
        return AnalyzeResponse(
            risk_score=0.5,
            decision="review",
            detection_mode=mode,
            component_scores=ComponentScores(),
            signals=[f"No score produced in '{mode}' mode (detector unavailable).{hint}"],
        )

    if mode == "stylometric" and stat_score is not None:
        risk = float(stat_score)
        decision = make_decision(risk)
    elif mode == "randomforest" and rf_score is not None:
        risk = float(rf_score)
        decision = make_decision(risk)
    elif mode == "logisticregression" and lr_score is not None:
        risk = float(lr_score)
        decision = make_decision(risk)
    elif mode == "codebert" and codebert_score is not None:
        risk = float(codebert_score)
        decision = make_decision(risk)
    elif mode == "graphcodebert" and graphcodebert_score is not None:
        risk = float(graphcodebert_score)
        decision = make_decision(risk)
    elif mode == "unixcoder" and unixcoder_score is not None:
        risk = float(unixcoder_score)
        decision = make_decision(risk)
    elif mode == "llm" and llm_score is not None:
        risk = float(llm_score)
        decision = make_decision(risk)
    else:
        result = scorer.score(component_scores)
        risk = float(result["risk_score"])
        decision = str(result["decision"])

    signals: list[str] = []
    if mode == "fusion":
        signals.append("Fusion: stylometric + CodeBERT (LLM audit disabled for this request).")
    elif mode == "stylometric":
        signals.append("Single detector: stylometric features only.")
    elif mode == "randomforest":
        signals.append("Single detector: Random Forest on AST, identifier, and comment features.")
    elif mode == "logisticregression":
        signals.append("Single detector: Logistic Regression on AST, identifier, and comment features.")
    elif mode == "codebert":
        signals.append("Single detector: CodeBERT neural encoder.")
    elif mode == "graphcodebert":
        signals.append("Single detector: GraphCodeBERT (data-flow enhanced encoder).")
    elif mode == "unixcoder":
        signals.append("Single detector: UniXcoder (unified cross-modal encoder).")
    elif mode == "llm":
        signals.append(
            "Single detector: LLM-as-judge (set Problem ID when possible for human-reference context)."
        )

    if stat_score is not None and stat_score > 0.6:
        signals.append(f"Stylometric features suggest AI origin (score={stat_score:.2f})")
    if rf_score is not None and rf_score > 0.6:
        signals.append(f"Random Forest flags as AI-generated (score={rf_score:.2f})")
    if lr_score is not None and lr_score > 0.6:
        signals.append(f"Logistic Regression flags as AI-generated (score={lr_score:.2f})")
    if codebert_score is not None and codebert_score > 0.6:
        signals.append(f"CodeBERT classifier flags as AI (score={codebert_score:.2f})")
    if graphcodebert_score is not None and graphcodebert_score > 0.6:
        signals.append(f"GraphCodeBERT flags as AI-generated (score={graphcodebert_score:.2f})")
    if unixcoder_score is not None and unixcoder_score > 0.6:
        signals.append(f"UniXcoder flags as AI-generated (score={unixcoder_score:.2f})")
    if llm_score is not None and llm_score > 0.6:
        signals.append(f"LLM-judge classifies as AI-generated (score={llm_score:.2f})")

    return AnalyzeResponse(
        risk_score=risk,
        decision=decision,
        detection_mode=mode,
        component_scores=ComponentScores(
            statistical=stat_score,
            random_forest=rf_score,
            logistic_regression=lr_score,
            codebert=codebert_score,
            graphcodebert=graphcodebert_score,
            unixcoder=unixcoder_score,
            llm_judge=llm_score,
        ),
        signals=signals,
    )


def _save_scan_history(
    user_id: str,
    *,
    code: str,
    language: str,
    problem_id: str | None,
    detection_mode: str,
    risk_score: float,
    decision: str,
    component_scores: dict,
    signals: list[str],
) -> None:
    try:
        add_history(
            user_id,
            code=code,
            language=language,
            problem_id=problem_id,
            detection_mode=detection_mode,
            risk_score=risk_score,
            decision=decision,
            component_scores=component_scores,
            signals=signals,
        )
        agent_log("routes.py:_save_scan_history", "history saved", {"user_id_prefix": user_id[:8]}, "H5")
    except Exception as exc:
        agent_log(
            "routes.py:_save_scan_history",
            "history save failed",
            {"error_type": type(exc).__name__, "error": str(exc)[:200]},
            "H5",
        )
        logger.warning(f"Could not save scan history: {exc}")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    user: User | None = Depends(get_optional_user),
):
    """Analyse a single code submission for AI generation."""
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Empty code submission")
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{request.language}'. "
                   f"Must be one of: {', '.join(sorted(SUPPORTED_LANGUAGES))}",
        )

    logger.info("Analyze request: language=%s detection_mode=%s", request.language, request.detection_mode)
    result = await _analyze_single(request)
    agent_log(
        "routes.py:analyze",
        "analyze complete",
        {"user_present": user is not None, "user_id_prefix": (user.id[:8] if user else None)},
        "H4",
    )
    if user is not None:
        comp = result.component_scores.model_dump() if result.component_scores else {}
        background_tasks.add_task(
            _save_scan_history,
            user.id,
            code=request.code,
            language=request.language,
            problem_id=request.problem_id,
            detection_mode=result.detection_mode,
            risk_score=result.risk_score,
            decision=result.decision,
            component_scores=comp,
            signals=result.signals,
        )
    return result


@router.post("/batch", response_model=BatchResponse)
async def batch_analyze(request: BatchRequest):
    """Analyse multiple submissions in batch."""
    if not request.submissions:
        raise HTTPException(status_code=400, detail="No submissions provided")
    if len(request.submissions) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 submissions per batch")
    for sub in request.submissions:
        _parse_detection_mode(sub.detection_mode)
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
    return HealthResponse(
        status="ok",
        models_loaded=_models_loaded,
        threshold_auto_accept=THRESHOLD_AUTO_ACCEPT,
        threshold_flag_review=THRESHOLD_FLAG_REVIEW,
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    """Record admin feedback on a submission for future retraining."""
    feedback_id = str(uuid.uuid4())[:8]
    logger.info(
        f"Feedback {feedback_id}: label={request.true_label}, "
        f"problem={request.problem_id}, reviewer={request.reviewer_id}"
    )
    return FeedbackResponse(status="recorded", feedback_id=feedback_id)
