"""
Token-level perplexity scoring using a pre-trained code language model.

AI-generated code tends to have lower perplexity (it is "too fluent" under a
code LM trained on human code). This asymmetry is a strong distinguishing signal.
"""

import math

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PERPLEXITY_MODEL_NAME

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    logger.info(f"Loading perplexity model: {PERPLEXITY_MODEL_NAME}")
    _tokenizer = AutoTokenizer.from_pretrained(PERPLEXITY_MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(PERPLEXITY_MODEL_NAME)
    _model.eval()
    if torch.cuda.is_available():
        _model = _model.cuda()


def compute_perplexity(code: str, max_length: int = 1024) -> dict:
    """
    Compute token-level perplexity of a code snippet.

    Returns:
        {
            "perplexity": float,          # overall perplexity
            "mean_log_prob": float,       # mean log probability per token
            "token_count": int,           # number of tokens evaluated
            "max_surprise": float,        # highest -log_prob for any token
        }
    """
    _load_model()

    encodings = _tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encodings.input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    with torch.no_grad():
        outputs = _model(input_ids, labels=input_ids)
        loss = outputs.loss.item()

    perplexity = math.exp(loss)

    log_probs = []
    if input_ids.shape[1] > 1:
        with torch.no_grad():
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = log_softmax.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            log_probs = token_log_probs.squeeze(0).cpu().tolist()

    return {
        "perplexity": perplexity,
        "mean_log_prob": -loss,
        "token_count": input_ids.shape[1],
        "max_surprise": max(-lp for lp in log_probs) if log_probs else 0.0,
    }


def extract_features(code: str) -> dict:
    """Public interface: extract perplexity features from a code string."""
    try:
        return compute_perplexity(code)
    except Exception as e:
        logger.warning(f"Perplexity extraction failed: {e}")
        return {
            "perplexity": float("nan"),
            "mean_log_prob": float("nan"),
            "token_count": 0,
            "max_surprise": float("nan"),
        }
