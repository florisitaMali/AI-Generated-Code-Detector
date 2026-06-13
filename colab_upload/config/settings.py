from pathlib import Path
from dotenv import load_dotenv
import os

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
DATA_DIR = ROOT_DIR / os.getenv("DATA_DIR", "data")
RAW_DIR = DATA_DIR / "raw"
GENERATED_DIR = DATA_DIR / "generated"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT_DIR / "models"

CODENET_URL = os.getenv(
    "CODENET_URL",
    "https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz",
)
CODENET_METADATA_URL = os.getenv(
    "CODENET_METADATA_URL",
    "https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_metadata.tar.gz",
)

LANGUAGES = {"C++": "cpp", "Python": "python", "Java": "java"}
LANGUAGE_EXTENSIONS = {"cpp": [".cpp", ".cc", ".cxx"], "python": [".py"], "java": [".java"]}

NUM_PROBLEMS = int(os.getenv("NUM_PROBLEMS", "500"))
MIN_ACCEPTED_SOLUTIONS = int(os.getenv("MIN_ACCEPTED_SOLUTIONS", "10"))
AI_SOLUTIONS_PER_PROBLEM = int(os.getenv("AI_SOLUTIONS_PER_PROBLEM", "5"))

AI_GENERATION_MODELS = [
    m.strip() for m in os.getenv(
        "AI_GENERATION_MODELS",
        "gemini-2.0-flash",
    ).split(",")
    if m.strip()
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Seconds between OpenAI Chat Completions requests (per client; lowers RPM/TPM burst).
OPENAI_MIN_INTERVAL = float(os.getenv("OPENAI_MIN_INTERVAL", "4.0"))
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# CodeBERT training (override on e.g. Colab: CODEBERT_BATCH_SIZE=4 CODEBERT_EPOCHS=2)
CODEBERT_MODEL_NAME = "microsoft/codebert-base"
CODEBERT_MAX_LENGTH = 512
# Applied as sigmoid(logit / T). T>1 softens probabilities (better calibration plots).
# Optional file models/codebert_final/inference_calibration.json overrides this when present.
CODEBERT_INFERENCE_TEMPERATURE = float(os.getenv("CODEBERT_INFERENCE_TEMPERATURE", "1.0"))
CODEBERT_LR = 2e-5
CODEBERT_BATCH_SIZE = int(os.getenv("CODEBERT_BATCH_SIZE", "16"))
CODEBERT_EPOCHS = int(os.getenv("CODEBERT_EPOCHS", "5"))
CODEBERT_WARMUP_RATIO = float(os.getenv("CODEBERT_WARMUP_RATIO", "0.1"))
# EarlyStoppingCallback counts validation *runs*; with step-based eval below, default is higher than epoch-only training.
CODEBERT_PATIENCE = int(os.getenv("CODEBERT_PATIENCE", "6"))
# Save/eval every N optimizer steps (frequent = safer resume on Colab). Capped to ≤ 1× per epoch in code.
CODEBERT_SAVE_STEPS = int(os.getenv("CODEBERT_SAVE_STEPS", "250"))
# Rolling checkpoints kept under models/codebert/ (latest used for resume via get_last_checkpoint)
CODEBERT_SAVE_TOTAL_LIMIT = int(os.getenv("CODEBERT_SAVE_TOTAL_LIMIT", "8"))

# Perplexity model
PERPLEXITY_MODEL_NAME = "microsoft/CodeGPT-small-py"

# Ensemble decision thresholds (product policy; override via .env)
# Calibrated test-split values were ~0.17 / ~0.30 — use those only if you want tighter auto-accept.
THRESHOLD_AUTO_ACCEPT = float(os.getenv("THRESHOLD_AUTO_ACCEPT", "0.4"))
THRESHOLD_FLAG_REVIEW = float(os.getenv("THRESHOLD_FLAG_REVIEW", "0.7"))

# Minimum raw component score to trigger LLM-as-judge (separate from decision thresholds)
LLM_GATE_THRESHOLD = float(os.getenv("LLM_GATE_THRESHOLD", "0.45"))

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# TraceCoder accounts (change JWT_SECRET in production)
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "168"))
TRACECODER_DB_PATH = Path(os.getenv("TRACECODER_DB_PATH", str(DATA_DIR / "tracecoder.db")))
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()

# Supabase (recommended auth + history persistence)
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
# Project Settings → API → JWT Settings → JWT Secret (for HS256 user access tokens)
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "").strip()
SUPABASE_JWT_AUDIENCE = os.getenv("SUPABASE_JWT_AUDIENCE", "authenticated").strip()
SUPABASE_HISTORY_TABLE = os.getenv("SUPABASE_HISTORY_TABLE", "scan_history").strip()
SUPABASE_USERS_TABLE = os.getenv("SUPABASE_USERS_TABLE", "app_users").strip()

# Dataset splits (legacy parquet builder)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# AI-CodeNet / merge (schema-identical to IBM CodeNet tree + CSV)
AI_CODENET_ROOT = Path(os.getenv("AI_CODENET_ROOT", str(DATA_DIR / "ai_codenet")))
MERGED_CODENET_ROOT = Path(os.getenv("MERGED_CODENET_ROOT", str(DATA_DIR / "merged_codenet")))
MAX_CODENET_PROBLEMS_RAW = os.getenv("MAX_CODENET_PROBLEMS", "").strip()
MAX_CODENET_PROBLEMS = (
    int(MAX_CODENET_PROBLEMS_RAW) if MAX_CODENET_PROBLEMS_RAW.isdigit() else None
)
AI_CODENET_RUN_TESTS = os.getenv("AI_CODENET_RUN_TESTS", "true").lower() in (
    "1",
    "true",
    "yes",
)
AI_CODENET_RATE_SLEEP = float(os.getenv("AI_CODENET_RATE_SLEEP", "1.2"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
MERGE_TRAIN_RATIO = float(os.getenv("MERGE_TRAIN_RATIO", "0.80"))
MERGE_VAL_RATIO = float(os.getenv("MERGE_VAL_RATIO", "0.10"))
MERGE_TEST_RATIO = float(os.getenv("MERGE_TEST_RATIO", "0.10"))
