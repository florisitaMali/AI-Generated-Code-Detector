"""
Generate AI solutions for selected CodeNet problems.

For each problem, reads the problem statement from CodeNet metadata and asks
multiple AI models (GPT-4, Claude, CodeLlama, StarCoder) to generate solutions
in C++, Python, and Java. Uses async HTTP calls with rate-limiting and retries.
"""

import asyncio
import re
from pathlib import Path

import httpx
from loguru import logger
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    AI_GENERATION_MODELS,
    AI_SOLUTIONS_PER_PROBLEM,
    ANTHROPIC_API_KEY,
    DATA_DIR,
    GEMINI_MODEL,
    GENERATED_DIR,
    GOOGLE_API_KEY,
    HF_TOKEN,
    OPENAI_API_KEY,
    OPENAI_MIN_INTERVAL,
)

LANG_DISPLAY = {"cpp": "C++", "python": "Python", "java": "Java"}

# Rate-limiting: semaphore = 1 concurrent call per provider. OpenAI also uses
# OPENAI_MIN_INTERVAL from .env (default 4s). If your org shows low RPM (e.g. 3/min),
# set OPENAI_MIN_INTERVAL≈ceil(60/RPM)+1 before each request (including retries).
SEMAPHORES = {
    "openai": asyncio.Semaphore(1),
    "anthropic": asyncio.Semaphore(1),
    "huggingface": asyncio.Semaphore(3),
    "google": asyncio.Semaphore(1),
}

RETRY_ATTEMPTS = 4
RETRY_BACKOFF = 2.0
GEMINI_MIN_INTERVAL = 5.0


def _openai_error_meta(response: httpx.Response) -> tuple[str | None, float]:
    """Parse JSON error.code and optional 'try again in Ns' from the response body."""
    text = response.text or ""
    code = None
    try:
        err = response.json().get("error") or {}
        code = err.get("code") or err.get("type")
    except Exception:
        pass
    m = re.search(r"try again in (\d+)\s*s", text, re.I)
    try_again = float(m.group(1)) if m else 0.0
    return code, try_again


def load_problem_statements(metadata_dir: Path) -> dict[str, str]:
    """
    Load problem descriptions. CodeNet stores HTML descriptions per problem.
    Returns {problem_id: description_text}.
    """
    statements = {}

    desc_dir = None
    for candidate in metadata_dir.rglob("problem_descriptions"):
        if candidate.is_dir():
            desc_dir = candidate
            break

    if desc_dir is None:
        for candidate in metadata_dir.rglob("description"):
            if candidate.is_dir():
                desc_dir = candidate
                break

    if desc_dir and desc_dir.exists():
        for html_file in desc_dir.iterdir():
            pid = html_file.stem
            try:
                text = html_file.read_text(encoding="utf-8", errors="replace")
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) > 50:
                    statements[pid] = text[:4000]
            except Exception as e:
                logger.warning(f"Error reading {html_file}: {e}")

    logger.info(f"Loaded {len(statements)} problem statements")
    return statements


def build_prompt(problem_statement: str, language: str) -> str:
    lang_name = LANG_DISPLAY.get(language, language)
    return (
        f"Solve the following competitive programming problem in {lang_name}. "
        f"Output ONLY the complete source code, no explanation.\n\n"
        f"Problem:\n{problem_statement}"
    )


async def call_openai(
    prompt: str, model: str = "gpt-4o-mini", client: httpx.AsyncClient | None = None
) -> str | None:
    if not OPENAI_API_KEY:
        return None
    async with SEMAPHORES["openai"]:
        for attempt in range(RETRY_ATTEMPTS):
            await asyncio.sleep(OPENAI_MIN_INTERVAL)
            try:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7 + attempt * 0.1,
                        "max_tokens": 2048,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                err_code, try_again_hint = _openai_error_meta(e.response)
                if err_code == "insufficient_quota":
                    logger.error(
                        "OpenAI insufficient_quota — update billing / usage limits at "
                        "https://platform.openai.com/account/billing (retries skipped for this request)."
                    )
                    return None

                body_preview = ""
                try:
                    body_preview = (e.response.text or "")[:800].replace("\n", " ")
                except Exception:
                    pass
                if body_preview:
                    logger.warning(
                        f"OpenAI attempt {attempt + 1} HTTP {e.response.status_code}: {body_preview}"
                    )
                else:
                    logger.warning(f"OpenAI attempt {attempt + 1} failed: {e}")
                if attempt >= RETRY_ATTEMPTS - 1:
                    break
                retry_after = 0.0
                ra_hdr = e.response.headers.get("retry-after") or ""
                try:
                    retry_after = float(ra_hdr.strip())
                except ValueError:
                    pass
                if e.response.status_code == 429:
                    if try_again_hint > 0:
                        wait = max(try_again_hint + 1.0, retry_after)
                    else:
                        wait = max(30.0 * (2**attempt), retry_after)
                    logger.info(
                        f"Rate limited — waiting {wait:.0f}s "
                        f"(API hint={try_again_hint:.0f}s, retry-after={retry_after:.0f}s) …"
                    )
                else:
                    wait = RETRY_BACKOFF ** (attempt + 1)
                    logger.info(f"Waiting {wait:.0f}s before retry …")
                await asyncio.sleep(wait)
            except Exception as e:
                logger.warning(f"OpenAI attempt {attempt+1} failed: {e}")
                if attempt < RETRY_ATTEMPTS - 1:
                    is_rate_limit = "429" in str(e)
                    wait = 30.0 * (2**attempt) if is_rate_limit else RETRY_BACKOFF ** (attempt + 1)
                    logger.info(f"Waiting {wait:.0f}s before retry …")
                    await asyncio.sleep(wait)
    return None


async def call_anthropic(
    prompt: str,
    model: str = "claude-3-5-sonnet-20241022",
    client: httpx.AsyncClient | None = None,
) -> str | None:
    if not ANTHROPIC_API_KEY:
        return None
    async with SEMAPHORES["anthropic"]:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 2048,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7 + attempt * 0.1,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["content"][0]["text"]
            except Exception as e:
                logger.warning(f"Anthropic attempt {attempt+1} failed: {e}")
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_BACKOFF ** (attempt + 1))
    return None


async def call_huggingface(
    prompt: str, model: str, client: httpx.AsyncClient | None = None
) -> str | None:
    if not HF_TOKEN:
        return None
    async with SEMAPHORES["huggingface"]:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                resp = await client.post(
                    f"https://api-inference.huggingface.co/models/{model}",
                    headers={"Authorization": f"Bearer {HF_TOKEN}"},
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 2048,
                            "temperature": 0.7 + attempt * 0.1,
                            "return_full_text": False,
                        },
                    },
                    timeout=180,
                )
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list) and data:
                    return data[0].get("generated_text", "")
                return None
            except Exception as e:
                logger.warning(f"HuggingFace ({model}) attempt {attempt+1} failed: {e}")
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_BACKOFF ** (attempt + 1))
    return None


async def call_gemini(
    prompt: str,
    model: str,
    client: httpx.AsyncClient | None = None,
) -> str | None:
    if not GOOGLE_API_KEY:
        return None
    if client is None:
        return None
    async with SEMAPHORES["google"]:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model}:generateContent?key={GOOGLE_API_KEY}"
                )
                resp = await client.post(
                    url,
                    json={
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": prompt}],
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0.7 + attempt * 0.1,
                            "maxOutputTokens": 2048,
                        },
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                candidates = data.get("candidates") or []
                if not candidates:
                    return None
                parts = (candidates[0].get("content") or {}).get("parts") or []
                if not parts:
                    return None
                # Mandatory inter-call pause so we stay within the 15 RPM free tier
                await asyncio.sleep(GEMINI_MIN_INTERVAL)
                return parts[0].get("text")
            except Exception as e:
                logger.warning(f"Google Gemini attempt {attempt+1} failed: {e}")
                if attempt < RETRY_ATTEMPTS - 1:
                    # 429 rate-limit: back off much longer (30 s, 60 s, 120 s)
                    is_rate_limit = "429" in str(e)
                    wait = 30.0 * (2 ** attempt) if is_rate_limit else RETRY_BACKOFF ** (attempt + 1)
                    logger.info(f"Waiting {wait:.0f}s before retry …")
                    await asyncio.sleep(wait)
    return None


def extract_code(raw_response: str, language: str) -> str:
    """Extract code from a model response, stripping markdown fences."""
    patterns = [
        rf"```{LANG_DISPLAY.get(language, language).lower()}\n(.*?)```",
        rf"```{language}\n(.*?)```",
        r"```\n(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_response, re.DOTALL)
        if match:
            return match.group(1).strip()
    return raw_response.strip()


def get_model_provider(model_name: str) -> str:
    m = model_name.lower()
    if "gpt" in m or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if "claude" in m:
        return "anthropic"
    if "gemini" in m or m in ("google", "google-gemini"):
        return "google"
    return "huggingface"


async def generate_solution(
    model_name: str,
    prompt: str,
    language: str,
    client: httpx.AsyncClient,
) -> str | None:
    provider = get_model_provider(model_name)
    if provider == "openai":
        raw = await call_openai(prompt, model=model_name, client=client)
    elif provider == "anthropic":
        raw = await call_anthropic(prompt, model=model_name, client=client)
    elif provider == "google":
        m = (model_name or "").strip() or GEMINI_MODEL
        if "gemini" not in m.lower():
            m = GEMINI_MODEL
        raw = await call_gemini(prompt, model=m, client=client)
    else:
        raw = await call_huggingface(prompt, model=model_name, client=client)

    if raw:
        return extract_code(raw, language)
    return None


async def generate_for_problem(
    problem_id: str,
    statement: str,
    models: list[str],
    solutions_per_problem: int,
    client: httpx.AsyncClient,
) -> int:
    """Generate AI solutions for one problem across all languages. Returns count."""
    count = 0
    ext_map = {"cpp": ".cpp", "python": ".py", "java": ".java"}

    for lang in ext_map:
        prompt = build_prompt(statement, lang)
        out_dir = GENERATED_DIR / problem_id
        out_dir.mkdir(parents=True, exist_ok=True)

        generated_for_lang = 0
        for idx in range(solutions_per_problem):
            model = models[idx % len(models)]
            safe_model = model.replace("/", "_").replace("-", "_")
            filename = f"{safe_model}_{lang}_{idx}{ext_map[lang]}"
            out_path = out_dir / filename

            if out_path.exists():
                generated_for_lang += 1
                count += 1
                continue

            code = await generate_solution(model, prompt, lang, client)
            if code and len(code) > 20:
                out_path.write_text(code, encoding="utf-8")
                generated_for_lang += 1
                count += 1

    return count


async def main():
    logger.info("=== AI Solution Generation ===")

    selected_path = DATA_DIR / "selected_problems.txt"
    if not selected_path.exists():
        logger.error(
            f"No selected problems file at {selected_path}. "
            "Run scripts/download_hf_dataset.py (or download_codenet.py) first."
        )
        return

    problems = selected_path.read_text().strip().split("\n")
    logger.info(f"Generating solutions for {len(problems)} problems")

    metadata_dir = DATA_DIR / "downloads" / "metadata"
    statements = load_problem_statements(metadata_dir)

    models = list(AI_GENERATION_MODELS)
    if not models:
        logger.error("No AI models configured. Set AI_GENERATION_MODELS in .env")
        return

    available_models = []
    for m in models:
        provider = get_model_provider(m)
        if provider == "openai" and not OPENAI_API_KEY:
            logger.warning(f"Skipping {m}: no OPENAI_API_KEY")
            continue
        if provider == "anthropic" and not ANTHROPIC_API_KEY:
            logger.warning(f"Skipping {m}: no ANTHROPIC_API_KEY")
            continue
        if provider == "huggingface" and not HF_TOKEN:
            logger.warning(f"Skipping {m}: no HF_TOKEN")
            continue
        if provider == "google" and not GOOGLE_API_KEY:
            logger.warning(f"Skipping {m}: no GOOGLE_API_KEY")
            continue
        available_models.append(m)

    if not available_models:
        logger.error("No models available with current API keys.")
        return

    logger.info(f"Using models: {available_models}")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    async with httpx.AsyncClient() as client:
        for pid in tqdm(problems, desc="Generating AI solutions"):
            statement = statements.get(pid)
            if not statement:
                logger.warning(f"No problem statement for {pid}, using placeholder prompt")
                statement = (
                    f"Solve competitive programming problem {pid}. "
                    f"Write a correct and efficient solution."
                )

            n = await generate_for_problem(
                pid, statement, available_models, AI_SOLUTIONS_PER_PROBLEM, client
            )
            total += n

    logger.info(f"Generated {total} AI solution files in {GENERATED_DIR}")
    logger.info("Done. Run scripts/build_dataset.py next.")


if __name__ == "__main__":
    asyncio.run(main())
