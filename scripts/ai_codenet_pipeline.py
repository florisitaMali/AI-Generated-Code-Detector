"""
AI-CodeNet generator — mirrors IBM Project_CodeNet layout under AI_CODENET_ROOT:

  data/{problem_id}/{Language}/{submission_id}.{ext}
  metadata/problem_list.csv (copy from CodeNet)
  metadata/{problem_id}.csv   (one row per AI submission)

Resumable: existing submission files are skipped.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from bs4 import BeautifulSoup
from loguru import logger

from config.settings import (
    AI_CODENET_RATE_SLEEP,
    AI_CODENET_ROOT,
    AI_CODENET_RUN_TESTS,
    AI_GENERATION_MODELS,
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    HF_TOKEN,
    MAX_CODENET_PROBLEMS,
    OPENAI_API_KEY,
)
from src.datasets.codenet_paths import get_codenet_root
from src.datasets.codenet_schema import (
    CODENET_LANGUAGES,
    LANG_DISPLAY_TO_META,
    SUBMISSION_CSV_HEADER,
    make_submission_id,
    make_user_id,
    strip_code_fences,
)

OUTPUT_ROOT = AI_CODENET_ROOT

SYSTEM_PROMPT = (
    "You are an expert competitive programmer. "
    "Solve the given programming problem. "
    "Output ONLY the complete, compilable source code with no explanation, "
    "no markdown fences, and no preamble. "
    "The code must read from stdin and write to stdout."
)


def _user_prompt(problem_text: str, language: str) -> str:
    return f"Solve this competitive programming problem in {language}.\n\n{problem_text[:6000]}"


def parse_problem_description(codenet_root: Path, problem_id: str) -> str:
    for name in (f"{problem_id}.html", f"{problem_id}.htm"):
        html_path = codenet_root / "problem_descriptions" / name
        if html_path.exists():
            with open(html_path, encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            return soup.get_text(separator="\n", strip=True)
    return ""


def load_problem_list(codenet_root: Path) -> list[dict[str, str]]:
    csv_path = codenet_root / "metadata" / "problem_list.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Set CODENET_ROOT or download CodeNet metadata.")
    problems: list[dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            problems.append(row)
    return problems


def load_sample_io(codenet_root: Path, problem_id: str) -> tuple[str, str]:
    base = codenet_root / "derived" / "input_output" / "data" / problem_id
    inp_file = base / "input" / "input.txt"
    out_file = base / "output" / "output.txt"
    if inp_file.exists() and out_file.exists():
        return inp_file.read_text(), out_file.read_text()
    return "", ""


def infer_provider(model_name: str) -> str:
    if "/" in model_name:
        return "huggingface"
    m = model_name.lower()
    if "claude" in m:
        return "anthropic"
    if "gemini" in m or m.startswith("google/"):
        return "google"
    if "gpt" in m or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "openai"
    return "openai"


def generate_code(
    model_name: str, problem_text: str, language: str
) -> tuple[str, str, str]:
    """
    Returns (raw_code, prompt_tokens, completion_tokens) as strings for CSV.
    """
    user = _user_prompt(problem_text, language)
    pr = ""
    comp = ""
    text = ""

    prov = infer_provider(model_name)
    if prov == "anthropic":
        import anthropic

        if not ANTHROPIC_API_KEY:
            raise RuntimeError(f"No ANTHROPIC_API_KEY for {model_name}")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=model_name,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        text = msg.content[0].text.strip()
        if getattr(msg, "usage", None):
            pr = str(getattr(msg.usage, "input_tokens", "") or "")
            comp = str(getattr(msg.usage, "output_tokens", "") or "")
    elif prov == "openai":
        from openai import OpenAI

        if not OPENAI_API_KEY:
            raise RuntimeError(f"No OPENAI_API_KEY for {model_name}")
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model_name,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        if resp.usage:
            pr = str(resp.usage.prompt_tokens or "")
            comp = str(resp.usage.completion_tokens or "")
    elif prov == "google":
        import google.generativeai as genai

        key = GEMINI_API_KEY
        if not key:
            raise RuntimeError(f"No GEMINI_API_KEY or GOOGLE_API_KEY for {model_name}")
        genai.configure(api_key=key)
        model_id = model_name if "gemini" in model_name.lower() else GEMINI_MODEL
        model = genai.GenerativeModel(model_id)
        resp = model.generate_content(SYSTEM_PROMPT + "\n\n" + user)
        try:
            text = (resp.text or "").strip()
        except ValueError:
            text = ""
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            pr = str(getattr(um, "prompt_token_count", "") or "")
            comp = str(getattr(um, "candidates_token_count", "") or "")
    else:
        if not HF_TOKEN:
            raise RuntimeError(f"No HF_TOKEN for HuggingFace model {model_name}")
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user}"
        with httpx.Client(timeout=180.0) as http:
            r = http.post(
                f"https://api-inference.huggingface.co/models/{model_name}",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={
                    "inputs": full_prompt,
                    "parameters": {
                        "max_new_tokens": 2048,
                        "temperature": 0.7,
                        "return_full_text": False,
                    },
                },
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data:
                text = (data[0].get("generated_text") or "").strip()
            elif isinstance(data, dict) and "generated_text" in data:
                text = str(data["generated_text"]).strip()

    code = strip_code_fences(text)
    return code, pr, comp


def test_solution_cpp(code: str, sample_input: str, expected_output: str) -> tuple[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "sol.cpp")
        exe = os.path.join(tmp, "sol.exe" if sys.platform == "win32" else "sol")
        with open(src, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            r = subprocess.run(
                ["g++", "-O2", "-std=c++17", "-o", exe, src],
                capture_output=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "Compile Error", "g++ missing or timeout"
        if r.returncode != 0:
            return "Compile Error", (r.stderr or b"").decode(errors="replace")[:500]
        try:
            r = subprocess.run(
                [exe],
                input=sample_input,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            return "Time Limit Exceeded", ""
        if r.returncode != 0:
            return "Runtime Error", (r.stderr or "")[:200]
        actual = r.stdout.strip()
        exp = expected_output.strip()
        if actual == exp:
            return "Accepted", actual
        return "Wrong Answer", f"Got {actual!r} expected {exp!r}"


def test_solution_python(code: str, sample_input: str, expected_output: str) -> tuple[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        sol = Path(tmp) / "sol.py"
        sol.write_text(code, encoding="utf-8")
        try:
            r = subprocess.run(
                [sys.executable, str(sol)],
                input=sample_input,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            return "Time Limit Exceeded", ""
        if r.returncode != 0:
            return "Runtime Error", (r.stderr or "")[:500]
        actual = r.stdout.strip()
        exp = expected_output.strip()
        if actual == exp:
            return "Accepted", actual
        return "Wrong Answer", f"Got {actual!r} expected {exp!r}"


def test_solution_java(code: str, sample_input: str, expected_output: str) -> tuple[str, str]:
    if shutil.which("javac") is None or shutil.which("java") is None:
        return "Unknown", "Java toolchain not found"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        src = d / "Main.java"
        src.write_text(code, encoding="utf-8")
        try:
            r = subprocess.run(
                ["javac", str(src)],
                cwd=tmp,
                capture_output=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return "Compile Error", ""
        if r.returncode != 0:
            return "Compile Error", (r.stderr or b"").decode(errors="replace")[:500]
        try:
            r = subprocess.run(
                ["java", "-cp", tmp, "Main"],
                input=sample_input,
                capture_output=True,
                text=True,
                timeout=5,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return "Time Limit Exceeded", ""
        if r.returncode != 0:
            return "Runtime Error", (r.stderr or "")[:200]
        actual = r.stdout.strip()
        exp = expected_output.strip()
        if actual == exp:
            return "Accepted", actual
        return "Wrong Answer", f"Got {actual!r} expected {exp!r}"


def judge_solution(
    code: str,
    language: str,
    sample_in: str,
    sample_out: str,
    run_tests: bool,
) -> str:
    if not run_tests:
        return "Unknown"
    if not sample_in:
        return "Judge Not Available"
    if language == "C++":
        status, _ = test_solution_cpp(code, sample_in, sample_out)
        return status
    if language == "Python":
        status, _ = test_solution_python(code, sample_in, sample_out)
        return status
    if language == "Java":
        return test_solution_java(code, sample_in, sample_out)[0]
    return "Unknown"


def ensure_dirs(problem_id: str, language: str) -> None:
    (OUTPUT_ROOT / "data" / problem_id / language).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "metadata").mkdir(parents=True, exist_ok=True)


def write_source_file(problem_id: str, language: str, submission_id: str, code: str) -> int:
    ext = LANG_DISPLAY_TO_META[language]["ext"]
    path = OUTPUT_ROOT / "data" / problem_id / language / f"{submission_id}.{ext}"
    path.write_text(code, encoding="utf-8")
    return path.stat().st_size


def append_metadata_row(problem_id: str, row: dict[str, str]) -> None:
    csv_path = OUTPUT_ROOT / "metadata" / f"{problem_id}.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_CSV_HEADER, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def copy_problem_list(codenet_root: Path) -> None:
    src = codenet_root / "metadata" / "problem_list.csv"
    dst = OUTPUT_ROOT / "metadata" / "problem_list.csv"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)


def already_done(problem_id: str, model_name: str, language: str) -> bool:
    ext = LANG_DISPLAY_TO_META[language]["ext"]
    sid = make_submission_id(problem_id, model_name, language)
    return (OUTPUT_ROOT / "data" / problem_id / language / f"{sid}.{ext}").exists()


def filter_available_models(models: list[str]) -> list[str]:
    out: list[str] = []
    for m in models:
        prov = infer_provider(m)
        if prov == "anthropic" and not ANTHROPIC_API_KEY:
            logger.warning(f"Skipping {m}: no ANTHROPIC_API_KEY")
            continue
        if prov == "openai" and not OPENAI_API_KEY:
            logger.warning(f"Skipping {m}: no OPENAI_API_KEY")
            continue
        if prov == "google" and not GEMINI_API_KEY:
            logger.warning(f"Skipping {m}: no GEMINI_API_KEY / GOOGLE_API_KEY")
            continue
        if prov == "huggingface" and not HF_TOKEN:
            logger.warning(f"Skipping {m}: no HF_TOKEN")
            continue
        out.append(m)
    return out


def run() -> None:
    codenet_root = get_codenet_root()
    logger.info(f"Using CodeNet root: {codenet_root}")

    models = filter_available_models(list(AI_GENERATION_MODELS))
    if not models:
        raise RuntimeError(
            "No models available. Configure AI_GENERATION_MODELS and API keys in .env"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    copy_problem_list(codenet_root)

    problems = load_problem_list(codenet_root)
    if MAX_CODENET_PROBLEMS is not None:
        problems = problems[: MAX_CODENET_PROBLEMS]

    total = len(problems) * len(models) * len(CODENET_LANGUAGES)
    done = 0

    logger.info(f"Models: {models}; languages: {CODENET_LANGUAGES}; problems: {len(problems)}")

    for prob in problems:
        pid = prob.get("id", "").strip()
        if not pid:
            continue
        problem_text = parse_problem_description(codenet_root, pid)
        if not problem_text:
            logger.warning(f"[{pid}] No HTML description — skipping")
            continue

        sample_in, sample_out = (
            load_sample_io(codenet_root, pid) if AI_CODENET_RUN_TESTS else ("", "")
        )

        for model_name in models:
            for language in CODENET_LANGUAGES:
                done += 1
                if already_done(pid, model_name, language):
                    logger.info(f"[{done}/{total}] {pid} {model_name} {language} — skip (exists)")
                    continue

                logger.info(f"[{done}/{total}] {pid} {model_name} {language} …")

                try:
                    code, ptok, ctok = generate_code(model_name, problem_text, language)
                except Exception as e:
                    logger.warning(f"Generation failed: {e}")
                    time.sleep(5)
                    continue

                if len(code.strip()) < 10:
                    logger.warning("Empty or trivial code — skipping write")
                    time.sleep(AI_CODENET_RATE_SLEEP)
                    continue

                status = judge_solution(
                    code, language, sample_in, sample_out, AI_CODENET_RUN_TESTS
                )

                sid = make_submission_id(pid, model_name, language)
                uid = make_user_id(model_name, language)
                lang_meta = LANG_DISPLAY_TO_META[language]
                ensure_dirs(pid, language)
                code_size = write_source_file(pid, language, sid, code)

                append_metadata_row(
                    pid,
                    {
                        "submission_id": sid,
                        "problem_id": pid,
                        "user_id": uid,
                        "date": str(int(time.time())),
                        "language": language,
                        "original_language": str(lang_meta["original_language"]),
                        "filename_ext": str(lang_meta["ext"]),
                        "status": status,
                        "cpu_time": "-1",
                        "memory": "-1",
                        "code_size": str(code_size),
                        "accuracy": "",
                        "model_name": model_name,
                        "generation_prompt_tokens": ptok,
                        "generation_completion_tokens": ctok,
                    },
                )

                logger.info(f"  → {status} ({code_size} bytes)")
                time.sleep(AI_CODENET_RATE_SLEEP)

    logger.info(f"Done. AI-CodeNet dataset written to {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    run()
