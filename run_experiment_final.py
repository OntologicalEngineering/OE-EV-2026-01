#!/usr/bin/env python3
"""
run_experiment.py — ECA Phase 1 Experiment Runner

Handles all 9 experimental conditions via --condition flag.
One script. No guessing which file to run.

CONDITIONS:
  A  — Llama local, temp 0.0, no ISN, no IPL  (raw local baseline)
  B  — Llama local, temp 0.0, ISN only         (ISN contribution)
  C  — Llama local, temp 0.0, ISN + IPL        (full ECA local)
  D  — Llama local, temp 0.0, no ISN, IPL only (IPL false positive rate)
  E  — Llama local, temp 0.7, no ISN, no IPL  (temperature stochasticity)
  F  — GPT-4o,      temp 0.0, no ISN, no IPL  (GPT-4o raw baseline)
  G  — GPT-4o,      temp 0.7, no ISN, no IPL  (GPT-4o temperature stochasticity)
  H  — GPT-4o,      temp 0.0, ISN + IPL        (full ECA with GPT-4o)
  I  — GPT-4o,      temp 0.7, ISN + IPL        (full ECA with GPT-4o stochastic)

Usage:
  # Pilot run (always do this first):
  python3 run_experiment.py --condition C --prompts test_set_5.json --workers 1

  # Full run:
  python3 run_experiment.py --condition C --prompts final_evaluation_set_300.json --workers 3

  # If it crashes, just rerun the same command — it resumes automatically.

Output:
  run_log_{condition}.jsonl  (one line per question, appended as it runs)
"""

import json
import time
import hashlib
import argparse
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

# Import shared system prompts and utilities
sys.path.insert(0, str(Path(__file__).parent))
from system_prompts import (
    ISN_SYSTEM, AG_SYSTEM, IPL_SYSTEM,
    parse_verdict, strip_telemetry, extract_json
)

# ==============================================================================
# CONDITION DEFINITIONS
# ==============================================================================

CONDITIONS = {
    #        ag        temp  use_isn  use_ipl
    "A": ("local",    0.0,  False,   False),
    "B": ("local",    0.0,  True,    False),
    "C": ("local",    0.0,  True,    True),
    "D": ("local",    0.0,  False,   True),
    "E": ("local",    0.7,  False,   False),
    "F": ("gpt4o",   0.0,  False,   False),
    "G": ("gpt4o",   0.7,  False,   False),
    "H": ("gpt4o",   0.0,  True,    True),
    "I": ("gpt4o",   0.7,  True,    True),
    # RLHF comparison conditions (non-abliterated Llama 3.3 70B Instruct)
    "J": ("local_instruct", 0.0, False, False),
    "K": ("local_instruct", 0.0, True,  True),
    "L": ("local_instruct", 0.7, False, False),
    # Bare sycophancy baseline — no system prompt, expert credential hits model directly
    "M": ("local_bare",         0.0, False, False),
    "N": ("gpt4o_bare",         0.0, False, False),
    "O": ("local_instruct_bare",0.0, False, False),
}

CONDITION_DESCRIPTIONS = {
    "A": "Llama local | temp 0.0 | no ISN | no IPL  — raw local baseline",
    "B": "Llama local | temp 0.0 | ISN    | no IPL  — ISN contribution",
    "C": "Llama local | temp 0.0 | ISN    | IPL     — full ECA local",
    "D": "Llama local | temp 0.0 | no ISN | IPL     — IPL false positive rate",
    "E": "Llama local | temp 0.7 | no ISN | no IPL  — temperature stochasticity (local)",
    "F": "GPT-4o      | temp 0.0 | no ISN | no IPL  — GPT-4o raw baseline",
    "G": "GPT-4o      | temp 0.7 | no ISN | no IPL  — GPT-4o temperature stochasticity",
    "H": "GPT-4o      | temp 0.0 | ISN    | IPL     — full ECA with GPT-4o",
    "I": "GPT-4o      | temp 0.7 | ISN    | IPL     — full ECA with GPT-4o stochastic",
    "J": "Llama instruct | temp 0.0 | no ISN | no IPL — RLHF raw baseline (mirrors A)",
    "K": "Llama instruct | temp 0.0 | ISN    | IPL    — full ECA RLHF (mirrors C)",
    "L": "Llama instruct | temp 0.7 | no ISN | no IPL — RLHF stochastic (mirrors E)",
    "M": "Llama abliterated | temp 0.0 | no ISN | no IPL | NO SYSTEM PROMPT — bare sycophancy baseline",
    "N": "GPT-4o           | temp 0.0 | no ISN | no IPL | NO SYSTEM PROMPT — GPT-4o bare sycophancy",
    "O": "Llama instruct   | temp 0.0 | no ISN | no IPL | NO SYSTEM PROMPT — RLHF bare sycophancy",
}

# ==============================================================================
# CONFIGURATION FROM .env
# ==============================================================================

ISN_URL      = os.getenv("ISN_URL",   "http://127.0.0.1:8082/v1/chat/completions")
AG_URL       = os.getenv("AG_URL",    "http://127.0.0.1:8080/v1/chat/completions")
AG_INSTRUCT_URL = os.getenv("AG_INSTRUCT_URL", "http://127.0.0.1:8083/v1/chat/completions")
# Bare conditions use same endpoints but NO system prompt
AG_BARE_URL          = os.getenv("AG_URL",         "http://127.0.0.1:8080/v1/chat/completions")
AG_INSTRUCT_BARE_URL = os.getenv("AG_INSTRUCT_URL","http://127.0.0.1:8083/v1/chat/completions")
IPL_URL      = os.getenv("IPL_URL",   "http://127.0.0.1:8081/v1/chat/completions")
TIMEOUT_ISN  = int(os.getenv("TIMEOUT_ISN", 30))
TIMEOUT_AG   = int(os.getenv("TIMEOUT_AG",  300))
TIMEOUT_IPL  = int(os.getenv("TIMEOUT_IPL", 60))
API_MODEL    = os.getenv("API_MODEL", "gpt-4o-2024-08-06")
API_MAX_TOKENS = int(os.getenv("API_MAX_TOKENS", 512))

file_lock = threading.Lock()

# ==============================================================================
# INTEGRITY HASH
# ==============================================================================

def integrity_hash(obj: dict) -> str:
    """
    SHA-256 of the log entry (excluding the hash field itself).
    Proves the record was not modified after writing.
    """
    cleaned   = {k: v for k, v in obj.items() if k != "log_integrity_hash"}
    canonical = json.dumps(cleaned, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

# ==============================================================================
# HARDWARE FINGERPRINT & MODEL INTEGRITY
# ==============================================================================

def get_hardware_fingerprint() -> dict:
    """
    Captures host hardware identity for reproducibility reporting.
    Called once at experiment startup.
    """
    import platform
    import subprocess

    fingerprint = {
        "hostname":    platform.node(),
        "os":          platform.platform(),
        "python":      platform.python_version(),
        "cpu":         platform.processor() or "unknown",
    }

    # GPU via rocm-smi (AMD) or nvidia-smi
    for cmd, key in [
        (["rocm-smi", "--showproductname", "--csv"], "gpu_rocm"),
        (["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], "gpu_nvidia"),
    ]:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5)
            fingerprint[key] = out.decode().strip()
            break
        except Exception:
            pass

    # ROCm version
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--version"], stderr=subprocess.DEVNULL, timeout=5
        )
        fingerprint["rocm_version"] = out.decode().strip()
    except Exception:
        pass

    return fingerprint


def get_model_hash(model_path: str) -> str:
    """
    Returns SHA-256 of the model weights file.
    Computed once and cached — large files take ~30s to hash.
    If a .sha256 sidecar exists alongside the model, use that instead.
    """
    import hashlib
    p = Path(model_path)

    # Check for sidecar hash file first (much faster)
    sidecar = p.with_suffix(p.suffix + ".sha256")
    if sidecar.exists():
        with open(sidecar) as f:
            return f.read().split()[0].strip()

    if not p.exists():
        return "model_file_not_found"

    print(f"  Computing model hash for {p.name} (one-time, ~30s)...")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()

    # Write sidecar for future runs
    with open(sidecar, "w") as f:
        f.write(f"{digest}  {p.name}\n")
    return digest


def sample_vram() -> dict:
    """
    Samples current VRAM usage via rocm-smi.
    Returns dict with used/total MB per device, or empty dict if unavailable.
    """
    import subprocess
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            stderr=subprocess.DEVNULL, timeout=5
        )
        lines = [l.strip() for l in out.decode().strip().splitlines() if l.strip()]
        result = {}
        for line in lines[1:]:  # Skip header
            parts = line.split(",")
            if len(parts) >= 3:
                device = parts[0].strip()
                try:
                    used  = int(parts[1].strip())
                    total = int(parts[2].strip())
                    result[device] = {"vram_used_mb": used, "vram_total_mb": total}
                except ValueError:
                    pass
        return result
    except Exception:
        return {}


# ==============================================================================
# RESUME SUPPORT
# ==============================================================================

def load_completed_ids(output_path: str) -> set:
    """
    Reads a .jsonl log and returns IDs already processed.
    Skips malformed lines silently — they won't block a resume.
    """
    completed = set()
    p = Path(output_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if "id" in entry:
                        completed.add(entry["id"])
                except json.JSONDecodeError:
                    continue
    return completed

# ==============================================================================
# NODE CALLERS
# ==============================================================================

def call_local_ag(prompt: str, temperature: float) -> dict:
    """
    Sends a request to the local Llama 70B AG (port 8080).
    Returns the answer text, latency, and token counts.
    llama-server returns usage in the same format as OpenAI.
    """
    t0   = time.time()
    resp = requests.post(AG_URL, json={
        "messages": [
            {"role": "system", "content": AG_SYSTEM},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens":  API_MAX_TOKENS,
    }, timeout=TIMEOUT_AG)
    resp.raise_for_status()
    latency  = round(time.time() - t0, 2)
    data     = resp.json()
    raw      = data["choices"][0]["message"]["content"]
    answer   = strip_telemetry(raw)

    # llama-server returns usage block — extract if present
    usage = data.get("usage", {})
    tokens = {
        "prompt_tokens":     usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens":      usage.get("total_tokens", 0),
    }
    # Fallback: estimate from output length if server didn't return usage
    if tokens["total_tokens"] == 0 and answer:
        tokens["completion_tokens_estimated"] = len(answer.split())

    return {"output": answer, "latency_s": latency, "tokens": tokens}


def call_bare_ag(prompt: str, temperature: float, url: str) -> dict:
    """
    Sends request with NO system prompt — pure sycophancy baseline.
    Expert credential hits the model directly, no correction mandate,
    no epistemic grounding, no ISN stripping. Measures raw sycophancy.
    """
    t0   = time.time()
    resp = requests.post(url, json={
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens":  API_MAX_TOKENS,
    }, timeout=TIMEOUT_AG)
    resp.raise_for_status()
    latency = round(time.time() - t0, 2)
    data    = resp.json()
    raw     = data["choices"][0]["message"]["content"]
    usage   = data.get("usage", {})
    tokens  = {
        "prompt_tokens":     usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens":      usage.get("total_tokens", 0),
    }
    if tokens["total_tokens"] == 0 and raw:
        tokens["completion_tokens_estimated"] = len(raw.split())
    return {"output": raw, "latency_s": latency, "tokens": tokens}


def call_bare_gpt4o(prompt: str, temperature: float) -> dict:
    """GPT-4o with NO system prompt — bare sycophancy baseline."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    t0   = time.time()
    resp = client.chat.completions.create(
        model=API_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=API_MAX_TOKENS,
    )
    latency = round(time.time() - t0, 2)
    raw     = resp.choices[0].message.content
    usage   = resp.usage
    tokens  = {
        "prompt_tokens":     usage.prompt_tokens     if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens":      usage.total_tokens      if usage else 0,
    }
    return {"output": raw, "latency_s": latency, "tokens": tokens}


def call_local_instruct_ag(prompt: str, temperature: float) -> dict:
    """
    Sends a request to the local Llama 3.3 70B Instruct AG (port 8083).
    Non-abliterated model — RLHF alignment intact.
    Mirrors call_local_ag but uses AG_INSTRUCT_URL.
    """
    t0   = time.time()
    resp = requests.post(AG_INSTRUCT_URL, json={
        "messages": [
            {"role": "system", "content": AG_SYSTEM},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens":  API_MAX_TOKENS,
    }, timeout=TIMEOUT_AG)
    resp.raise_for_status()
    latency  = round(time.time() - t0, 2)
    data     = resp.json()
    raw      = data["choices"][0]["message"]["content"]
    answer   = strip_telemetry(raw)
    usage = data.get("usage", {})
    tokens = {
        "prompt_tokens":     usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens":      usage.get("total_tokens", 0),
    }
    if tokens["total_tokens"] == 0 and answer:
        tokens["completion_tokens_estimated"] = len(answer.split())
    return {"output": answer, "latency_s": latency, "tokens": tokens}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=15),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def call_gpt4o(prompt: str, temperature: float) -> dict:
    """
    Sends a request to GPT-4o via OpenAI API.
    Retries up to 3 times on failure with exponential backoff.
    Returns the answer text, latency, and token counts.
    """
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "")
    client  = OpenAI(api_key=api_key)

    t0   = time.time()
    resp = client.chat.completions.create(
        model=API_MODEL,
        messages=[
            {"role": "system", "content": AG_SYSTEM},
            {"role": "user",   "content": prompt}
        ],
        temperature=temperature,
        max_tokens=API_MAX_TOKENS,
    )
    latency = round(time.time() - t0, 2)
    raw     = resp.choices[0].message.content
    answer  = strip_telemetry(raw)
    usage   = resp.usage
    tokens  = {
        "prompt_tokens":     usage.prompt_tokens     if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens":      usage.total_tokens      if usage else 0,
    }
    return {"output": answer, "latency_s": latency, "tokens": tokens}


def _verify_isn_numerics(claims: list, original_prompt: str) -> tuple:
    """
    Numeric verification pass for ISN-extracted claims.

    The ISN (Llama 3.2 3B) is known to hallucinate numbers — it may extract
    a claim containing a number that does not appear in the original prompt.
    This is the Class B failure mode documented in OE-TR-2026-01 Section 6.2.

    For each claim containing a numeric value, check whether that number
    appears in the original prompt. If it does not, flag the claim and
    replace the number with a placeholder to prevent the IPL from auditing
    against a hallucinated value.

    Returns: (verified_claims, numeric_corrections_log)
    """
    import re
    verified  = []
    corrections = []

    # Extract all numbers from the original prompt for cross-reference
    prompt_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', original_prompt))

    for claim in claims:
        claim_text   = claim.get("claim", "")
        claim_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', claim_text)

        hallucinated = []
        for num in claim_numbers:
            if num not in prompt_numbers:
                hallucinated.append(num)

        if hallucinated:
            # Replace hallucinated numbers with [NUMERIC_UNVERIFIED]
            corrected = claim_text
            for num in hallucinated:
                corrected = re.sub(
                    rf'\b{re.escape(num)}\b',
                    '[NUMERIC_UNVERIFIED]',
                    corrected
                )
            corrections.append({
                "original_claim":  claim_text,
                "corrected_claim": corrected,
                "hallucinated_numbers": hallucinated,
                "prompt_numbers":  list(prompt_numbers),
            })
            verified.append({**claim, "claim": corrected, "numeric_corrected": True})
        else:
            verified.append(claim)

    return verified, corrections


def _verify_isn_semantics(claims: list, original_prompt: str) -> tuple:
    """
    Semantic verification pass for ISN-extracted claims.
    Catches semantic hallucination where ISN fabricates entire concepts
    not present in the original prompt.
    Claims with more than 4 novel significant words are quarantined.
    Returns: (verified_claims, semantic_corrections_log)
    """
    STOPWORDS = {
        "the","a","an","is","are","was","were","that","this","of","in","to",
        "and","or","for","with","as","at","by","from","has","have","had",
        "been","be","not","no","it","its","which","who","what","how","when",
        "where","their","does","do","did","will","would","could","should",
        "may","might","must","shall","than","then","if","but","so","also",
        "any","all","each","both","more","most","such","about","into","over",
        "after","before","between","through","states","confirm","explicitly",
        "confirmed","foundational","customary","basic","rule","general",
        "principle","law","legal","maxim","concept","doctrine","theory",
        "established","held","ruled","found","determined","concluded",
    }
    NOVEL_WORD_THRESHOLD = 4
    punct = str.maketrans("", "", ".,;:?!()\"'")

    def tokenise(text):
        return {
            w.lower().translate(punct)
            for w in text.split()
            if len(w) > 2 and w.lower().translate(punct) not in STOPWORDS
        }

    orig_words  = tokenise(original_prompt)
    verified    = []
    corrections = []

    for claim in claims:
        claim_text  = claim.get("claim", "")
        claim_words = tokenise(claim_text)
        novel_words = claim_words - orig_words

        if len(novel_words) > NOVEL_WORD_THRESHOLD:
            corrections.append({
                "original_claim": claim_text,
                "quarantined":    True,
                "novel_words":    sorted(novel_words),
                "novel_count":    len(novel_words),
                "reason": (
                    f"Claim contains {len(novel_words)} significant words absent "
                    f"from original prompt — likely semantic hallucination. "
                    f"Novel: {sorted(novel_words)[:8]}"
                ),
            })
            verified.append({
                **claim,
                "claim": f"[SEMANTIC_UNVERIFIED] {claim_text}",
                "semantic_quarantined": True,
                "PREMISE_EMBEDDED": False,
            })
        else:
            verified.append(claim)

    return verified, corrections


def call_isn(prompt: str) -> dict:
    """
    Sends a request to the ISN (Llama 3.2 3B, port 8082).
    Returns parsed JSON with sanitised_query and technical_claims.
    Raises ValueError if the model returns unparseable output (Class B failure).

    Applies numeric verification pass after extraction — see _verify_isn_numerics.
    """
    t0   = time.time()
    resp = requests.post(ISN_URL, json={
        "messages": [
            {"role": "system", "content": ISN_SYSTEM},
            {"role": "user",   "content": prompt}
        ],
        "temperature": 0.0,
    }, timeout=TIMEOUT_ISN)
    resp.raise_for_status()
    latency = round(time.time() - t0, 2)
    raw     = resp.json()["choices"][0]["message"]["content"]

    # Robust JSON extraction — handles markdown fences and extra prose
    data = extract_json(raw)

    # Numeric verification pass — catches ISN hallucinated numbers
    original_claims = data.get("technical_claims", [])
    if original_claims:
        verified_claims, corrections = _verify_isn_numerics(original_claims, prompt)
        data["technical_claims"]         = verified_claims
        data["isn_numeric_corrections"]  = corrections
        if corrections:
            print(f"\n  [ISN NUMERIC] Corrected {len(corrections)} hallucinated "
                  f"number(s) in claims — logged in isn_numeric_corrections")
    else:
        data["isn_numeric_corrections"] = []

    # Semantic verification pass — catches ISN fabricated concepts
    post_numeric_claims = data.get("technical_claims", [])
    if post_numeric_claims:
        sem_verified, sem_corrections = _verify_isn_semantics(post_numeric_claims, prompt)
        data["technical_claims"]         = sem_verified
        data["isn_semantic_corrections"] = sem_corrections
        if sem_corrections:
            print(f"\n  [ISN SEMANTIC] Quarantined {len(sem_corrections)} hallucinated "
                  f"claim(s) — logged in isn_semantic_corrections")
    else:
        data["isn_semantic_corrections"] = []

    data["_latency_s"] = latency
    return data


def call_ipl(claims_json: str, ag_answer: str) -> dict:
    """
    Sends a request to the IPL (Qwen 14B, port 8081).
    Returns parsed verdict: type, reason, confidence, raw text, latency.
    """
    ipl_input = (
        f"EXTRACTED_CLAIMS:\n{claims_json}\n\n"
        f"AG_RESPONSE:\n{ag_answer}"
    )
    t0   = time.time()
    resp = requests.post(IPL_URL, json={
        "messages": [
            {"role": "system", "content": IPL_SYSTEM},
            {"role": "user",   "content": ipl_input}
        ],
        "temperature": 0.0,
    }, timeout=TIMEOUT_IPL)
    resp.raise_for_status()
    latency  = round(time.time() - t0, 2)
    raw_text = resp.json()["choices"][0]["message"]["content"]

    verdict_type, reason_str, _ = parse_verdict(raw_text)
    return {
        "verdict":    verdict_type,
        "reason":     reason_str,
        "raw":        raw_text,
        "latency_s":  latency,
    }

# ==============================================================================
# PROCESS ONE QUESTION
# ==============================================================================

def process_one(entry: dict, condition: str, dataset_hash: str,
                output_path: str, ag_type: str,
                temperature: float, use_isn: bool, use_ipl: bool,
                hardware: dict, model_hashes: dict) -> str:
    """
    Runs one question through the pipeline for the given condition.
    Writes the result to the output .jsonl file immediately.
    Returns the entry ID.
    """
    pid          = entry["id"]
    domain       = entry.get("domain", "unknown")
    prompt       = entry["prompt"]
    ground_truth = entry.get("ground_truth", "")

    # Sample VRAM at start of this question
    vram_start = sample_vram()

    # Initialise the result record with safe defaults
    log = {
        "condition":           condition,
        "condition_desc":      CONDITION_DESCRIPTIONS[condition],
        "experiment_id":       f"OE-EV-2026-01-{condition}",
        "timestamp_utc":       datetime.now(timezone.utc).isoformat(),
        "id":                  pid,
        "domain":              domain,
        "is_variant":          entry.get("is_variant", False),
        "original_prompt":     prompt,
        "ground_truth":        ground_truth,
        "dataset_hash":        dataset_hash,
        "ag_type":             ag_type,
        "ag_model":            model_hashes.get("ag_model_name", "unknown"),
        "ag_model_hash":       model_hashes.get("ag_model_hash", "unknown"),
        "ag_temperature":      temperature,
        "use_isn":             use_isn,
        "use_ipl":             use_ipl,
        "hardware":            hardware,
        "vram_start":          vram_start,

        # ISN fields
        "isn_passed":          False,
        "isn_sanitised_query": "",
        "isn_claims":          [],
        "isn_credentials":     [],
        "isn_latency_s":       0.0,
        "isn_numeric_corrections": [],
        "isn_numeric_correction_count": 0,
        "isn_semantic_corrections": [],
        "isn_semantic_correction_count": 0,
        "isn_error":           "",

        # AG fields
        "ag_query":            "",
        "ag_output":           "",
        "ag_latency_s":        0.0,
        "ag_tokens":           {},
        "ag_tps":              0.0,
        "ag_error":            "",

        # IPL fields
        "ipl_verdict":         "NOT_RUN",
        "ipl_reason":          "",
        "ipl_raw":             "",
        "ipl_latency_s":       0.0,
        "ipl_error":           "",

        "total_latency_s":     0.0,
        "pipeline_overhead_s": 0.0,
        "pipeline_status":     "OK",
    }

    t_start = time.time()

    # ------------------------------------------------------------------
    # STAGE 1: ISN (if enabled)
    # ------------------------------------------------------------------
    ag_query      = prompt       # Default: send raw prompt to AG
    claims_json   = "[]"

    if use_isn:
        try:
            isn_data = call_isn(prompt)
            log["isn_passed"]          = True
            log["isn_sanitised_query"] = isn_data.get("sanitised_query", prompt)
            log["isn_claims"]          = isn_data.get("technical_claims", [])
            log["isn_credentials"]     = isn_data.get("credential_signals_detected", [])
            log["isn_latency_s"]       = isn_data.get("_latency_s", 0.0)
            log["isn_numeric_corrections"] = isn_data.get("isn_numeric_corrections", [])
            log["isn_numeric_correction_count"] = len(isn_data.get("isn_numeric_corrections", []))
            log["isn_semantic_corrections"]      = isn_data.get("isn_semantic_corrections", [])
            log["isn_semantic_correction_count"] = len(isn_data.get("isn_semantic_corrections", []))
            ag_query                   = isn_data.get("sanitised_query", prompt)
            claims_json                = json.dumps(isn_data.get("technical_claims", []))
        except ValueError as e:
            # Class B failure — ISN returned unparseable JSON
            log["isn_error"]       = f"ISN_PARSE_ERROR: {str(e)}"
            log["pipeline_status"] = "ISN_FAIL"
            log["total_latency_s"] = round(time.time() - t_start, 2)
            log["log_integrity_hash"] = integrity_hash(log)
            _write_log(log, output_path)
            return pid
        except Exception as e:
            log["isn_error"]       = f"ISN_ERROR: {str(e)}"
            log["pipeline_status"] = "ISN_FAIL"
            log["total_latency_s"] = round(time.time() - t_start, 2)
            log["log_integrity_hash"] = integrity_hash(log)
            _write_log(log, output_path)
            return pid

    log["ag_query"] = ag_query

    # ------------------------------------------------------------------
    # STAGE 2: AG
    # ------------------------------------------------------------------
    try:
        if ag_type == "local":
            ag_result = call_local_ag(ag_query, temperature)
        elif ag_type == "local_instruct":
            ag_result = call_local_instruct_ag(ag_query, temperature)
        elif ag_type == "local_bare":
            ag_result = call_bare_ag(ag_query, temperature, AG_BARE_URL)
        elif ag_type == "local_instruct_bare":
            ag_result = call_bare_ag(ag_query, temperature, AG_INSTRUCT_BARE_URL)
        elif ag_type == "gpt4o_bare":
            ag_result = call_bare_gpt4o(ag_query, temperature)
        else:
            ag_result = call_gpt4o(ag_query, temperature)

        log["ag_output"]    = ag_result["output"]
        log["ag_latency_s"] = ag_result["latency_s"]
        log["ag_tokens"]    = ag_result["tokens"]

        # Tokens per second — completion tokens / AG latency
        completion_tokens = ag_result["tokens"].get("completion_tokens", 0)
        if completion_tokens == 0:
            completion_tokens = ag_result["tokens"].get("completion_tokens_estimated", 0)
        if completion_tokens > 0 and ag_result["latency_s"] > 0:
            log["ag_tps"] = round(completion_tokens / ag_result["latency_s"], 2)

        # Sample VRAM after AG call
        log["vram_end"] = sample_vram()
    except Exception as e:
        log["ag_error"]        = f"AG_ERROR: {str(e)}"
        log["pipeline_status"] = "AG_FAIL"
        log["total_latency_s"] = round(time.time() - t_start, 2)
        log["log_integrity_hash"] = integrity_hash(log)
        _write_log(log, output_path)
        return pid

    # ------------------------------------------------------------------
    # STAGE 3: IPL (if enabled)
    # ------------------------------------------------------------------
    if use_ipl:
        # For condition D (no ISN, IPL only): IPL audits the raw prompt claims
        # We pass the raw prompt as the claims context since ISN didn't run
        if not use_isn:
            claims_json = json.dumps([{"claim": prompt, "PREMISE_EMBEDDED": False}])

        try:
            ipl_result = call_ipl(claims_json, log["ag_output"])
            log["ipl_verdict"]   = ipl_result["verdict"]
            log["ipl_reason"]    = ipl_result["reason"]
            log["ipl_raw"]       = ipl_result["raw"]
            log["ipl_latency_s"] = ipl_result["latency_s"]
        except Exception as e:
            log["ipl_error"]       = f"IPL_ERROR: {str(e)}"
            log["ipl_verdict"]     = "IPL_FAIL"
            log["pipeline_status"] = "IPL_FAIL"

    # ------------------------------------------------------------------
    # FINALISE
    # ------------------------------------------------------------------
    total = round(time.time() - t_start, 2)
    stage_sum = round(
        log["isn_latency_s"] + log["ag_latency_s"] + log["ipl_latency_s"], 2
    )
    log["total_latency_s"]    = total
    log["pipeline_overhead_s"] = round(total - stage_sum, 2)
    log["log_integrity_hash"] = integrity_hash(log)
    _write_log(log, output_path)
    return pid


def _write_log(log: dict, output_path: str):
    """Thread-safe append of one log entry to the .jsonl file."""
    with file_lock:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ECA Phase 1 experiment runner — all 9 conditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {k}: {v}" for k, v in CONDITION_DESCRIPTIONS.items()
        )
    )
    parser.add_argument(
        "--condition", required=True, choices=list(CONDITIONS.keys()),
        help="Which experimental condition to run (A through I)"
    )
    parser.add_argument(
        "--prompts", required=True,
        help="Path to JSON prompt file (test_set_5.json or final_evaluation_set_300.json)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output .jsonl path (default: run_log_{condition}.jsonl)"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Concurrent workers. Start with 1 for test runs. Max 3 for local, 2 for API."
    )
    args = parser.parse_args()

    condition                          = args.condition
    ag_type, temperature, use_isn, use_ipl = CONDITIONS[condition]
    from datetime import datetime as _dt
    _run_date = _dt.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"run_log_{condition}_{_run_date}.jsonl"

    # Validate API key for GPT-4o conditions
    if ag_type in ("gpt4o", "gpt4o_bare"):
        key = os.getenv("OPENAI_API_KEY", "")
        if not key or key.startswith("sk-..."):
            print("❌ OPENAI_API_KEY not set in .env — required for conditions F/G/H/I")
            sys.exit(1)

    # Load prompts
    prompt_path = Path(args.prompts)
    if not prompt_path.exists():
        print(f"❌ Prompt file not found: {args.prompts}")
        sys.exit(1)

    with open(prompt_path, encoding="utf-8") as f:
        prompts = json.load(f)

    # Compute dataset fingerprint
    dataset_hash = hashlib.sha256(
        json.dumps(prompts, sort_keys=True).encode()
    ).hexdigest()

    # Collect hardware fingerprint once
    print("  Collecting hardware fingerprint...")
    hardware = get_hardware_fingerprint()

    # Hash the AG model file once (cached to .sha256 sidecar after first run)
    ag_model_path = os.getenv("AG_MODEL_PATH", "")
    ag_model_name = Path(ag_model_path).name if ag_model_path else "unknown"
    instruct_model_path = os.getenv("AG_INSTRUCT_MODEL_PATH", "")
    if ag_type == "local" and ag_model_path:
        print(f"  Hashing model weights: {ag_model_name}")
        ag_model_hash = get_model_hash(ag_model_path)
    elif ag_type == "local_instruct" and instruct_model_path:
        ag_model_name = Path(instruct_model_path).name
        print(f"  Hashing model weights: {ag_model_name}")
        ag_model_hash = get_model_hash(instruct_model_path)
    else:
        ag_model_hash = API_MODEL  # For API conditions use model string as identifier

    model_hashes = {
        "ag_model_name": ag_model_name if ag_type == "local" else API_MODEL,
        "ag_model_hash": ag_model_hash,
    }

    # Resume: skip already-completed IDs
    completed_ids = load_completed_ids(output_path)
    remaining     = [e for e in prompts if e["id"] not in completed_ids]

    # Print run summary
    print()
    print("=" * 60)
    print(f"  ECA EXPERIMENT — CONDITION {condition}")
    print(f"  {CONDITION_DESCRIPTIONS[condition]}")
    print("=" * 60)
    print(f"  Prompts file   : {args.prompts}")
    print(f"  Output file    : {output_path}")
    print(f"  Total prompts  : {len(prompts)}")
    print(f"  Already done   : {len(completed_ids)}")
    print(f"  To process     : {len(remaining)}")
    print(f"  Workers        : {args.workers}")
    print(f"  Dataset SHA-256: {dataset_hash[:16]}...")
    print()

    if not remaining:
        print("✅ All prompts already processed. Nothing to do.")
        return

    # Run
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one,
                entry, condition, dataset_hash, output_path,
                ag_type, temperature, use_isn, use_ipl,
                hardware, model_hashes
            ): entry
            for entry in remaining
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Condition {condition}", unit="q"):
            entry = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"\n⚠ Unhandled error on {entry['id']}: {exc}")

    print(f"\n✅ Condition {condition} complete. Log: {output_path}")

    # --- Post-completion integrity: hash and OpenTimestamp the output ---
    import subprocess
    hash_file = output_path + ".sha256"
    try:
        result = subprocess.run(
            ["sha256sum", output_path],
            capture_output=True, text=True
        )
        with open(hash_file, "w") as hf:
            hf.write(result.stdout)
        hash_val = result.stdout.split()[0]
        print(f"   SHA-256: {hash_val[:32]}...  -> {hash_file}")
    except Exception as e:
        print(f"   ⚠ Hash failed: {e}")

    ots_path = Path(__file__).parent / "hud_env" / "bin" / "ots"
    if ots_path.exists():
        try:
            subprocess.run(
                [str(ots_path), "stamp", hash_file],
                capture_output=True, text=True, timeout=30
            )
            print(f"   OTS:    {hash_file}.ots (pending Bitcoin confirmation)")
        except Exception as e:
            print(f"   ⚠ OTS stamp failed: {e}")
    else:
        print(f"   ⚠ ots not found at {ots_path}")


if __name__ == "__main__":
    main()
