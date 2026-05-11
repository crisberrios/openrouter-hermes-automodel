#!/usr/bin/env python3
"""
Automodel runner

Collects signals from OpenRouter (catalog + pricing, intelligence leaderboard,
per-agentic-app model usage) and from a single web-enabled LLM call (recent
news + community sentiment for agentic LLM use), then ranks models into three
10-item lists:

  - free-models.json       — best free models currently available
  - balanced-models.json   — best mix of free + paid by quality-per-dollar
  - best-models.json       — best models overall, regardless of price

Lists are written to ~/automodel/output/ and a Telegram notification is sent
on completion.

The sentiment-enrichment prompt lives alongside this script in
`sentiment_prompt.md` so it can be edited without touching the code.

Designed to be invoked by a Hermes cron job. Has no Hermes runtime dependency
beyond the .env file under ~/.hermes/.env.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------------
# Paths & config
# ----------------------------------------------------------------------------
HOME = Path(os.path.expanduser("~"))
ROOT = HOME / "automodel"
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / "cache"
LOG_DIR = ROOT / "logs"
ENV_FILE = HOME / ".hermes" / ".env"

SCRIPT_DIR = Path(__file__).resolve().parent
# Override location of the sentiment prompt with AUTOMODEL_SENTIMENT_PROMPT.
SENTIMENT_PROMPT_PATH = Path(
    os.environ.get("AUTOMODEL_SENTIMENT_PROMPT") or (SCRIPT_DIR / "sentiment_prompt.md")
)
COMPARATIVE_PROMPT_PATH = Path(
    os.environ.get("AUTOMODEL_COMPARATIVE_PROMPT") or (SCRIPT_DIR / "comparative_prompt.md")
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Stage-1 produces this many candidates per list; stage-2 (head-to-head
# benchmark comparison) re-ranks them down to LIST_SIZE for the final output.
LIST_SIZE = 10
SHORTLIST_SIZE = 15
TRACKED_APPS = ["hermes-agent", "openclaw"]

USER_AGENT = "automodel-runner/1.0 (+https://openrouter.ai)"
HTTP_TIMEOUT = 30
# Longer ceiling for the two LLM calls. `:online` models route through
# OpenRouter's web plugin which can push end-to-end latency past 60s on
# bigger prompts (notably the 32-model comparative re-rank).
LLM_HTTP_TIMEOUT = 180

# LLM used for the news + sentiment enrichment call. `:online` enables
# OpenRouter's built-in web plugin so the model can actually search the web.
SENTIMENT_MODEL = os.environ.get("AUTOMODEL_SENTIMENT_MODEL", "openai/gpt-5.4:online")
SENTIMENT_MAX_TOKENS = 1500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "automodel.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("automodel")


# ----------------------------------------------------------------------------
# Env loading (no python-dotenv dep)
# ----------------------------------------------------------------------------
def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if not ENV_FILE.exists():
        return env
    for raw in ENV_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if key and value:
            env[key] = value
    return env


ENV = load_env()
OPENROUTER_API_KEY = ENV.get("OPENROUTER_API_KEY", "")
TELEGRAM_BOT_TOKEN = ENV.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_HOME_CHANNEL = ENV.get("TELEGRAM_HOME_CHANNEL", "")


# ----------------------------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------------------------
def _http(
    url: str,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict | None = None,
    timeout: int = HTTP_TIMEOUT,
) -> bytes:
    req = urllib.request.Request(url, method=method, data=data)
    req.add_header("User-Agent", USER_AGENT)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def http_get_json(url: str, headers: dict | None = None) -> Any:
    return json.loads(_http(url, headers=headers).decode("utf-8"))


def http_get_text(url: str, headers: dict | None = None) -> str:
    return _http(url, headers=headers).decode("utf-8", "ignore")


# ----------------------------------------------------------------------------
# OpenRouter catalog
# ----------------------------------------------------------------------------
@dataclass
class ModelEntry:
    id: str
    name: str
    context_length: int = 0
    prompt_price: float = 0.0      # USD per token
    completion_price: float = 0.0  # USD per token
    is_free: bool = False
    supports_tools: bool = False
    supports_reasoning: bool = False
    description: str = ""

    # Release metadata (from OpenRouter catalog `created` field)
    release_date: datetime | None = None

    # Signals (filled later)
    intelligence_score: float = 0.0
    # Aggregate usage across tracked agentic apps (TRACKED_APPS) — replaces
    # the previous global weekly leaderboard signal.
    agentic_token_volume: int = 0
    agentic_rank: int = 0
    sentiment_score: float = 0.0   # -1..+1
    sentiment_notes: str = ""
    recency_score: float = 0.0     # 0..1, derived from release_date
    comparative_score: float = 0.0       # 0..1, filled by stage-2 head-to-head call
    comparative_rationale: str = ""      # short reasoning string from stage-2

    # Composite (filled at ranking time)
    preliminary_quality_score: float = 0.0  # stage-1, before comparative re-rank
    quality_score: float = 0.0              # final, after stage-2 blend (or = preliminary if stage-2 skipped)
    value_score: float = 0.0       # quality per dollar


def _parse_release(raw_value: Any) -> datetime | None:
    """Parse OpenRouter's `created` field. Usually a Unix timestamp (int or
    float); sometimes an ISO-8601 string. Returns None if unparseable or
    obviously bogus (before 2020 or in the far future)."""
    if raw_value is None:
        return None
    try:
        if isinstance(raw_value, (int, float)):
            dt = datetime.fromtimestamp(float(raw_value), tz=timezone.utc)
        else:
            s = str(raw_value).strip()
            if not s:
                return None
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return None
    # Sanity: OpenRouter launched after 2020, and a future date by more than
    # a few days is almost certainly a unit-conversion bug upstream.
    now = datetime.now(timezone.utc)
    if dt.year < 2020 or dt > now.replace(year=now.year + 1):
        return None
    return dt


def fetch_catalog() -> dict[str, ModelEntry]:
    log.info("Fetching OpenRouter /models")
    data = http_get_json("https://openrouter.ai/api/v1/models")
    models: dict[str, ModelEntry] = {}
    for raw in data.get("data", []):
        model_id = raw.get("id", "")
        if not model_id:
            continue
        pricing = raw.get("pricing") or {}
        prompt = float(pricing.get("prompt") or 0)
        completion = float(pricing.get("completion") or 0)
        is_free = model_id.endswith(":free") or (prompt == 0 and completion == 0)
        params = set(raw.get("supported_parameters") or [])
        models[model_id] = ModelEntry(
            id=model_id,
            name=raw.get("name") or model_id,
            context_length=int(raw.get("context_length") or 0),
            prompt_price=prompt,
            completion_price=completion,
            is_free=is_free,
            supports_tools="tools" in params or "tool_choice" in params,
            supports_reasoning="reasoning" in params or "include_reasoning" in params,
            description=(raw.get("description") or "")[:600],
            release_date=_parse_release(raw.get("created")),
        )
    with_dates = sum(1 for m in models.values() if m.release_date)
    log.info("  catalog: %d models (%d with release date)", len(models), with_dates)
    return models


# ----------------------------------------------------------------------------
# OpenRouter Next.js payload helpers
# ----------------------------------------------------------------------------
_NEXT_PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', re.DOTALL)


def _decode_streamed_payload(html: str) -> str:
    chunks = _NEXT_PUSH_RE.findall(html)
    return "".join(chunks).encode("utf-8").decode("unicode_escape", errors="ignore")


def _extract_balanced_json_object(text: str, start_idx: int) -> str | None:
    """Walk from start_idx (which must point at '{') and return the matching
    JSON object as a string. Tolerates strings with embedded braces."""
    if start_idx >= len(text) or text[start_idx] != "{":
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start_idx : i + 1]
    return None


def _collapse_permaslug(slug: str) -> str:
    """Strip OpenRouter's dated version suffix (e.g. `-20260421`) so the
    permaslug collapses to the base model id used in the catalog."""
    return re.sub(r"-2\d{7,9}.*$", "", slug)


# ----------------------------------------------------------------------------
# OpenRouter rankings page — intelligence leaderboard only
# ----------------------------------------------------------------------------
def fetch_intelligence_scores() -> dict[str, float]:
    """Return {slug: intelligence_score} from openrouter.ai/rankings.

    Models in the embedded payload are keyed by `heuristic_openrouter_slug`
    (e.g. `openai/gpt-5.1`) — the same slug the catalog uses.
    """
    log.info("Fetching openrouter.ai/rankings (intelligence)")
    try:
        html = http_get_text("https://openrouter.ai/rankings")
    except Exception as e:
        log.warning("rankings fetch failed: %s", e)
        return {}
    text = _decode_streamed_payload(html)
    out: dict[str, float] = {}

    intel_idx = text.find('"intelligence":[')
    if intel_idx == -1:
        log.info("  intelligence entries: 0")
        return out

    arr_start = text.find("[", intel_idx)
    depth = 0
    in_str = False
    esc = False
    end = -1
    for i in range(arr_start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        log.warning("intelligence array unterminated")
        return out

    try:
        arr = json.loads(text[arr_start:end])
    except json.JSONDecodeError as e:
        log.warning("intelligence parse failed: %s", e)
        return out

    for entry in arr:
        slug = entry.get("heuristic_openrouter_slug") or entry.get("openrouter_slug") or entry.get("uid")
        score = entry.get("score")
        if not slug or score is None:
            continue
        out[slug] = float(score)
    log.info("  intelligence entries: %d", len(out))
    return out


# ----------------------------------------------------------------------------
# Per-agentic-app pages — model usage signal & app's own rank
# ----------------------------------------------------------------------------
@dataclass
class AppSignals:
    slug: str
    rank: int = 0                       # daily global rank
    models_used: int = 0                # distinct models the app uses
    total_tokens: int = 0               # sum across model_tokens
    model_tokens: dict[str, int] = field(default_factory=dict)  # base_slug -> tokens


def fetch_app_signals(app_slug: str) -> AppSignals | None:
    """Scrape https://openrouter.ai/apps/<slug> for per-model usage within
    that app. Returns None on fetch failure."""
    url = f"https://openrouter.ai/apps/{app_slug}"
    log.info("Fetching %s", url)
    try:
        html = http_get_text(url)
    except Exception as e:
        log.warning("app fetch failed for %s: %s", app_slug, e)
        return None
    text = _decode_streamed_payload(html)
    sig = AppSignals(slug=app_slug)

    # The app's own daily-global rank is a sibling of the app object, shaped:
    #   "slug":"hermes-agent"},"totalTokens":N,"rank":1,"modelsUsed":351
    m = re.search(
        rf'"slug":"{re.escape(app_slug)}"\}}\s*,\s*"totalTokens":(\d+)\s*,\s*"rank":(\d+)\s*,\s*"modelsUsed":(\d+)',
        text,
    )
    if m:
        # totalTokens here is the app-level lifetime/30d figure; we still
        # recompute sig.total_tokens from per-model entries below for
        # consistency with model_tokens.
        sig.rank = int(m.group(2))
        sig.models_used = int(m.group(3))

    # Per-model usage entries on the page:
    #   {"model_permaslug":"vendor/model-20260421","total_tokens":N}
    for mm in re.finditer(
        r'"model_permaslug":"([a-z0-9._\-]+/[a-z0-9._\-:]+)"\s*,\s*"total_tokens":(\d+)',
        text,
    ):
        base = _collapse_permaslug(mm.group(1))
        tokens = int(mm.group(2))
        sig.model_tokens[base] = sig.model_tokens.get(base, 0) + tokens

    sig.total_tokens = sum(sig.model_tokens.values())
    log.info(
        "  %s: rank=%d models_in_page=%d total_tokens=%d",
        app_slug, sig.rank, len(sig.model_tokens), sig.total_tokens,
    )
    return sig


def aggregate_app_usage(app_signals: list[AppSignals]) -> dict[str, int]:
    """Sum per-model token counts across all tracked apps to produce a single
    `agentic_token_volume` signal per model base slug."""
    totals: dict[str, int] = {}
    for sig in app_signals:
        for slug, tokens in sig.model_tokens.items():
            totals[slug] = totals.get(slug, 0) + tokens
    return totals


# ----------------------------------------------------------------------------
# Sentiment / news enrichment via one OpenRouter web-enabled LLM call
# ----------------------------------------------------------------------------
def _load_sentiment_prompt() -> str:
    try:
        return SENTIMENT_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        log.error("Sentiment prompt not found at %s", SENTIMENT_PROMPT_PATH)
        raise


def _load_comparative_prompt() -> str:
    try:
        return COMPARATIVE_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.error("Comparative prompt not found at %s", COMPARATIVE_PROMPT_PATH)
        raise


def fetch_comparative_rankings(candidates: list[ModelEntry]) -> dict[str, dict]:
    """Stage-2 head-to-head benchmark comparison.

    Sends a shortlist of already-shortlisted models to a web-enabled LLM and
    asks for relative benchmark-based rankings. Returns
    `{openrouter_slug: {"score": float, "rationale": str}}`. Empty dict on
    failure; the caller should fall back to stage-1 scores in that case.
    """
    if not OPENROUTER_API_KEY:
        log.warning("OPENROUTER_API_KEY missing — skipping comparative re-rank")
        return {}
    if not candidates:
        return {}

    prompt = _load_comparative_prompt()
    lines = []
    for m in candidates:
        hints = []
        if m.intelligence_score > 0:
            hints.append(f"intelligence={m.intelligence_score:.1f}")
        if m.agentic_rank:
            hints.append(f"agentic_rank=#{m.agentic_rank}")
        if abs(m.sentiment_score) > 0.05:
            hints.append(f"sentiment={m.sentiment_score:+.2f}")
        if m.release_date:
            hints.append(f"released={m.release_date.date().isoformat()}")
        if m.is_free:
            hints.append("FREE")
        if not m.supports_tools:
            hints.append("no-tools")
        if not m.supports_reasoning:
            hints.append("no-reasoning")
        hint_str = ", ".join(hints) if hints else "—"
        lines.append(f"- {m.id} | {m.name} | ctx={m.context_length} | {hint_str}")
    full_prompt = prompt + "\n".join(lines) + "\n"

    log.info("Calling %s for comparative re-rank of %d models", SENTIMENT_MODEL, len(candidates))
    body = json.dumps({
        "model": SENTIMENT_MODEL,
        "max_tokens": 4000,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You produce strict JSON only — no prose, no markdown fences."},
            {"role": "user", "content": full_prompt},
        ],
    }).encode("utf-8")
    try:
        raw = _http(
            "https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            data=body,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "X-Title": "automodel-runner",
                "HTTP-Referer": "https://automodel.local",
            },
            timeout=LLM_HTTP_TIMEOUT,
        )
    except Exception as e:
        log.warning("comparative call HTTP failed: %s", e)
        return {}

    # Always cache the raw envelope for debugging — even on parse failure,
    # so we can see why (rate limit, content filter, refusal, etc.).
    try:
        raw_text = raw.decode("utf-8")
    except Exception:
        raw_text = repr(raw)
    (CACHE_DIR / "last_comparative_envelope.json").write_text(raw_text, encoding="utf-8")

    try:
        resp = json.loads(raw_text)
        content = resp["choices"][0]["message"]["content"]
    except Exception as e:
        snippet = raw_text[:400].replace("\n", " ")
        log.warning("comparative parse failed: %s — envelope head: %s", e, snippet)
        return {}

    (CACHE_DIR / "last_comparative_raw.txt").write_text(content, encoding="utf-8")

    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content.strip())
    first_brace = content.find("{")
    if first_brace > 0:
        content = content[first_brace:]

    parsed = _try_parse_json(content)
    if not parsed or not isinstance(parsed.get("rankings"), list):
        log.warning("comparative response wasn't valid JSON; got len=%d", len(content))
        return {}

    out: dict[str, dict] = {}
    for entry in parsed["rankings"]:
        slug = (entry.get("openrouter_slug") or "").strip()
        if not slug:
            continue
        try:
            score = float(entry.get("relative_score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        rationale = (entry.get("rationale") or "").strip()
        out[slug] = {"score": score, "rationale": rationale}

    (CACHE_DIR / "last_comparative.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    log.info("comparative re-rank produced scores for %d/%d candidates", len(out), len(candidates))
    return out


def apply_comparative_scores(
    catalog: dict[str, ModelEntry],
    comparative: dict[str, dict],
) -> int:
    """Apply comparative scores to the catalog (matched by normalized slug).
    Returns the number of models that got a score."""
    by_norm: dict[str, list[ModelEntry]] = {}
    for m in catalog.values():
        by_norm.setdefault(_slug_norm(m.id), []).append(m)

    n = 0
    for slug, info in comparative.items():
        score = info.get("score", 0.0)
        rationale = info.get("rationale", "")
        for m in by_norm.get(_slug_norm(slug), []):
            m.comparative_score = score
            if rationale and not m.comparative_rationale:
                m.comparative_rationale = rationale
            n += 1
    return n


def fetch_sentiment() -> dict:
    if not OPENROUTER_API_KEY:
        log.warning("OPENROUTER_API_KEY missing — skipping sentiment enrichment")
        return {"models": [], "summary": ""}

    prompt = _load_sentiment_prompt()
    log.info("Calling %s for sentiment+news enrichment (prompt: %s)", SENTIMENT_MODEL, SENTIMENT_PROMPT_PATH)
    body = json.dumps({
        "model": SENTIMENT_MODEL,
        "max_tokens": SENTIMENT_MAX_TOKENS,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a meticulous AI-research analyst. Output strict JSON only — no prose, no markdown fences, no citations outside the JSON."},
            {"role": "user", "content": prompt},
        ],
    }).encode("utf-8")
    try:
        raw = _http(
            "https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            data=body,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "X-Title": "automodel-runner",
                "HTTP-Referer": "https://automodel.local",
            },
            timeout=LLM_HTTP_TIMEOUT,
        )
        resp = json.loads(raw.decode("utf-8"))
        content = resp["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning("sentiment call failed: %s", e)
        return {"models": [], "summary": ""}

    # Always cache the raw content for debugging
    (CACHE_DIR / "last_sentiment_raw.txt").write_text(content, encoding="utf-8")

    # Strip ```json fences if the model added them despite instructions
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content.strip())
    # Some online-augmented responses prefix citations like "[1] ..."; strip until first '{'.
    first_brace = content.find("{")
    if first_brace > 0:
        content = content[first_brace:]

    parsed = _try_parse_json(content)
    if parsed is not None:
        return parsed
    log.warning("sentiment response wasn't valid JSON; raw len=%d (saved to cache)", len(content))
    return {"models": [], "summary": ""}


def _try_parse_json(text: str) -> dict | None:
    """Best-effort JSON parser: direct, then balanced-object extraction, then
    trim trailing junk after the last `}`."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start != -1:
        obj = _extract_balanced_json_object(text, start)
        if obj:
            try:
                return json.loads(obj)
            except json.JSONDecodeError:
                pass

    # Try trimming back to the last close-brace.
    last_brace = text.rfind("}")
    if last_brace != -1 and start != -1 and last_brace > start:
        try:
            return json.loads(text[start : last_brace + 1])
        except json.JSONDecodeError:
            pass
    return None


# ----------------------------------------------------------------------------
# Composite ranking
# ----------------------------------------------------------------------------
def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _slug_norm(s: str) -> str:
    """Normalize a slug for matching: lowercase, strip :variant, collapse dots
    and dashes (so `claude-opus-4.7` and `claude-opus-4-7` compare equal)."""
    s = s.strip().lower()
    if ":" in s:
        s = s.split(":", 1)[0]
    s = re.sub(r"[\.\-_]+", "-", s)
    return s


def merge_signals(
    catalog: dict[str, ModelEntry],
    intelligence: dict[str, float],
    agentic_usage: dict[str, int],
    sentiment: dict,
) -> None:
    # Index catalog by normalized base slug → list of model entries.
    by_norm: dict[str, list[ModelEntry]] = {}
    for model in catalog.values():
        by_norm.setdefault(_slug_norm(model.id), []).append(model)

    # Merge intelligence scores
    for slug, score in intelligence.items():
        for model in by_norm.get(_slug_norm(slug), []):
            model.intelligence_score = max(model.intelligence_score, score)

    # Merge agentic-app usage (token volume across tracked apps)
    ranked = sorted(agentic_usage.items(), key=lambda kv: kv[1], reverse=True)
    for i, (slug, tokens) in enumerate(ranked, start=1):
        for model in by_norm.get(_slug_norm(slug), []):
            model.agentic_token_volume = max(model.agentic_token_volume, tokens)
            model.agentic_rank = i if model.agentic_rank == 0 else min(model.agentic_rank, i)

    # Merge sentiment
    for entry in sentiment.get("models", []):
        slug = entry.get("openrouter_slug") or ""
        if not slug:
            continue
        try:
            score = float(entry.get("sentiment_score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        notes = entry.get("notes") or ""
        for model in by_norm.get(_slug_norm(slug), []):
            if abs(score) > abs(model.sentiment_score):
                model.sentiment_score = score
            if notes and not model.sentiment_notes:
                model.sentiment_notes = notes


def _recency_score(release_date: datetime | None, now: datetime) -> float:
    """Map a release date to a 0..1 recency score with a soft decay.

    < 30 days  → ~1.0
    30-180 d   → linear-ish decline from ~0.9 to ~0.4
    > 365 d    → ~0.1 floor
    None       → 0.5 (neutral; don't penalize models with no metadata)
    """
    if release_date is None:
        return 0.5
    age_days = max(0, (now - release_date).days)
    if age_days <= 30:
        return 1.0
    if age_days <= 365:
        # Exponential-ish decay from 1.0 at 30d to ~0.25 at 365d
        return round(max(0.1, 1.0 * (0.9 ** ((age_days - 30) / 45))), 4)
    return 0.1


QUALITY_WEIGHT = 2 / 3
COST_WEIGHT = 1 / 3
# Applied only when ranking/displaying the balanced list. Keeps the
# half-free / half-paid mix from being swept by paid models that beat
# free on the base formula by a slim margin.
BALANCED_FREE_BOOST = 0.10


def _value_score(quality: float, completion_price: float, is_free: bool) -> float:
    """Weighted blend of quality and cost-efficiency. Both components live in
    [0, 1]; quality counts twice as much as cost. No free-model boost is
    applied here — that's list-specific and added at finalize time."""
    out_price_per_mtok = completion_price * 1_000_000
    if is_free or out_price_per_mtok <= 0.001:
        cost_efficiency = 1.0
    else:
        cost_efficiency = 1.0 / (1.0 + out_price_per_mtok / 5.0)
    return round(QUALITY_WEIGHT * quality + COST_WEIGHT * cost_efficiency, 4)


def _value_for_list(m: ModelEntry, ranking_kind: str) -> float:
    """The value score as it should appear (and sort) in a given list.
    Balanced gets a +0.10 free-model boost; other lists use the base value."""
    if ranking_kind == "balanced" and m.is_free:
        return round(m.value_score + BALANCED_FREE_BOOST, 4)
    return m.value_score


# Composite weights. Stage-1 omits `comparative` (it's filled by stage-2);
# the runtime always uses the appropriate dict for each pass.
WEIGHTS_STAGE1 = {
    "intelligence": 0.35,
    "volume":       0.13,
    "sentiment":    0.17,
    "context":      0.10,
    "recency":      0.10,
    "tools":        0.10,
    "reasoning":    0.05,
}
# After stage-2, `comparative` is mixed in and the other weights are scaled
# down proportionally. Sums to 1.0.
WEIGHTS_STAGE2 = {
    "intelligence": 0.20,
    "volume":       0.08,
    "sentiment":    0.10,
    "context":      0.06,
    "recency":      0.07,
    "tools":        0.06,
    "reasoning":    0.03,
    "comparative":  0.40,
}


def compute_scores(catalog: dict[str, ModelEntry], use_comparative: bool = False) -> None:
    """Compute composite quality + value scores.

    Called twice during a run:

    1. After signal merging, before the stage-2 LLM call. Sets
       `preliminary_quality_score` (and `quality_score` to match, so any
       intermediate sort behaves consistently).
    2. After `fetch_comparative_rankings`, with `use_comparative=True`.
       Re-blends with the comparative signal and overwrites
       `quality_score` + `value_score`.
    """
    # Candidates: any model with a signal, OR free tool-capable model with
    # decent context (so the free list never ends up empty just because the
    # rankings page doesn't list free models).
    def is_candidate(m: ModelEntry) -> bool:
        if m.intelligence_score > 0 or m.agentic_token_volume > 0 or m.sentiment_score != 0:
            return True
        if m.is_free and m.supports_tools and m.context_length >= 8000:
            return True
        return False

    candidates = [m for m in catalog.values() if is_candidate(m)]
    if not candidates:
        return

    now = datetime.now(timezone.utc)
    intel_norm = _normalize([m.intelligence_score for m in candidates])
    volume_norm = _normalize([float(m.agentic_token_volume or 0) for m in candidates])
    ctx_norm = _normalize([float(m.context_length or 0) for m in candidates])
    # Sentiment is already -1..+1 → shift to 0..1; neutral (0) becomes 0.5
    sent_norm = [(m.sentiment_score + 1.0) / 2.0 if m.sentiment_score != 0 else 0.5 for m in candidates]
    rec_scores = [_recency_score(m.release_date, now) for m in candidates]
    for m, r in zip(candidates, rec_scores):
        m.recency_score = r

    weights = WEIGHTS_STAGE2 if use_comparative else WEIGHTS_STAGE1

    for i, m in enumerate(candidates):
        tool_bonus = 1.0 if m.supports_tools else 0.0
        reasoning_bonus = 0.5 if m.supports_reasoning else 0.0

        if use_comparative and m.comparative_score > 0:
            # Stage 2 blend — full reweight with the comparative signal.
            quality = (
                weights["intelligence"] * intel_norm[i]
                + weights["volume"]     * volume_norm[i]
                + weights["sentiment"]  * sent_norm[i]
                + weights["context"]    * ctx_norm[i]
                + weights["recency"]    * rec_scores[i]
                + weights["tools"]      * tool_bonus
                + weights["reasoning"]  * reasoning_bonus
                + weights["comparative"] * m.comparative_score
            )
        elif use_comparative:
            # Stage 2 was requested but the LLM didn't score this model —
            # keep its stage-1 quality rather than penalizing it for the
            # LLM's omission. (Without this, comparative_score=0 multiplied
            # by 0.40 would drag the score down by 0.4 in absolute terms.)
            quality = m.preliminary_quality_score
        else:
            # Stage 1 — composite without comparative.
            quality = (
                weights["intelligence"] * intel_norm[i]
                + weights["volume"]     * volume_norm[i]
                + weights["sentiment"]  * sent_norm[i]
                + weights["context"]    * ctx_norm[i]
                + weights["recency"]    * rec_scores[i]
                + weights["tools"]      * tool_bonus
                + weights["reasoning"]  * reasoning_bonus
            )

        m.quality_score = round(quality, 4)
        if not use_comparative:
            m.preliminary_quality_score = m.quality_score
        m.value_score = _value_score(m.quality_score, m.completion_price, m.is_free)


# ----------------------------------------------------------------------------
# List builders
# ----------------------------------------------------------------------------
def _model_card(m: ModelEntry, ranking_kind: str, rank: int) -> dict:
    age_days: int | None = None
    if m.release_date:
        age_days = max(0, (datetime.now(timezone.utc) - m.release_date).days)
    return {
        "rank": rank,
        "id": m.id,
        "name": m.name,
        "is_free": m.is_free,
        "supports_tools": m.supports_tools,
        "supports_reasoning": m.supports_reasoning,
        "context_length": m.context_length,
        "release_date": m.release_date.date().isoformat() if m.release_date else None,
        "age_days": age_days,
        "pricing": {
            "prompt_usd_per_mtok": round(m.prompt_price * 1_000_000, 4),
            "completion_usd_per_mtok": round(m.completion_price * 1_000_000, 4),
        },
        "signals": {
            "intelligence_score": m.intelligence_score,
            "agentic_token_volume": m.agentic_token_volume,
            "agentic_rank": m.agentic_rank,
            "sentiment_score": m.sentiment_score,
            "recency_score": m.recency_score,
            "comparative_score": m.comparative_score,
        },
        "scores": {
            "quality_score": m.quality_score,
            "preliminary_quality_score": m.preliminary_quality_score,
            # Free list ranks on capability alone, so value carries no extra
            # signal there. Balanced list shows the boosted value (matches
            # how it sorts). Best list shows the base value.
            "value_score": 0.0 if ranking_kind == "free" else _value_for_list(m, ranking_kind),
        },
        "notes": m.sentiment_notes,
        "comparative_rationale": m.comparative_rationale,
        "ranking_kind": ranking_kind,
    }


def _shortlist_for_kind(pool: list[ModelEntry], kind: str, size: int) -> list[ModelEntry]:
    """Stage-1 candidate selection per category. Returns up to `size` models
    in stage-1 order — stage-2 may reorder them."""
    if kind == "free":
        free_pool = [m for m in pool if m.is_free]
        free_pool.sort(key=lambda m: (m.preliminary_quality_score, m.agentic_token_volume), reverse=True)
        return free_pool[:size]

    if kind == "balanced":
        # Half free + half paid, picked by their respective natural orders.
        free_pool = sorted(
            [m for m in pool if m.is_free],
            key=lambda m: (m.preliminary_quality_score, m.agentic_token_volume),
            reverse=True,
        )
        paid_pool = sorted(
            [m for m in pool if not m.is_free],
            key=lambda m: (m.value_score, m.preliminary_quality_score),
            reverse=True,
        )
        half = max(1, size // 2)
        chosen: list[ModelEntry] = []
        seen: set[str] = set()
        for m in free_pool:
            if m.id in seen:
                continue
            chosen.append(m); seen.add(m.id)
            if len(chosen) >= half:
                break
        for m in paid_pool:
            if m.id in seen:
                continue
            chosen.append(m); seen.add(m.id)
            if len(chosen) >= size:
                break
        return chosen

    # "best": price-blind, by preliminary quality
    return sorted(pool, key=lambda m: (m.preliminary_quality_score, m.agentic_token_volume), reverse=True)[:size]


def build_shortlists(catalog: dict[str, ModelEntry]) -> dict[str, list[ModelEntry]]:
    """Stage-1: produce SHORTLIST_SIZE candidates per category from
    preliminary-quality scores. Returns ModelEntry objects (not cards) so
    stage-2 can attach comparative scores in place."""
    pool = [m for m in catalog.values() if m.preliminary_quality_score > 0 or m.is_free]
    return {kind: _shortlist_for_kind(pool, kind, SHORTLIST_SIZE) for kind in ("free", "balanced", "best")}


def finalize_lists(shortlists: dict[str, list[ModelEntry]]) -> dict[str, list[dict]]:
    """Stage-2: re-sort each shortlist by the final (post-comparative)
    scores, then trim to LIST_SIZE.

    - `free` ranks on raw capability (quality only) — every entry would
      get the same +0.25 boost, so value carries no extra signal.
    - `balanced` and `best` rank on value, which blends quality with cost
      and is the more useful read for "which should I actually use".
    """
    out: dict[str, list[dict]] = {}
    for kind, candidates in shortlists.items():
        if kind == "free":
            ordered = sorted(candidates, key=lambda m: (m.quality_score, m.agentic_token_volume), reverse=True)
        else:
            # Use the list-specific value for sorting so the balanced
            # +0.10 free boost actually shifts the order.
            ordered = sorted(candidates, key=lambda m: (_value_for_list(m, kind), m.quality_score), reverse=True)
        out[kind] = [_model_card(m, kind, i + 1) for i, m in enumerate(ordered[:LIST_SIZE])]
    return out


# ----------------------------------------------------------------------------
# Output + Telegram
# ----------------------------------------------------------------------------
def write_outputs(
    lists: dict[str, list[dict]],
    sentiment: dict,
    app_signals: list[AppSignals],
    comparative_count: int = 0,
) -> dict[str, Path]:
    now = datetime.now(timezone.utc).isoformat()
    tracked_apps_payload = {
        sig.slug: {
            "rank": sig.rank,
            "models_used": sig.models_used,
            "total_tokens": sig.total_tokens,
        }
        for sig in app_signals
    }
    paths: dict[str, Path] = {}
    for kind in ("free", "balanced", "best"):
        path = OUTPUT_DIR / f"{kind}.json"
        payload = {
            "generated_at": now,
            "kind": kind,
            "list_size": len(lists[kind]),
            "shortlist_size": SHORTLIST_SIZE,
            "comparative_rerank_applied": comparative_count > 0,
            "comparative_scored_count": comparative_count,
            "tracked_apps": tracked_apps_payload,
            "summary": sentiment.get("summary", ""),
            "models": lists[kind],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths[kind] = path
        log.info("wrote %s (%d models)", path, len(lists[kind]))
    # Also keep a sentiment cache for debugging
    (CACHE_DIR / "last_sentiment.json").write_text(json.dumps(sentiment, indent=2), encoding="utf-8")
    return paths


def send_telegram(message: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_HOME_CHANNEL:
        log.warning("Telegram env missing — skipping notification")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    body = urllib.parse.urlencode({
        "chat_id": TELEGRAM_HOME_CHANNEL,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": "true",
    }).encode("utf-8")
    try:
        _http(url, method="POST", data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
        log.info("Telegram notification sent")
    except Exception as e:
        log.warning("Telegram send failed: %s", e)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    started = time.monotonic()
    log.info("==== automodel run start ====")

    try:
        catalog = fetch_catalog()
    except Exception as e:
        log.exception("catalog fetch failed")
        send_telegram(f"❌ *automodel* failed at catalog step: `{e}`")
        return 1

    try:
        intelligence = fetch_intelligence_scores()
    except Exception as e:
        log.warning("intelligence fetch failed: %s", e)
        intelligence = {}

    app_signals: list[AppSignals] = []
    for app_slug in TRACKED_APPS:
        try:
            sig = fetch_app_signals(app_slug)
        except Exception as e:
            log.warning("app fetch failed for %s: %s", app_slug, e)
            continue
        if sig is not None:
            app_signals.append(sig)

    agentic_usage = aggregate_app_usage(app_signals)
    log.info("agentic usage covers %d unique models", len(agentic_usage))

    sentiment = fetch_sentiment()

    merge_signals(catalog, intelligence, agentic_usage, sentiment)
    # Stage 1: preliminary composite score → shortlists per category.
    compute_scores(catalog, use_comparative=False)
    shortlists = build_shortlists(catalog)

    # Stage 2: union the three shortlists and ask an LLM to rank them
    # head-to-head on benchmarks. Re-score the full catalog with the
    # comparative weight blended in, then re-sort each shortlist.
    shortlist_union: dict[str, ModelEntry] = {}
    for cand_list in shortlists.values():
        for m in cand_list:
            shortlist_union.setdefault(m.id, m)
    union_models = list(shortlist_union.values())
    log.info("stage-2 shortlist union: %d unique models across %d lists",
             len(union_models), len(shortlists))

    comparative = fetch_comparative_rankings(union_models)
    comparative_count = apply_comparative_scores(catalog, comparative) if comparative else 0

    if comparative_count:
        compute_scores(catalog, use_comparative=True)
    else:
        log.warning("stage-2 unavailable — falling back to stage-1 scores")

    lists = finalize_lists(shortlists)
    paths = write_outputs(lists, sentiment, app_signals, comparative_count=comparative_count)

    duration = time.monotonic() - started
    head = lists["best"][0] if lists["best"] else None
    head_free = lists["free"][0] if lists["free"] else None
    msg = (
        f"✅ *automodel* updated in {duration:.0f}s\n"
        f"• Best overall: `{head['id']}` (q={head['scores']['quality_score']})\n" if head else ""
    )
    msg += (
        f"• Best free: `{head_free['id']}`\n" if head_free else ""
    )
    if app_signals:
        ordered = sorted(app_signals, key=lambda s: s.rank or 9999)
        msg += "• Tracked apps: " + ", ".join(f"{s.slug}=#{s.rank or '?'}" for s in ordered) + "\n"
    if comparative_count:
        msg += f"• Comparative re-rank: applied to {comparative_count} models\n"
    else:
        msg += "• Comparative re-rank: skipped\n"
    msg += "Files: `" + "`, `".join(p.name for p in paths.values()) + "` in `~/automodel/output/`"

    publish_line = _maybe_publish_to_repo()
    if publish_line:
        msg += "\n" + publish_line

    send_telegram(msg)
    log.info("==== automodel run done in %.1fs ====", duration)
    return 0


def _maybe_publish_to_repo() -> str:
    """Copy fresh JSON into the automodel-repo `site/` dir and push, if configured.

    Gated by `AUTOMODEL_PUBLISH` (default: enabled if AUTOMODEL_REPO_PATH is set,
    or if the runner is running from inside the repo's scripts/ dir). Failures
    are logged but never abort the run — Telegram still gets the success line.
    """
    enabled = os.environ.get("AUTOMODEL_PUBLISH")
    if enabled == "0":
        return ""
    repo_env = os.environ.get("AUTOMODEL_REPO_PATH")
    in_repo = (SCRIPT_DIR.parent / ".git").is_dir()
    default_repo = (HOME / "automodel-repo" / ".git").is_dir()
    if enabled != "1" and not repo_env and not in_repo and not default_repo:
        return ""

    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        import publish_site  # type: ignore
    except Exception as e:
        log.warning("publish: could not import publish_site (%s)", e)
        return f"• Publish: skipped (import failed: `{e}`)"
    finally:
        if sys.path and sys.path[0] == str(SCRIPT_DIR):
            sys.path.pop(0)

    try:
        result = publish_site.publish_to_repo()
    except Exception as e:
        log.exception("publish: unexpected error")
        return f"• Publish: error `{e}`"

    icon = "🌐" if result.ok and result.pushed else ("📝" if result.ok else "⚠️")
    log.info("publish: ok=%s pushed=%s msg=%s", result.ok, result.pushed, result.message)
    return f"{icon} Publish: {result.message}"


if __name__ == "__main__":
    raise SystemExit(main())
