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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LIST_SIZE = 10
TRACKED_APPS = ["hermes-agent", "openclaw"]

USER_AGENT = "automodel-runner/1.0 (+https://openrouter.ai)"
HTTP_TIMEOUT = 30

# LLM used for the news + sentiment enrichment call. `:online` enables
# OpenRouter's built-in web plugin so the model can actually search the web.
SENTIMENT_MODEL = os.environ.get("AUTOMODEL_SENTIMENT_MODEL", "openai/gpt-4o-mini:online")
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
def _http(url: str, method: str = "GET", data: bytes | None = None, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, method=method, data=data)
    req.add_header("User-Agent", USER_AGENT)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
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

    # Signals (filled later)
    intelligence_score: float = 0.0
    # Aggregate usage across tracked agentic apps (TRACKED_APPS) — replaces
    # the previous global weekly leaderboard signal.
    agentic_token_volume: int = 0
    agentic_rank: int = 0
    sentiment_score: float = 0.0   # -1..+1
    sentiment_notes: str = ""

    # Composite (filled at ranking time)
    quality_score: float = 0.0
    value_score: float = 0.0       # quality per dollar


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
        )
    log.info("  catalog: %d models", len(models))
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


def compute_scores(catalog: dict[str, ModelEntry]) -> None:
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

    intel_norm = _normalize([m.intelligence_score for m in candidates])
    volume_norm = _normalize([float(m.agentic_token_volume or 0) for m in candidates])
    ctx_norm = _normalize([float(m.context_length or 0) for m in candidates])
    # Sentiment is already -1..+1 → shift to 0..1; neutral (0) becomes 0.5
    sent_norm = [(m.sentiment_score + 1.0) / 2.0 if m.sentiment_score != 0 else 0.5 for m in candidates]

    # Bonus for tool-calling support — required for agentic work
    for i, m in enumerate(candidates):
        tool_bonus = 1.0 if m.supports_tools else 0.0
        reasoning_bonus = 0.5 if m.supports_reasoning else 0.0
        # Weights: intelligence 0.40, agentic-app usage 0.15, sentiment 0.20,
        # context 0.10, tool-call 0.10, reasoning 0.05.
        m.quality_score = round(
            0.40 * intel_norm[i]
            + 0.15 * volume_norm[i]
            + 0.20 * sent_norm[i]
            + 0.10 * ctx_norm[i]
            + 0.10 * tool_bonus
            + 0.05 * reasoning_bonus,
            4,
        )

        # Value = quality per million-output-tokens cost. Free models get
        # the highest value bucket because cost is zero.
        out_price_per_mtok = m.completion_price * 1_000_000
        if m.is_free or out_price_per_mtok <= 0.001:
            m.value_score = round(m.quality_score + 0.50, 4)  # large boost
        else:
            m.value_score = round(m.quality_score / (1.0 + (out_price_per_mtok / 5.0)), 4)


# ----------------------------------------------------------------------------
# List builders
# ----------------------------------------------------------------------------
def _model_card(m: ModelEntry, ranking_kind: str, rank: int) -> dict:
    return {
        "rank": rank,
        "id": m.id,
        "name": m.name,
        "is_free": m.is_free,
        "supports_tools": m.supports_tools,
        "supports_reasoning": m.supports_reasoning,
        "context_length": m.context_length,
        "pricing": {
            "prompt_usd_per_mtok": round(m.prompt_price * 1_000_000, 4),
            "completion_usd_per_mtok": round(m.completion_price * 1_000_000, 4),
        },
        "signals": {
            "intelligence_score": m.intelligence_score,
            "agentic_token_volume": m.agentic_token_volume,
            "agentic_rank": m.agentic_rank,
            "sentiment_score": m.sentiment_score,
        },
        "scores": {
            "quality_score": m.quality_score,
            "value_score": m.value_score,
        },
        "notes": m.sentiment_notes,
        "ranking_kind": ranking_kind,
    }


def build_lists(catalog: dict[str, ModelEntry]) -> dict[str, list[dict]]:
    # compute_scores() already filtered to candidates; quality_score == 0 still
    # appears for free models that have no leaderboard signal but were
    # admitted via the tool-call + context heuristic.
    pool = [m for m in catalog.values() if m.quality_score > 0 or m.is_free]

    # ---- Free list ----
    free_pool = [m for m in pool if m.is_free]
    free_pool.sort(key=lambda m: (m.quality_score, m.agentic_token_volume), reverse=True)
    free_list = [_model_card(m, "free", i + 1) for i, m in enumerate(free_pool[:LIST_SIZE])]

    # ---- Balanced list: top 5 free + top 5 paid by value_score ----
    paid_pool = sorted(
        [m for m in pool if not m.is_free],
        key=lambda m: (m.value_score, m.quality_score),
        reverse=True,
    )
    seen: set[str] = set()
    balanced: list[ModelEntry] = []
    for m in free_pool:
        if m.id in seen:
            continue
        balanced.append(m)
        seen.add(m.id)
        if len(balanced) >= LIST_SIZE // 2:
            break
    for m in paid_pool:
        if m.id in seen:
            continue
        balanced.append(m)
        seen.add(m.id)
        if len(balanced) >= LIST_SIZE:
            break
    balanced.sort(key=lambda m: (m.quality_score, m.value_score), reverse=True)
    balanced_list = [_model_card(m, "balanced", i + 1) for i, m in enumerate(balanced[:LIST_SIZE])]

    # ---- Best list: top quality, no price filter ----
    best_pool = sorted(pool, key=lambda m: (m.quality_score, m.agentic_token_volume), reverse=True)
    best_list = [_model_card(m, "best", i + 1) for i, m in enumerate(best_pool[:LIST_SIZE])]

    return {"free": free_list, "balanced": balanced_list, "best": best_list}


# ----------------------------------------------------------------------------
# Output + Telegram
# ----------------------------------------------------------------------------
def write_outputs(
    lists: dict[str, list[dict]],
    sentiment: dict,
    app_signals: list[AppSignals],
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
    compute_scores(catalog)
    lists = build_lists(catalog)
    paths = write_outputs(lists, sentiment, app_signals)

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
