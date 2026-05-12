#!/usr/bin/env python3
"""automodel driver — apply curated OpenRouter model lists to Hermes config.

Subcommands
-----------
  init                              Configure where the JSON lists come from.
                                    Interactive when run with no flags;
                                    accepts `--mode local|url --url …` for
                                    scripted setup.

  set <free|balanced|best|default>  Set the active selection and apply it.
                                    `default` removes the openrouter routing
                                    override (restores stock /auto routing).

  apply                             Re-apply the currently saved selection
                                    using the currently saved source.

Sources
-------
  local  — read JSON from ~/automodel/output/ (the cron job's output dir,
           or any local folder if overridden in source.json).
  url    — HTTP GET <base_url>/<selection>.json (with a fallback to
           <base_url>/<selection>-models.json for legacy layouts).

State files
-----------
  data/selection.json   {"selected": "free|balanced|best|default"}
  data/source.json      {"mode": "local"|"url",
                         "base_url": "https://…",       (url mode only)
                         "local_path": "/abs/path"}    (local mode only,
                                                        optional)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERMES_HOME = Path(os.path.expanduser("~/.hermes"))
CONFIG_PATH = HERMES_HOME / "config.yaml"
SKILL_DIR = Path(__file__).resolve().parent
DATA_DIR = SKILL_DIR / "data"
SELECTION_PATH = DATA_DIR / "selection.json"
SOURCE_PATH = DATA_DIR / "source.json"
DEFAULT_LOCAL_PATH = Path.home() / "automodel" / "output"

LIST_SELECTIONS = ("free", "balanced", "best")
VALID_SELECTIONS = LIST_SELECTIONS + ("default",)

USER_AGENT = "automodel-driver/2.0"
HTTP_TIMEOUT = 20

DEFAULT_BASE_URL = "https://openrouter-hermes-automodel.netlify.app/"


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def _load_yaml():
    try:
        import yaml  # type: ignore
    except ImportError:
        sys.stderr.write(
            "ERROR: PyYAML not installed in the active interpreter.\n"
            "Install it: `python3 -m pip install --user 'PyYAML==6.*'` and retry.\n"
        )
        sys.exit(3)
    return yaml


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------
# Source resolution (local folder vs remote URL)
# ----------------------------------------------------------------------------
def load_source() -> dict:
    if not SOURCE_PATH.exists():
        # First-run default: read from the cron job's local output folder.
        return {"mode": "local"}
    try:
        cfg = _read_json(SOURCE_PATH)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"source.json invalid ({e}); falling back to 'local'.\n")
        return {"mode": "local"}
    if cfg.get("mode") not in ("local", "url"):
        sys.stderr.write("source.json missing valid 'mode'; falling back to 'local'.\n")
        return {"mode": "local"}
    return cfg


def fetch_models(selection: str, source: dict) -> list[dict]:
    """Return the `models` array for the given selection from the configured
    source. Tries both <sel>.json and <sel>-models.json (in that order)."""
    if selection not in LIST_SELECTIONS:
        raise ValueError(f"selection {selection!r} is not a list-backed selection")

    candidates = (f"{selection}.json", f"{selection}-models.json")

    if source["mode"] == "url":
        base = source.get("base_url", "").rstrip("/")
        if not base:
            sys.stderr.write("URL mode configured but base_url is empty.\n")
            sys.exit(2)
        last_err: Exception | None = None
        for filename in candidates:
            url = f"{base}/{filename}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                models = payload.get("models") or []
                if not models:
                    raise ValueError(f"{url} contained no models")
                return models
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    last_err = e
                    break
                last_err = e
                continue
            except Exception as e:
                last_err = e
                break
        sys.stderr.write(
            f"ERROR: could not load {selection!r} from {base!r}: {last_err}\n"
            f"Tried: {', '.join(candidates)}\n"
        )
        sys.exit(2)

    # local mode
    local_root = Path(source.get("local_path") or DEFAULT_LOCAL_PATH).expanduser()
    for filename in candidates:
        path = local_root / filename
        if path.is_file():
            data = _read_json(path)
            models = data.get("models") or []
            if not models:
                sys.stderr.write(f"ERROR: {path} contained no models.\n")
                sys.exit(2)
            return models
    sys.stderr.write(
        f"ERROR: no list file found in {local_root}.\n"
        f"Looked for: {', '.join(candidates)}\n"
        "Run the cron job (or `python3 ~/automodel/scripts/automodel_runner.py`) first.\n"
    )
    sys.exit(2)


# ----------------------------------------------------------------------------
# config.yaml mutations
# ----------------------------------------------------------------------------
def backup_config() -> Path | None:
    if not CONFIG_PATH.exists():
        print("config.yaml not found, skipping backup.")
        return None
    backup = CONFIG_PATH.with_suffix(
        CONFIG_PATH.suffix + ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    shutil.copy2(CONFIG_PATH, backup)
    print(f"Backed up config.yaml -> {backup}")
    return backup


def build_routing(models: list[dict]) -> dict:
    """Top-ranked model gets the highest weight; weights decay linearly."""
    n = len(models)
    return {
        "openrouter": {
            "models": [
                {"model": m["id"], "weight": n - i} for i, m in enumerate(models)
            ]
        }
    }


def apply_routing(routing: dict) -> None:
    yaml = _load_yaml()
    config: dict = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp) or {}

    config.setdefault("provider_routing", {})
    config["provider_routing"]["openrouter"] = routing["openrouter"]

    with CONFIG_PATH.open("w", encoding="utf-8") as fp:
        yaml.dump(config, fp, sort_keys=False)

    print(f"Patched provider_routing.openrouter with {len(routing['openrouter']['models'])} models.")
    for entry in routing["openrouter"]["models"]:
        print(f"  weight={entry['weight']:>2}  {entry['model']}")


def restore_default() -> None:
    """Remove provider_routing.openrouter (restore stock auto-routing)."""
    yaml = _load_yaml()
    if not CONFIG_PATH.exists():
        print(f"{CONFIG_PATH} does not exist — nothing to restore.")
        return
    with CONFIG_PATH.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}

    routing = config.get("provider_routing")
    if not isinstance(routing, dict) or "openrouter" not in routing:
        print("provider_routing.openrouter not set — already at default. No changes written.")
        return

    routing.pop("openrouter", None)
    if not routing:
        config.pop("provider_routing", None)
        msg = "Removed provider_routing entirely (was empty after dropping openrouter)."
    else:
        config["provider_routing"] = routing
        msg = "Removed provider_routing.openrouter; other provider_routing entries preserved."

    with CONFIG_PATH.open("w", encoding="utf-8") as fp:
        yaml.dump(config, fp, sort_keys=False)
    print(msg)


# ----------------------------------------------------------------------------
# Subcommand: init
# ----------------------------------------------------------------------------
def _normalize_url(raw: str) -> str:
    url = raw.strip()
    if not url:
        raise ValueError("URL is empty.")
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://.")
    return url.rstrip("/")


def cmd_init(args: argparse.Namespace) -> int:
    mode = args.mode
    base_url = args.url
    local_path = args.local_path

    # Interactive default: always prompt for the base URL (URL mode).
    # Scripted callers can still pick local mode explicitly via --mode local.
    if not mode:
        mode = "url"
        if not base_url:
            entered = input(f"Base URL for JSON lists [{DEFAULT_BASE_URL}]: ").strip()
            base_url = entered or DEFAULT_BASE_URL

    cfg: dict[str, Any] = {"mode": mode, "configured_at": _utcnow_iso()}

    if mode == "url":
        if not base_url:
            base_url = DEFAULT_BASE_URL
        try:
            cfg["base_url"] = _normalize_url(base_url)
        except ValueError as e:
            sys.stderr.write(f"ERROR: {e}\n")
            return 2
    elif mode == "local":
        if local_path:
            cfg["local_path"] = str(Path(local_path).expanduser().resolve())
    else:
        sys.stderr.write(f"ERROR: --mode must be 'local' or 'url', got {mode!r}.\n")
        return 2

    _write_json(SOURCE_PATH, cfg)
    print(f"Wrote {SOURCE_PATH}:")
    print(json.dumps(cfg, indent=2))
    if mode == "url":
        print(
            "\nNext: `/automodel set best` (or free|balanced|default) — the driver will "
            f"fetch JSON from {cfg['base_url']}/<selection>.json on demand."
        )
    else:
        target = cfg.get("local_path") or str(DEFAULT_LOCAL_PATH)
        print(
            f"\nNext: ensure the cron job is producing files under {target}, then "
            "`/automodel set best`."
        )
    return 0


# ----------------------------------------------------------------------------
# Subcommand: set / apply
# ----------------------------------------------------------------------------
def _save_selection(selection: str) -> None:
    payload = {"selected": selection, "selected_at": _utcnow_iso()}
    _write_json(SELECTION_PATH, payload)


def _load_selection() -> str:
    if not SELECTION_PATH.exists():
        return "best"
    try:
        data = _read_json(SELECTION_PATH)
    except json.JSONDecodeError:
        return "best"
    sel = (data.get("selected") or "best").strip().lower()
    return sel if sel in VALID_SELECTIONS else "best"


def _apply(selection: str) -> int:
    print(f"Applying automodel selection: {selection}")
    backup_config()
    if selection == "default":
        restore_default()
    else:
        source = load_source()
        print(f"Source: {source}")
        models = fetch_models(selection, source)
        apply_routing(build_routing(models))
    print("\nDone. Restart the gateway or `/reset` your CLI session to pick up the new routing.")
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    selection = args.selection.strip().lower()
    if selection not in VALID_SELECTIONS:
        sys.stderr.write(
            f"ERROR: selection must be one of {VALID_SELECTIONS}, got {selection!r}.\n"
        )
        return 2
    _save_selection(selection)
    return _apply(selection)


def cmd_apply(args: argparse.Namespace) -> int:
    return _apply(_load_selection())


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="automodel driver", description=__doc__)
    sub = p.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Configure JSON source (local folder or URL).")
    p_init.add_argument("--mode", choices=("local", "url"), help="Skip the interactive prompt.")
    p_init.add_argument("--url", help="Base URL when --mode url.")
    p_init.add_argument("--local-path", help="Override the local output folder (default ~/automodel/output).")
    p_init.set_defaults(func=cmd_init)

    p_set = sub.add_parser("set", help="Save the selection and apply it.")
    p_set.add_argument("selection", choices=list(VALID_SELECTIONS))
    p_set.set_defaults(func=cmd_set)

    p_apply = sub.add_parser("apply", help="Re-apply the saved selection without changing it.")
    p_apply.set_defaults(func=cmd_apply)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        # Backwards compatibility: no subcommand → apply current selection.
        return cmd_apply(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
