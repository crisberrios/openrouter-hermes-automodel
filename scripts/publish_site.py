#!/usr/bin/env python3
"""
publish_site.py — copy fresh automodel JSON into the repo's `site/` dir
and commit + push so Netlify rebuilds.

Resolves the repo path in this order:
  1. $AUTOMODEL_REPO_PATH
  2. the parent of this script's directory if it looks like a git repo
  3. ~/automodel-repo

Source JSON dir defaults to ~/automodel/output/ (overridable via
$AUTOMODEL_OUTPUT_DIR).

Designed to be called from the cron runner; safe to invoke by hand:

  python3 scripts/publish_site.py

Exits 0 on success or when there is nothing to commit, non-zero on
unexpected failure. Git command failures are reported via the return
value of `publish_to_repo()` so the caller can log them without aborting.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("automodel.publish")

JSON_NAMES = ("free.json", "balanced.json", "best.json")
SITE_DIRNAME = "site"


@dataclass
class PublishResult:
    ok: bool
    pushed: bool
    message: str
    commit: str | None = None


def _resolve_repo() -> Path | None:
    env = os.environ.get("AUTOMODEL_REPO_PATH")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / ".git").is_dir():
            return p
        log.warning("AUTOMODEL_REPO_PATH=%s is not a git repo", p)
        return None

    here = Path(__file__).resolve().parent.parent
    if (here / ".git").is_dir():
        return here

    fallback = Path.home() / "automodel-repo"
    if (fallback / ".git").is_dir():
        return fallback

    return None


def _resolve_source() -> Path:
    env = os.environ.get("AUTOMODEL_OUTPUT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / "automodel" / "output"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def publish_to_repo(
    repo: Path | None = None,
    source_dir: Path | None = None,
    push: bool | None = None,
) -> PublishResult:
    """Copy the three JSON files into <repo>/site/, commit, and push.

    Returns a `PublishResult`; never raises for normal git/copy failures so
    the caller can log and continue.
    """
    repo = repo or _resolve_repo()
    if repo is None:
        return PublishResult(False, False, "no automodel-repo found (set AUTOMODEL_REPO_PATH)")

    source_dir = source_dir or _resolve_source()
    if not source_dir.is_dir():
        return PublishResult(False, False, f"source dir missing: {source_dir}")

    if push is None:
        push = os.environ.get("AUTOMODEL_PUBLISH_PUSH", "1") != "0"

    site = repo / SITE_DIRNAME
    site.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []
    for name in JSON_NAMES:
        src = source_dir / name
        if not src.is_file():
            missing.append(name)
            continue
        shutil.copy2(src, site / name)
        copied.append(name)

    if not copied:
        return PublishResult(False, False, f"no JSON files in {source_dir} (missing: {', '.join(missing)})")

    status = _git(repo, "status", "--porcelain", "--", *(f"{SITE_DIRNAME}/{n}" for n in copied))
    if status.returncode != 0:
        return PublishResult(False, False, f"git status failed: {status.stderr.strip()}")

    if not status.stdout.strip():
        return PublishResult(True, False, f"no changes (site/ already current; copied {len(copied)})")

    add = _git(repo, "add", *(f"{SITE_DIRNAME}/{n}" for n in copied))
    if add.returncode != 0:
        return PublishResult(False, False, f"git add failed: {add.stderr.strip()}")

    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    msg = f"data: refresh model lists ({stamp})"
    if missing:
        msg += f"\n\nmissing source files: {', '.join(missing)}"

    commit = _git(repo, "commit", "-m", msg)
    if commit.returncode != 0:
        return PublishResult(False, False, f"git commit failed: {commit.stderr.strip()}")

    rev = _git(repo, "rev-parse", "--short", "HEAD")
    sha = rev.stdout.strip() if rev.returncode == 0 else None

    if not push:
        return PublishResult(True, False, f"committed {sha} (push disabled)", commit=sha)

    pushed = _git(repo, "push")
    if pushed.returncode != 0:
        return PublishResult(
            False, False,
            f"git push failed for {sha}: {pushed.stderr.strip()}",
            commit=sha,
        )

    return PublishResult(True, True, f"pushed {sha}", commit=sha)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = publish_to_repo()
    if result.ok:
        log.info("publish: %s", result.message)
        return 0
    log.error("publish failed: %s", result.message)
    return 1


if __name__ == "__main__":
    sys.exit(main())
