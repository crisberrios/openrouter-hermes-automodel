# automodel

A Hermes Agent skill + cron-job runner that keeps three ranked OpenRouter model lists
fresh and applies any of them to your Hermes `provider_routing` with one slash command.

> **Live demo:** [openrouter-hermes-automodel.netlify.app](https://openrouter-hermes-automodel.netlify.app)
> — rendered client-side from the same JSON the skill consumes.
> Direct links: [`/free.json`](https://openrouter-hermes-automodel.netlify.app/free.json) ·
> [`/balanced.json`](https://openrouter-hermes-automodel.netlify.app/balanced.json) ·
> [`/best.json`](https://openrouter-hermes-automodel.netlify.app/best.json)
> *(URL is a placeholder — replace with whatever you set in Netlify.)*

The runner pulls signals from:

- **OpenRouter `/api/v1/models`** — catalog, pricing, tool-call support
- **`openrouter.ai/rankings`** — Artificial-Analysis-style intelligence scores
- **`openrouter.ai/apps/<slug>`** — per-app model usage and the app's own rank
  for each tracked agentic app (defaults to `hermes-agent` + `openclaw`); the
  popularity signal is computed from these pages rather than the global "top
  models" leaderboard
- **One web-enabled LLM call** — current news + Reddit/HN/Twitter sentiment for
  agentic LLM use; the prompt lives in [`scripts/sentiment_prompt.md`](scripts/sentiment_prompt.md)
  so you can tune it without editing the runner

Three JSON outputs:

| File           | Selection logic |
|----------------|-----------------|
| `free.json`    | Best quality free models |
| `balanced.json`| Budget-minded mixture of free and paid models |
| `best.json`    | Models with best value per dollar, focusing on quality |

The skill (`/automodel`) reads any of these and patches `~/.hermes/config.yaml`'s
`provider_routing.openrouter.models` so OpenRouter's `/auto` route falls into the
weighted set. `/automodel set default` removes the override (stock auto-routing).

You can use **either** half on its own:

- **Just the cron job** — refresh JSON locally and serve it however you like.
- **Just the skill, pointed at someone else's URL** — `/automodel init` lets you
  fetch lists from any host that exposes `free.json` / `balanced.json` / `best.json`.

## Repository layout

```
automodel-repo/
├── README.md
├── LICENSE
├── SKILL.md                  # Skill manifest (point `hermes skills install` here)
├── netlify.toml              # Netlify publish config (publish = site/)
├── automodel/                # Skill files
│   ├── driver.py
│   └── data/                 # Per-user state (selection.json + source.json)
├── scripts/
│   ├── automodel_runner.py   # Cron-job runner
│   ├── publish_site.py       # Copies fresh JSON into site/ and pushes
│   └── sentiment_prompt.md   # Prompt used for the sentiment+news LLM call
└── site/                     # Netlify publish dir (committed)
    ├── index.html            # Dev demo, renders the JSON client-side
    ├── styles.css
    ├── app.js
    ├── free.json             # Refreshed by the cron job
    ├── balanced.json
    └── best.json
```

## Prerequisites

- Hermes Agent ≥ 0.13
- Python 3.10+
- OpenRouter API key in `~/.hermes/.env` (`OPENROUTER_API_KEY=…`)
- Optional: `TELEGRAM_BOT_TOKEN` + `TELEGRAM_HOME_CHANNEL` in `~/.hermes/.env`
  for run-complete notifications

## Setup paths

### A. Install the skill

```bash
hermes skills install https://raw.githubusercontent.com/crisberrios/openrouter-hermes-automodel/main/SKILL.md
```

or from a local clone:

```bash
hermes skills install /path/to/automodel-repo/SKILL.md
```

Then load it and tell it where to read JSON from:

```
/skill automodel
/automodel init
```

`init` asks one question: **local** (read from `~/automodel/output/` produced by
the cron job) or **url** (fetch from `https://your-host/<selection>.json`).
Scriptable variants:

```bash
python3 ~/.hermes/skills/.../automodel/driver.py init --mode local
python3 ~/.hermes/skills/.../automodel/driver.py init --mode url --url https://example.com/automodel
```

Apply a list:

```
/automodel set best
/automodel set free
/automodel set balanced
/automodel set default          # remove the override, restore stock routing
```

Or rerun the last selection (e.g. after the cron job refreshed the JSON):

```
/automodel apply
```

### B. Set up the cron job (refresh the JSON locally)

```bash
# 1. Copy the runner into the Hermes-managed scripts dir
cp scripts/automodel_runner.py ~/.hermes/scripts/

# 2. Create the cron job — every 6h
hermes cron create "0 */6 * * *" \
  --name automodel-refresh \
  --script automodel_runner.py \
  --no-agent \
  --deliver local

# 3. Optional smoke test
hermes cron run $(hermes cron list | awk '/automodel-refresh/{print $1}')
```

Files land at `~/automodel/output/{free,balanced,best}.json`. Each run also
writes a debug copy of the LLM call to `~/automodel/cache/` and logs to
`~/automodel/logs/automodel.log`.

> Hermes refuses symlinks under `~/.hermes/scripts/` for safety. If you want a
> single source of truth, write a shim like:
> ```python
> # ~/.hermes/scripts/automodel_runner.py
> import runpy, sys
> from pathlib import Path
> target = Path("/path/to/automodel-repo/scripts/automodel_runner.py")
> if not target.is_file():
>     sys.stderr.write(f"runner missing at {target}\n"); sys.exit(2)
> runpy.run_path(str(target), run_name="__main__")
> ```

### C. Both halves

Run the cron job locally (your machine generates the JSON) and `/automodel init`
with `--mode local`. The skill then reads the JSON the runner just wrote, no
network round-trip needed.

## Runner tunables

| Env var | Default | Meaning |
|---------|---------|---------|
| `AUTOMODEL_SENTIMENT_MODEL` | `openai/gpt-5.4:online` | Model used for the sentiment+news enrichment **and** the stage-2 head-to-head comparison call. Any OpenRouter slug; the `:online` suffix activates web search. |
| `AUTOMODEL_SENTIMENT_PROMPT` | `scripts/sentiment_prompt.md` next to the runner | Path to the markdown prompt sent as the user message in the sentiment call. Override to point at a different file. |

Composite scoring (in `compute_scores()`):

- intelligence 0.40 · agentic-app token volume 0.15 · sentiment 0.20 · context 0.10
- tool-call bonus 0.10 · reasoning bonus 0.05
- Free models get a +0.50 boost to `value_score`

The agentic-app volume signal is the sum of `total_tokens` for each model across
all tracked apps' OpenRouter pages (`hermes-agent` + `openclaw` by default).
Each output file also includes a `tracked_apps` block with each app's daily
global rank, `models_used`, and total tokens.

## Hosting JSON on Netlify (recommended)

`site/` is a self-contained Netlify deploy: a dev-oriented HTML demo that fetches
`free.json` / `balanced.json` / `best.json` from the same origin and renders them
as ranked tables. The cron job keeps those JSON files fresh by committing them
into the repo, which triggers a Netlify rebuild.

**1. Hook the repo up to Netlify (one-time)**

In the Netlify UI, "Add new site → Import from Git", pick this repo, and accept
the defaults — `netlify.toml` already sets `publish = "site"` with no build
command. The site URL it gives you (e.g. `your-slug.netlify.app`) is what you
share with other devs.

**2. Wire the cron job to publish (one-time)**

`scripts/publish_site.py` copies the JSON from `~/automodel/output/` into
`<repo>/site/`, commits, and pushes. The runner calls it automatically at the
end of each run if it can find a repo. Resolution order:

1. `$AUTOMODEL_REPO_PATH` if set
2. the parent of `scripts/automodel_runner.py` if it's inside a git repo
3. `~/automodel-repo` if it exists and is a git repo

To opt out, set `AUTOMODEL_PUBLISH=0`. To force-enable when none of the above
apply, set `AUTOMODEL_PUBLISH=1` and `AUTOMODEL_REPO_PATH=/path/to/repo`.

Cron-time prerequisites inside the WSL/Linux environment that runs the job:

- `git config --global user.email` and `user.name` set for the cron user
- `git push` from `~/automodel-repo` works non-interactively (passphraseless SSH
  key, a credential helper, or a deploy key with write access)

After each run, the Telegram notification appends one of:

- `🌐 Publish: pushed <sha>` — new commit landed on `main`, Netlify will rebuild
- `📝 Publish: no changes (site/ already current; copied 3)` — JSON unchanged
- `⚠️ Publish: …` — copy/commit/push failed; the run itself still succeeded

**3. Pointing the skill at your Netlify URL**

Other developers can install the skill and point it at your deploy:

```
/automodel init --mode url --url https://your-slug.netlify.app
```

The driver also accepts the legacy `<selection>-models.json` naming as a fallback,
but the cron job writes the short names by default.

### Alternative: serve JSON locally

If you'd rather not use Netlify, expose `~/automodel/output/` over any static
HTTP server, e.g. `cd ~/automodel/output && python3 -m http.server 8000`, then
point the skill at `http://your-host:8000`.

## License

MIT — see [LICENSE](./LICENSE).
