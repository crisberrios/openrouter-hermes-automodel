# automodel

A Hermes Agent skill + cron-job runner that keeps three ranked OpenRouter model lists
fresh and applies any of them to your Hermes `provider_routing` with one slash command.

The runner pulls signals from:

- **OpenRouter `/api/v1/models`** — catalog, pricing, tool-call support
- **`openrouter.ai/rankings`** — Artificial-Analysis-style intelligence scores and weekly token volume
- **OpenRouter app pages** — confirms the agentic apps you track (defaults to `hermes-agent` + `openclaw`)
- **One web-enabled LLM call** — current news + Reddit/HN/Twitter sentiment for agentic LLM use

Three JSON outputs:

| File           | Selection logic |
|----------------|-----------------|
| `free.json`    | Top 10 free models with tool calling |
| `balanced.json`| Top 5 free + top 5 paid, ranked by quality-per-dollar |
| `best.json`    | Top 10 by quality, ignoring price |

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
├── automodel/                # Skill files
│   ├── driver.py
│   └── data/                 # Per-user state (selection.json + source.json)
└── scripts/
    └── automodel_runner.py   # Cron-job runner
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
hermes skills install https://raw.githubusercontent.com/<you>/automodel-repo/main/SKILL.md
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
| `AUTOMODEL_SENTIMENT_MODEL` | `openai/gpt-4o-mini:online` | Model used for the news-and-sentiment enrichment call. Any OpenRouter slug; the `:online` suffix activates web search. |

Composite scoring (in `compute_scores()`):

- intelligence 0.40 · weekly volume 0.15 · sentiment 0.20 · context 0.10
- tool-call bonus 0.10 · reasoning bonus 0.05
- Free models get a +0.50 boost to `value_score`

## Hosting JSON on your own server

If you want others to point their skills at your output, expose
`~/automodel/output/` over any static HTTP server, e.g.

```bash
cd ~/automodel/output && python3 -m http.server 8000
```

Then other developers run `/automodel init --mode url --url http://your-host:8000`.
The driver also accepts the legacy `<selection>-models.json` naming as a fallback,
but the cron job writes the short names by default.

## License

MIT — see [LICENSE](./LICENSE).
