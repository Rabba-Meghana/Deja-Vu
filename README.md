# Déjà Vu — Decision Memory for Teams

Checks whether a new proposal, argument, or idea has already been substantively
discussed or decided before — using real indexed discussion history, semantic
search, and an LLM reasoning layer that judges *is this actually the same
decision*, not just keyword overlap.

Built as an MCP (Model Context Protocol) server, with a local dashboard for
direct use outside a chat client.

**100% free to run.** No paid APIs required for the core engine. The only
external service (Groq, for the reasoning layer) has a genuinely free tier —
no credit card needed.

---

## How it actually works

Most "we already discussed this" tools do one of two shallow things: keyword
search (misses paraphrases) or naive similarity threshold (can't tell a real
decision from a passing mention). This does neither.

**1. Ingestion** (`ingest.py`) — pulls real discussion threads (issues + all
their comments) from a GitHub repo via the free public REST API. No synthetic
data anywhere.

**2. Decision extraction** (`engine.py`) — splits threads into individual
"decision units" and scores each with a *decision-confidence* heuristic based
on general language patterns ("we decided," "closing as," "final comment
period" vs. "what if," "just thinking," "not sure"). This isn't hardcoded to
any company or domain — it's how humans phrase settled vs. tentative
statements in any discussion.

**3. Embedding index** (`engine.py`) — every decision unit is embedded with
`sentence-transformers` (`all-MiniLM-L6-v2`), a free, local, CPU-friendly
model. No API cost, no rate limit, runs entirely on your machine.

**4. Retrieval** — a new proposal is embedded and compared via cosine
similarity, combined with decision-confidence and a recency-decay weight (old
"decisions" on since-rewritten systems shouldn't carry the same weight as
last month's).

**5. Reasoning** (`reasoning.py`) — the retrieved evidence (never anything
else) is handed to Groq's free-tier LLM API, which judges whether this is
genuinely the same decision, reconstructs the original reasoning, and scores
its own confidence. The model is instructed to only use the provided evidence
— it can't invent a past decision that isn't in the retrieved text.

**6. Exposure** — as MCP tools (`server.py`) for any MCP client, and as a
local dashboard (`app.py` + `dashboard/index.html`) for direct use.

---

## Setup (all free)

```bash
git clone <this-repo>
cd deja-vu
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:
- `GROQ_API_KEY` — free, no credit card, get one at https://console.groq.com/keys
- `GITHUB_TOKEN` — optional, raises the GitHub API rate limit from 60/hr to
  5000/hr (still free). Create at https://github.com/settings/tokens with no
  scopes needed for public repos.

**Never commit `.env` or paste your keys anywhere outside it.** `.gitignore`
already excludes it.

### 1. Pull real discussion data

```bash
python ingest.py rust-lang/rfcs 60
```

Swap `rust-lang/rfcs` for any public repo — nothing is hardcoded to this one.
This example is included because it's a real, public archive of debated and
often-reversed engineering decisions, which makes it a strong proving ground.

### 2. Build the index

```bash
python engine.py
```

### 3a. Run the dashboard

```bash
python app.py
```

Open http://localhost:5050 — type a proposal, see if it's already been argued
out, with a link to the real source.

### 3b. Or run the MCP server

```bash
python server.py
```

Point your MCP client (e.g. Claude Desktop) at this via stdio. Exposes two
tools: `check_for_deja_vu` and `search_past_discussions`.

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

Tests cover the deterministic logic (decision-confidence scoring, recency
decay, JSON parsing safety) — no API key or network access required to run
them.

---

## Security notes

- API keys are read only from environment variables (via `.env`, never
  hardcoded, never logged, never echoed back in any tool output or error
  message).
- `.env` is git-ignored from the first commit.
- `.env.example` shows the required shape with placeholder values only.
- If you ever paste a real key somewhere it could be exposed (chat, a public
  gist, a commit), treat it as compromised and rotate it immediately.

---

## What's deliberately out of scope for this build

- Live meeting-transcript ingestion (Zoom/Meet transcription APIs generally
  require a paid tier or a self-hosted bot — noted here so it's not a silent
  gap)
- Multi-repo / cross-org indexing (the schema supports it; the CLI doesn't
  wire it up yet)
- A hosted version (this runs entirely on localhost by design — zero hosting
  cost, and it also means sensitive internal discussions never leave your
  machine unless you choose to deploy it somewhere)
