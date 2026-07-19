"""
Deja Vu — reasoning layer.
Takes retrieved evidence from engine.py and uses Groq's free-tier API
to judge: is this genuinely the same decision, what was the original
reasoning, and how confident should we be. The API key is read ONLY
from the environment — never hardcoded, never logged, never returned
in any output.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_MODEL = "llama-3.3-70b-versatile"  # free tier on Groq


def _client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
            "and add it to your local .env file — never paste it in chat or commit it."
        )
    return Groq(api_key=key)


def judge_deja_vu(proposal, candidates):
    """
    proposal: the new idea someone is about to propose
    candidates: ranked list of retrieved past discussion units from engine.DejaVuIndex.search()

    Returns a structured verdict — grounded in the retrieved text only,
    so the model can't hallucinate a "past decision" that isn't there.
    """
    if not candidates:
        return {"is_deja_vu": False, "confidence": "low", "explanation": "No related past discussion found."}

    evidence_block = "\n\n".join(
        f"[Source {i+1}] {c['source_title']} ({c['date'][:10]}, state: {c['state']})\n"
        f"URL: {c['url']}\n"
        f"Text: {c['text'][:600]}"
        for i, c in enumerate(candidates)
    )

    system_prompt = (
        "You are Deja Vu, a decision-memory assistant. You are given a NEW proposal "
        "and REAL excerpts from past discussions. Your job: judge honestly whether the "
        "new proposal re-raises something already substantively discussed or decided. "
        "Only use the provided evidence — never invent a past decision that isn't in it. "
        "If the evidence is only loosely related, say so and keep confidence low. "
        "Respond ONLY in this exact JSON shape, nothing else:\n"
        '{"is_deja_vu": true/false, "confidence": "low/medium/high", '
        '"matched_source": "<Source N or null>", "original_reasoning": "<short summary or null>", '
        '"explanation": "<one or two sentence explanation for a human>"}'
    )

    user_prompt = f"NEW PROPOSAL:\n{proposal}\n\nPAST EVIDENCE:\n{evidence_block}"

    resp = _client().chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )

    raw = resp.choices[0].message.content.strip()
    return _safe_parse(raw, candidates)


def _safe_parse(raw, candidates):
    import json
    try:
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()
        parsed = json.loads(raw)
        # attach real source URL/date for the matched source, don't trust model to reproduce it
        match_label = parsed.get("matched_source")
        if match_label and "Source" in str(match_label):
            try:
                idx = int(str(match_label).split()[-1]) - 1
                parsed["source_url"] = candidates[idx].get("url")
                parsed["source_date"] = candidates[idx].get("date")
                parsed["matched_source_title"] = candidates[idx].get("source_title")
            except (ValueError, IndexError):
                pass
        return parsed
    except json.JSONDecodeError:
        return {
            "is_deja_vu": None,
            "confidence": "low",
            "explanation": "Model response could not be parsed. Raw output logged for debugging.",
            "raw": raw,
        }


def _load_full_thread(source_title, source_url):
    """
    Pull the FULL real thread (all comments) for the matched source from
    the local cache — no re-fetching, no synthetic filler. This gives the
    deep-analysis pass much more real material than the short embedded
    chunk alone.
    """
    import json
    try:
        with open("data/discussions.json") as f:
            threads = json.load(f)
    except FileNotFoundError:
        return None

    for t in threads:
        if t["url"] in source_url or t["title"] == source_title:
            return t
    return None


def deep_analyze(proposal, verdict, candidates):
    """
    Given an initial verdict that found a real match, do a second grounded
    pass over the FULL matched thread's real comments to extract: what
    actually happened (outcome), what real objections were raised (with
    short attributed excerpts), and a recommendation for the new proposal
    that's explicit about being AI-generated judgment, not fact.

    Returns None if there's nothing to deepen (no match, or thread not found).
    """
    if not verdict.get("is_deja_vu") or not verdict.get("source_url"):
        return None

    matched_title = verdict.get("matched_source_title") or ""
    thread = _load_full_thread(matched_title, verdict["source_url"])
    if not thread:
        # fall back to whatever candidate text we already have
        return None

    comments_text = "\n\n".join(
        f"[{c['author']}, {c['created_at'][:10]}]: {c['body'][:500]}"
        for c in thread.get("comments", [])[:40]  # cap for token budget, still real
    )

    system_prompt = (
        "You are analyzing a REAL, closed engineering discussion thread to help a team "
        "avoid re-litigating settled ground. You are given the full real thread and a NEW "
        "proposal that resembles it. Extract ONLY what is actually present in the thread — "
        "never invent objections, outcomes, or people. If the thread doesn't clearly state "
        "an outcome or objections, say so explicitly rather than guessing. "
        "Respond ONLY in this exact JSON shape:\n"
        '{"outcome": "<what actually happened to this real proposal - merged/closed/rejected/stalled, in one sentence, or null if unclear>", '
        '"real_objections": [{"point": "<the objection>", "raised_by": "<username or null>", "date": "<YYYY-MM-DD or null>"}], '
        '"how_new_proposal_differs": "<honest comparison of what is genuinely different about the new proposal vs the old one, or null if essentially identical>", '
        '"open_questions_before_reproposing": ["<specific thing the new proposal would need to address, grounded in the real objections above>"]}'
    )

    user_prompt = (
        f"NEW PROPOSAL:\n{proposal}\n\n"
        f"REAL THREAD: {thread['title']} ({thread['state']}, {thread['url']})\n"
        f"REAL OPENING POST:\n{thread['body'][:1500]}\n\n"
        f"REAL COMMENTS:\n{comments_text}"
    )

    resp = _client().chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )

    raw = resp.choices[0].message.content.strip()
    try:
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()
        import json
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Could not parse deep analysis.", "raw": raw}


if __name__ == "__main__":
    import engine
    idx = engine.DejaVuIndex()
    idx.load()

    proposal = "I think we should let generic parameters have default trait bounds inferred automatically"
    candidates = idx.search(proposal, top_k=5)
    verdict = judge_deja_vu(proposal, candidates)
    import json
    print(json.dumps(verdict, indent=2))
