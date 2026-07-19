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
                parsed["source_url"] = candidates[idx]["url"]
                parsed["source_date"] = candidates[idx]["date"]
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


if __name__ == "__main__":
    import engine
    idx = engine.DejaVuIndex()
    idx.load()

    proposal = "I think we should let generic parameters have default trait bounds inferred automatically"
    candidates = idx.search(proposal, top_k=5)
    verdict = judge_deja_vu(proposal, candidates)
    import json
    print(json.dumps(verdict, indent=2))
