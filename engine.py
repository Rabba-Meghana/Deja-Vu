"""
Deja Vu — retrieval engine.
Turns real discussion threads into searchable "decision units", embeds
them with a FREE local model (no API cost, runs on CPU), and finds
semantic matches for a new proposal. No hardcoded keywords, no
domain-specific rules — everything is learned from whatever real data
you point it at.
"""

import json
import re
import numpy as np
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"  # free, local, ~90MB, downloads once
_model = None

# General decision-language patterns (not company/domain specific —
# these are how humans phrase settled vs. tentative statements in any
# discussion, in any org).
_DECIDED_PATTERNS = [
    r"\bwe(?:'ve| have)? decided\b", r"\bgoing with\b", r"\bfinal decision\b",
    r"\bclosing (?:this )?(?:as|in favor)\b", r"\brejected because\b",
    r"\baccepted\b.{0,20}\breasoning\b", r"\bconsensus\b", r"\bwon't fix\b",
    r"\bmerged\b", r"\bfinal comment period\b", r"\bmotion to (?:accept|close)\b",
]
_TENTATIVE_PATTERNS = [
    r"\bwhat if\b", r"\bjust thinking\b", r"\bmaybe we could\b",
    r"\bnot sure\b", r"\bopen question\b", r"\bcurious\b",
]


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _decision_confidence(text):
    """Heuristic score 0-1: does this text sound like a settled decision
    or a passing mention? General language patterns, not hardcoded topics."""
    text_l = text.lower()
    decided_hits = sum(1 for p in _DECIDED_PATTERNS if re.search(p, text_l))
    tentative_hits = sum(1 for p in _TENTATIVE_PATTERNS if re.search(p, text_l))
    score = 0.3 + 0.15 * decided_hits - 0.1 * tentative_hits
    return max(0.0, min(1.0, score))


def _recency_weight(date_str, half_life_days=730):
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return 0.5
    age_days = (datetime.now(timezone.utc) - dt).days
    return 0.5 ** (age_days / half_life_days)


def build_decision_units(threads):
    """
    Flattens raw threads into discrete searchable units: the opening
    post of each thread, plus each substantive comment. Each unit
    carries its own decision-confidence and recency weight.
    """
    units = []
    for t in threads:
        if t["body"] and len(t["body"]) > 40:
            units.append({
                "text": t["body"][:2000],
                "source_title": t["title"],
                "url": t["url"],
                "date": t["created_at"],
                "decision_confidence": _decision_confidence(t["body"]),
                "recency_weight": _recency_weight(t["created_at"]),
                "state": t["state"],
            })
        for c in t.get("comments", []):
            if len(c["body"]) > 60:
                units.append({
                    "text": c["body"][:2000],
                    "source_title": t["title"],
                    "url": t["url"] + f"#issuecomment",
                    "date": c["created_at"],
                    "decision_confidence": _decision_confidence(c["body"]),
                    "recency_weight": _recency_weight(c["created_at"]),
                    "state": t["state"],
                })
    return units


class DejaVuIndex:
    def __init__(self):
        self.units = []
        self.embeddings = None

    def build(self, units):
        self.units = units
        model = _get_model()
        texts = [u["text"] for u in units]
        self.embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def save(self, path_prefix="data/index"):
        np.save(f"{path_prefix}_embeddings.npy", self.embeddings)
        with open(f"{path_prefix}_units.json", "w") as f:
            json.dump(self.units, f)

    def load(self, path_prefix="data/index"):
        self.embeddings = np.load(f"{path_prefix}_embeddings.npy")
        with open(f"{path_prefix}_units.json") as f:
            self.units = json.load(f)

    def search(self, query, top_k=5, min_decision_confidence=0.0):
        model = _get_model()
        q_emb = model.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ q_emb  # cosine sim, since embeddings are normalized

        results = []
        for i, sim in enumerate(sims):
            unit = self.units[i]
            if unit["decision_confidence"] < min_decision_confidence:
                continue
            combined_score = float(sim) * (0.5 + 0.5 * unit["recency_weight"]) * (0.5 + 0.5 * unit["decision_confidence"])
            results.append({**unit, "similarity": float(sim), "combined_score": combined_score})

        results.sort(key=lambda r: -r["combined_score"])
        return results[:top_k]


if __name__ == "__main__":
    with open("data/discussions.json") as f:
        threads = json.load(f)

    print(f"Building decision units from {len(threads)} real threads...")
    units = build_decision_units(threads)
    print(f"Extracted {len(units)} searchable decision units")

    idx = DejaVuIndex()
    idx.build(units)
    idx.save()
    print("Index saved to data/index_*.{npy,json}")

    # sanity check search against real data
    test_query = "should we make async traits part of the language"
    print(f"\nTest query: {test_query!r}")
    for r in idx.search(test_query, top_k=3):
        print(f"  [{r['combined_score']:.3f}] {r['source_title']} — {r['url']}")
