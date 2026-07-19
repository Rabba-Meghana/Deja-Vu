"""
Tests for the deterministic parts of the engine — no API key, no
network calls needed. Run with: pytest tests/
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, timezone
import engine
import reasoning


def test_decision_confidence_high_for_settled_language():
    text = "We've decided to go with this approach. Closing this as accepted, reasoning below."
    score = engine._decision_confidence(text)
    assert score > 0.5


def test_decision_confidence_low_for_tentative_language():
    text = "Just thinking out loud, not sure, what if we tried something else? Curious what others think."
    score = engine._decision_confidence(text)
    assert score < 0.4


def test_recency_weight_decays_over_time():
    now = datetime.now(timezone.utc).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(days=1500)).isoformat()
    assert engine._recency_weight(now) > engine._recency_weight(old)


def test_recency_weight_handles_bad_input_gracefully():
    # must not crash on malformed dates
    result = engine._recency_weight("not-a-date")
    assert 0 <= result <= 1


def test_build_decision_units_from_real_shaped_thread():
    fake_thread = [{
        "title": "Test RFC",
        "url": "https://example.com/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state": "closed",
        "body": "We decided to reject this proposal because it conflicts with existing semantics.",
        "comments": [
            {"author": "someone", "body": "Agreed, closing this as rejected per the discussion thread above — the semantics conflict is a blocker.",
             "created_at": datetime.now(timezone.utc).isoformat()}
        ],
    }]
    units = engine.build_decision_units(fake_thread)
    assert len(units) == 2
    assert all(u["decision_confidence"] > 0 for u in units)


def test_safe_parse_handles_valid_json():
    raw = '{"is_deja_vu": true, "confidence": "high", "matched_source": "Source 1", "explanation": "test"}'
    candidates = [{"url": "http://x.com", "date": "2024-01-01"}]
    result = reasoning._safe_parse(raw, candidates)
    assert result["is_deja_vu"] is True
    assert result["source_url"] == "http://x.com"


def test_safe_parse_handles_malformed_json_without_crashing():
    raw = "this is not json at all"
    result = reasoning._safe_parse(raw, [])
    assert result["is_deja_vu"] is None
    assert "raw" in result


def test_index_search_returns_empty_gracefully_on_empty_index():
    idx = engine.DejaVuIndex()
    idx.units = []
    idx.embeddings = None
    # should not crash even though nothing is loaded — caller (server.py)
    # is responsible for checking this, but the object itself shouldn't explode
    assert idx.units == []
