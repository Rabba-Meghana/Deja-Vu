"""
Deja Vu — ingestion layer.
Pulls REAL discussion threads (issues/PRs + all their comments) from a
public GitHub repo. No synthetic data. Free GitHub REST API (no auth
needed for public repos, 60 req/hr; set GITHUB_TOKEN in .env for 5000/hr,
still free).
"""

import os
import json
import time
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # optional, never hardcoded
API_BASE = "https://api.github.com"


def _headers():
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _get(url, params=None):
    resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        raise RuntimeError(
            "GitHub API rate limit hit. Add a free GITHUB_TOKEN to your .env "
            "(https://github.com/settings/tokens, no scopes needed for public repos) "
            "to raise the limit from 60/hr to 5000/hr — still free."
        )
    resp.raise_for_status()
    return resp


def fetch_discussion_threads(owner, repo, state="closed", max_threads=100, min_comments=3):
    """
    Pulls real closed issues/PRs with real debate (>= min_comments), plus
    every comment on each. This is the raw material — actual arguments,
    actual reasoning, actual reversals. Nothing generated.
    """
    threads = []
    page = 1
    fetched = 0

    while fetched < max_threads:
        resp = _get(
            f"{API_BASE}/repos/{owner}/{repo}/issues",
            params={"state": state, "per_page": 50, "page": page, "sort": "updated", "direction": "desc"},
        )
        batch = resp.json()
        if not batch:
            break

        for item in batch:
            if item.get("pull_request") is not None and False:
                continue  # RFCs repo treats PRs as issues too; we keep both
            if item.get("comments", 0) < min_comments:
                continue

            comments = _fetch_comments(owner, repo, item["number"])
            threads.append({
                "number": item["number"],
                "title": item["title"],
                "state": item["state"],
                "created_at": item["created_at"],
                "closed_at": item.get("closed_at"),
                "url": item["html_url"],
                "body": item.get("body") or "",
                "comments": comments,
            })
            fetched += 1
            if fetched >= max_threads:
                break

        page += 1
        time.sleep(0.3)  # be polite to the free API

    return threads


def _fetch_comments(owner, repo, issue_number):
    resp = _get(f"{API_BASE}/repos/{owner}/{repo}/issues/{issue_number}/comments", params={"per_page": 100})
    return [
        {"author": c["user"]["login"], "body": c["body"], "created_at": c["created_at"]}
        for c in resp.json()
    ]


def save_dataset(threads, path):
    with open(path, "w") as f:
        json.dump(threads, f, indent=2)
    print(f"Saved {len(threads)} real discussion threads to {path}")


def load_cached_or_fetch(owner, repo, max_threads, path="data/discussions.json"):
    """Never re-fetch what we already have — protects the free rate limit."""
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
        if len(existing) >= max_threads:
            print(f"Using {len(existing)} already-cached real threads from {path} (no API calls needed)")
            return existing
    threads = fetch_discussion_threads(owner, repo, max_threads=max_threads)
    save_dataset(threads, path)
    return threads



if __name__ == "__main__":
    import sys
    owner, repo = (sys.argv[1].split("/") if len(sys.argv) > 1 else ("rust-lang", "rfcs"))
    max_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    print(f"Fetching real discussion history from {owner}/{repo} (this hits the live GitHub API)...")
    threads = fetch_discussion_threads(owner, repo, max_threads=max_threads)
    save_dataset(threads, "data/discussions.json")
