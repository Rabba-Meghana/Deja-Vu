"""
Deja Vu — local API server for the dashboard.
Runs entirely on localhost. No hosting cost. Bridges the static
dashboard HTML to engine.py + reasoning.py.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import engine
import reasoning
import os

app = Flask(__name__, static_folder="dashboard")
CORS(app)

_index = None


def get_index():
    global _index
    if _index is None:
        _index = engine.DejaVuIndex()
        _index.load()
    return _index


@app.route("/")
def dashboard():
    return send_from_directory("dashboard", "index.html")


@app.route("/api/status")
def status():
    idx = get_index()
    groq_ready = bool(os.getenv("GROQ_API_KEY"))
    return jsonify({
        "indexed_units": len(idx.units),
        "groq_ready": groq_ready,
    })


@app.route("/api/check", methods=["POST"])
def check():
    data = request.get_json()
    proposal = data.get("proposal", "").strip()
    if not proposal:
        return jsonify({"error": "Empty proposal"}), 400

    idx = get_index()
    candidates = idx.search(proposal, top_k=5)

    try:
        verdict = reasoning.judge_deja_vu(proposal, candidates)
        if verdict.get("is_deja_vu"):
            deep = reasoning.deep_analyze(proposal, verdict, candidates)
            if deep:
                verdict["deep_analysis"] = deep
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400

    verdict["candidates_considered"] = [
        {"title": c["source_title"], "url": c["url"], "date": c["date"][:10], "score": round(c["combined_score"], 3)}
        for c in candidates
    ]
    return jsonify(verdict)


if __name__ == "__main__":
    print("Deja Vu dashboard running at http://localhost:5050")
    app.run(port=5050, debug=False)
