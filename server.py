"""
Deja Vu — MCP server.
Exposes the engine + reasoning layer as MCP tools so any MCP-compatible
client (Claude Desktop, etc.) can call check_for_deja_vu live.
"""

import asyncio
import json
import os
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import engine
import reasoning

server = Server("deja-vu")

_index = None


def _get_index():
    global _index
    if _index is None:
        _index = engine.DejaVuIndex()
        _index.load()
    return _index


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="check_for_deja_vu",
            description=(
                "Checks whether a new proposal, idea, or argument has already been "
                "substantively discussed or decided before, using real indexed discussion "
                "history. Returns whether it's deja vu, confidence level, the original "
                "reasoning if found, and a source link. Use this whenever someone is about "
                "to propose or debate something that might have already been settled."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "proposal": {
                        "type": "string",
                        "description": "The new idea or proposal being raised, in plain language.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many past discussions to consider as evidence. Default 5.",
                        "default": 5,
                    },
                },
                "required": ["proposal"],
            },
        ),
        Tool(
            name="search_past_discussions",
            description=(
                "Raw semantic search over indexed real discussion history, without the "
                "LLM judgment layer. Useful for browsing what's been discussed on a topic."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic or question to search for."},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name, arguments):
    try:
        if name == "check_for_deja_vu":
            idx = _get_index()
            candidates = idx.search(arguments["proposal"], top_k=arguments.get("top_k", 5))
            verdict = reasoning.judge_deja_vu(arguments["proposal"], candidates)
            return [TextContent(type="text", text=json.dumps(verdict, indent=2))]

        elif name == "search_past_discussions":
            idx = _get_index()
            results = idx.search(arguments["query"], top_k=arguments.get("top_k", 5))
            slim = [
                {"title": r["source_title"], "url": r["url"], "date": r["date"], "score": round(r["combined_score"], 3)}
                for r in results
            ]
            return [TextContent(type="text", text=json.dumps(slim, indent=2))]

        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    except RuntimeError as e:
        # e.g. missing GROQ_API_KEY — surfaced as a clean structured error, not a crash
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
    except FileNotFoundError:
        return [TextContent(type="text", text=json.dumps({
            "error": "Index not built yet. Run: python ingest.py <owner>/<repo> && python engine.py"
        }))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
