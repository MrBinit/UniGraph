import argparse
import asyncio
import json
import re
from pathlib import Path

from app.core.paths import resolve_project_path
from app.services.serpapi_search_service import asearch_google_batch, search_google


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:80] or "query"


def _load_queries(args) -> list[str]:
    queries = [query.strip() for query in (args.query or []) if query and query.strip()]
    if args.queries_file:
        path = resolve_project_path(args.queries_file)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            queries.append(line)
    # Preserve order while deduping.
    deduped: list[str] = []
    seen = set()
    for query in queries:
        if query in seen:
            continue
        seen.add(query)
        deduped.append(query)
    return deduped


def _to_markdown(query: str, payload: dict) -> str:
    lines = [f"# {query}", ""]
    lines.extend(_search_parameter_lines(payload))
    lines.extend(_ai_overview_lines(payload))
    lines.extend(_organic_result_lines(payload))
    return "\n".join(lines).strip() + "\n"


def _search_parameter_lines(payload: dict) -> list[str]:
    params = payload.get("search_parameters", {})
    gl = str(params.get("gl", "")).strip()
    hl = str(params.get("hl", "")).strip()
    if not (gl or hl):
        return []
    return [f"Source: SerpAPI Google Search (gl={gl or '-'}, hl={hl or '-'})", ""]


def _ai_overview_list_item(item) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return ""
    title = str(item.get("title", "")).strip()
    snippet = str(item.get("snippet", "")).strip()
    return f"{title}: {snippet}".strip(": ")


def _ai_overview_lines(payload: dict) -> list[str]:
    ai_overview = payload.get("ai_overview")
    if not isinstance(ai_overview, dict) or not ai_overview:
        return []

    lines = ["## AI Overview"]
    for key in ("title", "text", "snippet", "description"):
        value = ai_overview.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(value.strip())

    bullets = ai_overview.get("list")
    if isinstance(bullets, list):
        for item in bullets:
            text = _ai_overview_list_item(item)
            if text:
                lines.append(f"- {text}")
    lines.append("")
    return lines


def _organic_result_lines(payload: dict) -> list[str]:
    organic_results = payload.get("organic_results")
    lines = ["## Organic Results"]
    if not isinstance(organic_results, list) or not organic_results:
        lines.append("No organic results returned.")
        return lines

    for index, result in enumerate(organic_results, start=1):
        if not isinstance(result, dict):
            continue
        title = str(result.get("title", "")).strip()
        link = str(result.get("link", "")).strip()
        snippet = str(result.get("snippet", "")).strip()
        lines.append(f"{index}. {title or '(untitled)'}")
        if link:
            lines.append(f"URL: {link}")
        if snippet:
            lines.append(snippet)
        lines.append("")
    return lines


def _write_result(output_dir: Path, query: str, payload: dict):
    file_stem = f"serpapi-google-{_slug(query)}"
    (output_dir / f"{file_stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{file_stem}.md").write_text(_to_markdown(query, payload), encoding="utf-8")


async def _run_async(queries: list[str], args, output_dir: Path):
    batch = await asearch_google_batch(
        queries,
        gl=args.gl,
        hl=args.hl,
        num=args.num,
    )
    for item in batch:
        query = item.get("query", "")
        error = str(item.get("error", "")).strip()
        if error:
            print(f"ERROR | query={query} | {error}")
            continue
        payload = item.get("result", {})
        if isinstance(payload, dict):
            _write_result(output_dir, query, payload)
            print(f"OK | query={query}")


def main():
    parser = argparse.ArgumentParser(description="Fetch Google Search results via SerpAPI.")
    parser.add_argument(
        "--query", action="append", help="Search query. Repeat for multiple queries."
    )
    parser.add_argument(
        "--queries-file",
        default="",
        help="Optional file path with one query per line.",
    )
    parser.add_argument("--gl", default=None, help="Country code (for example: us).")
    parser.add_argument("--hl", default=None, help="Language code (for example: en).")
    parser.add_argument("--num", type=int, default=None, help="Number of results.")
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory where raw .json and .md files are written.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Run single-query synchronous mode (no async queue).",
    )
    args = parser.parse_args()

    queries = _load_queries(args)
    if not queries:
        raise RuntimeError("Provide at least one --query or --queries-file.")

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sync:
        query = queries[0]
        payload = search_google(query, gl=args.gl, hl=args.hl, num=args.num)
        _write_result(output_dir, query, payload)
        print(f"OK | query={query}")
        return

    asyncio.run(_run_async(queries, args, output_dir))


if __name__ == "__main__":
    main()
