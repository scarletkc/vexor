import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vexor import cli
from vexor.search import SearchResult
from vexor.services.mcp_service import VexorMcpServer
from vexor.services.search_service import ContentBudget, SearchResponse


@pytest.mark.parametrize("state", ["content", "unavailable", "empty"])
def test_cli_and_mcp_keep_their_complete_search_payloads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], state: str,
) -> None:
    has_content = state == "content"
    content = "你好\nbody" if has_content else None
    unavailable = "stale_line_range" if state == "unavailable" else None
    row = SearchResult(
        path=tmp_path / "example.py", score=0.123456, chunk_index=2,
        start_line=3, end_line=9, content=content,
        content_start_line=4 if has_content else None,
        content_end_line=5 if has_content else None,
        content_truncated=has_content, content_unavailable=unavailable,
    )
    response = SearchResponse(
        base_path=tmp_path, backend="test", results=[] if state == "empty" else [row],
        is_stale=state == "unavailable", index_empty=state == "empty",
        content_budget=ContentBudget(limit=100, used=7) if has_content else None,
    )
    expected_row = {
        "rank": 1, "score": 0.1235, "path": "./example.py",
        "absolute_path": str(tmp_path / "example.py"),
        "start_line": 3, "end_line": 9, "preview": None, "content": content,
        "content_start_line": 4 if has_content else None,
        "content_end_line": 5 if has_content else None,
        "content_truncated": has_content, "content_unavailable": unavailable,
    }
    expected = {
        "path": str(tmp_path), "backend": "test", "reranker": None,
        "stale": state == "unavailable", "index_empty": state == "empty",
        "results": [] if state == "empty" else [expected_row],
        "content_budget": {"limit": 100, "used": 7} if has_content else None,
    }
    server = VexorMcpServer(
        default_path=tmp_path, client=SimpleNamespace(search=lambda *_args, **_kwargs: response),
    )
    result = server._tool_search({"query": " query "})
    assert result["structuredContent"] == {"query": "query", **expected}
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
    assert result["isError"] is False

    cli._render_results_json(response, tmp_path)
    expected["results"] = [] if state == "empty" else [{**expected_row, "chunk_index": 2}]
    assert json.loads(capsys.readouterr().out) == expected
