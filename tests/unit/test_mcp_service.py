from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from vexor import __version__
from vexor.search import SearchResult
from vexor.services.index_service import IndexResult, IndexStatus
from vexor.services.mcp_service import (
    INDEX_TOOL,
    JSONRPC_INVALID_PARAMS,
    JSONRPC_METHOD_NOT_FOUND,
    JSONRPC_PARSE_ERROR,
    LATEST_PROTOCOL_VERSION,
    SEARCH_TOOL,
    VexorMcpServer,
    build_tool_definitions,
    serve,
)
from vexor.services.search_service import SearchResponse


class FakeClient:
    def __init__(self, base_path: Path, *, search_error: Exception | None = None):
        self.base_path = base_path
        self.search_error = search_error
        self.search_calls: list[dict] = []
        self.index_calls: list[dict] = []

    def search(self, query, **kwargs):
        if self.search_error is not None:
            raise self.search_error
        self.search_calls.append({"query": query, **kwargs})
        return SearchResponse(
            base_path=self.base_path,
            backend="openai",
            results=[
                SearchResult(
                    path=self.base_path / "src" / "config.py",
                    score=0.91234,
                    preview="config loader",
                    start_line=1,
                    end_line=20,
                )
            ],
            is_stale=False,
            index_empty=False,
            reranker=None,
        )

    def index(self, path, **kwargs):
        self.index_calls.append({"path": path, **kwargs})
        return IndexResult(status=IndexStatus.STORED, cache_path=None, files_indexed=3)


def make_server(tmp_path: Path, **kwargs) -> tuple[VexorMcpServer, FakeClient]:
    client = FakeClient(tmp_path, **kwargs)
    server = VexorMcpServer(default_path=tmp_path, client=client)
    return server, client


def request(method: str, params=None, request_id=1) -> dict:
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def tool_call(name: str, arguments: dict, request_id=1) -> dict:
    return request(
        "tools/call", {"name": name, "arguments": arguments}, request_id=request_id
    )


def decode_tool_payload(response: dict) -> dict:
    content = response["result"]["content"]
    assert content[0]["type"] == "text"
    return json.loads(content[0]["text"])


def test_initialize_echoes_supported_protocol_version(tmp_path):
    server, _ = make_server(tmp_path)
    response = server.handle_message(
        request("initialize", {"protocolVersion": "2025-03-26"})
    )
    result = response["result"]
    assert result["protocolVersion"] == "2025-03-26"
    assert result["serverInfo"] == {
        "name": "vexor",
        "title": "Vexor",
        "version": __version__,
    }
    assert result["capabilities"]["tools"] == {"listChanged": False}
    assert result["instructions"]
    assert server.initialized is True


def test_initialize_falls_back_to_latest_protocol_version(tmp_path):
    server, _ = make_server(tmp_path)
    response = server.handle_message(
        request("initialize", {"protocolVersion": "1999-01-01"})
    )
    assert response["result"]["protocolVersion"] == LATEST_PROTOCOL_VERSION


def test_ping_returns_empty_result(tmp_path):
    server, _ = make_server(tmp_path)
    response = server.handle_message(request("ping", request_id=7))
    assert response == {"jsonrpc": "2.0", "id": 7, "result": {}}


def test_protocol_requests_do_not_construct_default_client(tmp_path):
    server = VexorMcpServer(default_path=tmp_path)

    server.handle_message(request("initialize"))
    server.handle_message(request("ping"))
    server.handle_message(request("tools/list"))

    assert server._client is None


def test_notifications_produce_no_response(tmp_path):
    server, _ = make_server(tmp_path)
    message = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    assert server.handle_message(message) is None


def test_unknown_method_returns_method_not_found(tmp_path):
    server, _ = make_server(tmp_path)
    response = server.handle_message(request("resources/list"))
    assert response["error"]["code"] == JSONRPC_METHOD_NOT_FOUND


def test_unknown_notification_is_ignored(tmp_path):
    server, _ = make_server(tmp_path)
    assert server.handle_message({"jsonrpc": "2.0", "method": "resources/list"}) is None


def test_invalid_request_without_jsonrpc_field(tmp_path):
    server, _ = make_server(tmp_path)
    response = server.handle_message({"id": 1, "method": "ping"})
    assert response["error"]["code"] == -32600


def test_tools_list_advertises_search_and_index(tmp_path):
    server, _ = make_server(tmp_path)
    response = server.handle_message(request("tools/list"))
    tools = response["result"]["tools"]
    names = [tool["name"] for tool in tools]
    assert names == [SEARCH_TOOL, INDEX_TOOL]
    search_tool = tools[0]
    assert search_tool["inputSchema"]["required"] == ["query"]
    assert str(tmp_path.resolve()) in search_tool["inputSchema"]["properties"]["path"][
        "description"
    ]


def test_tool_definitions_mode_enum_matches_available_modes(tmp_path):
    definitions = build_tool_definitions(tmp_path)
    modes = definitions[0]["inputSchema"]["properties"]["mode"]["enum"]
    assert "auto" in modes and "code" in modes and "outline" in modes


def test_search_tool_returns_ranked_results(tmp_path):
    server, client = make_server(tmp_path)
    response = server.handle_message(
        tool_call(SEARCH_TOOL, {"query": "config loader", "top": 3})
    )
    assert response["result"]["isError"] is False
    payload = decode_tool_payload(response)
    assert payload["query"] == "config loader"
    assert payload["backend"] == "openai"
    assert payload["index_empty"] is False
    hit = payload["results"][0]
    assert hit["rank"] == 1
    assert hit["score"] == 0.9123
    assert hit["path"] == "./src/config.py"
    assert hit["start_line"] == 1
    assert client.search_calls[0]["top"] == 3
    assert client.search_calls[0]["path"] == tmp_path.resolve()


def test_search_tool_top_defaults_and_rejects_out_of_range(tmp_path):
    server, client = make_server(tmp_path)
    server.handle_message(tool_call(SEARCH_TOOL, {"query": "q"}))
    assert client.search_calls[0]["top"] == 5
    for bad_top in (0, 999):
        response = server.handle_message(
            tool_call(SEARCH_TOOL, {"query": "q", "top": bad_top})
        )
        assert response["error"]["code"] == JSONRPC_INVALID_PARAMS, bad_top


def test_search_tool_passes_scan_flags(tmp_path):
    server, client = make_server(tmp_path)
    server.handle_message(
        tool_call(
            SEARCH_TOOL,
            {"query": "q", "respect_gitignore": False, "recursive": False},
        )
    )
    call = client.search_calls[0]
    assert call["respect_gitignore"] is False
    assert call["recursive"] is False
    # Defaults when omitted.
    server.handle_message(tool_call(SEARCH_TOOL, {"query": "q"}))
    call = client.search_calls[1]
    assert call["respect_gitignore"] is True
    assert call["recursive"] is True


def test_relative_path_resolves_against_default_path(tmp_path):
    (tmp_path / "src").mkdir()
    server, client = make_server(tmp_path)
    response = server.handle_message(
        tool_call(SEARCH_TOOL, {"query": "q", "path": "src"})
    )
    assert response["result"]["isError"] is False
    assert client.search_calls[0]["path"] == (tmp_path / "src").resolve()


def test_success_results_include_structured_content(tmp_path):
    server, _ = make_server(tmp_path)
    response = server.handle_message(tool_call(SEARCH_TOOL, {"query": "q"}))
    result = response["result"]
    assert result["structuredContent"] == decode_tool_payload(response)
    assert result["structuredContent"]["results"][0]["rank"] == 1


def test_error_results_omit_structured_content(tmp_path):
    server, _ = make_server(tmp_path, search_error=RuntimeError("boom"))
    response = server.handle_message(tool_call(SEARCH_TOOL, {"query": "q"}))
    assert response["result"]["isError"] is True
    assert "structuredContent" not in response["result"]


def test_tools_advertise_output_schemas_and_strict_input(tmp_path):
    definitions = build_tool_definitions(tmp_path)
    for tool in definitions:
        assert tool["inputSchema"]["additionalProperties"] is False
        assert "outputSchema" in tool
    search_tool, index_tool = definitions
    assert search_tool["inputSchema"]["properties"]["query"]["minLength"] == 1
    assert "respect_gitignore" in search_tool["inputSchema"]["properties"]
    assert "recursive" in index_tool["inputSchema"]["properties"]
    assert search_tool["outputSchema"]["properties"]["results"]["type"] == "array"
    assert index_tool["outputSchema"]["properties"]["status"]["enum"] == [
        "stored",
        "up_to_date",
        "empty",
    ]


@pytest.mark.parametrize("tool_name", [SEARCH_TOOL, INDEX_TOOL])
def test_tools_reject_unknown_arguments(tmp_path, tool_name):
    server, client = make_server(tmp_path)
    arguments = {"recusive": False}
    if tool_name == SEARCH_TOOL:
        arguments["query"] = "q"

    response = server.handle_message(tool_call(tool_name, arguments))

    assert response["error"]["code"] == JSONRPC_INVALID_PARAMS
    assert "recusive" in response["error"]["message"]
    assert client.search_calls == []
    assert client.index_calls == []


def test_tool_call_rejects_non_object_arguments_and_name(tmp_path):
    server, _ = make_server(tmp_path)
    for params in (
        {"name": INDEX_TOOL, "arguments": []},
        {"name": INDEX_TOOL, "arguments": None},
        {"name": 123, "arguments": {}},
    ):
        response = server.handle_message(request("tools/call", params))
        assert response["error"]["code"] == JSONRPC_INVALID_PARAMS, params


def test_search_tool_missing_query_is_invalid_params(tmp_path):
    server, _ = make_server(tmp_path)
    response = server.handle_message(tool_call(SEARCH_TOOL, {}))
    assert response["error"]["code"] == JSONRPC_INVALID_PARAMS


def test_search_tool_rejects_bad_argument_types(tmp_path):
    server, _ = make_server(tmp_path)
    cases = [
        {"query": "q", "top": "five"},
        {"query": "q", "mode": "invalid-mode"},
        {"query": "q", "include_hidden": "yes"},
        {"query": "q", "extensions": [1, 2]},
        {"query": "q", "extensions": ".py"},
        {"query": "q", "exclude_patterns": "build/**"},
        {"query": "q", "path": 42},
    ]
    for arguments in cases:
        response = server.handle_message(tool_call(SEARCH_TOOL, arguments))
        assert response["error"]["code"] == JSONRPC_INVALID_PARAMS, arguments


def test_search_tool_missing_directory_is_tool_error(tmp_path):
    server, _ = make_server(tmp_path)
    response = server.handle_message(
        tool_call(SEARCH_TOOL, {"query": "q", "path": str(tmp_path / "missing")})
    )
    assert response["result"]["isError"] is True
    payload = decode_tool_payload(response)
    assert "Directory does not exist" in payload["error"]


def test_search_tool_execution_error_keeps_server_alive(tmp_path):
    server, _ = make_server(tmp_path, search_error=RuntimeError("api key missing"))
    response = server.handle_message(tool_call(SEARCH_TOOL, {"query": "q"}))
    assert response["result"]["isError"] is True
    payload = decode_tool_payload(response)
    assert "api key missing" in payload["error"]
    # Follow-up request still works.
    assert server.handle_message(request("ping"))["result"] == {}


def test_index_tool_reports_status(tmp_path):
    server, client = make_server(tmp_path)
    response = server.handle_message(
        tool_call(INDEX_TOOL, {"mode": "name", "include_hidden": True})
    )
    payload = decode_tool_payload(response)
    assert payload["status"] == "stored"
    assert payload["files_indexed"] == 3
    assert payload["mode"] == "name"
    assert client.index_calls[0]["include_hidden"] is True


def test_unknown_tool_is_invalid_params(tmp_path):
    server, _ = make_server(tmp_path)
    response = server.handle_message(tool_call("vexor_delete_everything", {}))
    assert response["error"]["code"] == JSONRPC_INVALID_PARAMS


def test_serve_round_trip_with_parse_error(tmp_path):
    server, _ = make_server(tmp_path)
    stdin = io.BytesIO(
        b'{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}\n'
        b"not json\n"
        b"\n"
        b'{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}\n'
        b'{"jsonrpc": "2.0", "method": "notifications/initialized"}\n'
    )
    stdout = io.BytesIO()
    serve(server, stdin, stdout)
    lines = stdout.getvalue().decode("utf-8").strip().splitlines()
    assert len(lines) == 3
    first, second, third = (json.loads(line) for line in lines)
    assert first["id"] == 1
    assert first["result"]["protocolVersion"] == LATEST_PROTOCOL_VERSION
    assert second["error"]["code"] == JSONRPC_PARSE_ERROR
    assert second["id"] is None
    assert third["id"] == 2
    assert [tool["name"] for tool in third["result"]["tools"]] == [
        SEARCH_TOOL,
        INDEX_TOOL,
    ]


def test_serve_accepts_text_streams(tmp_path):
    server, _ = make_server(tmp_path)
    stdin = io.StringIO('{"jsonrpc": "2.0", "id": 5, "method": "ping"}\n')
    stdout = io.StringIO()
    serve(server, stdin, stdout)
    assert json.loads(stdout.getvalue().strip()) == {
        "jsonrpc": "2.0",
        "id": 5,
        "result": {},
    }


def test_search_result_paths_outside_base_stay_absolute(tmp_path):
    class OutsideClient(FakeClient):
        def search(self, query, **kwargs):
            response = super().search(query, **kwargs)
            outside = Path(tmp_path.anchor) / "elsewhere" / "file.py"
            response.results[0].path = outside
            return response

    client = OutsideClient(tmp_path)
    server = VexorMcpServer(default_path=tmp_path, client=client)
    response = server.handle_message(tool_call(SEARCH_TOOL, {"query": "q"}))
    payload = decode_tool_payload(response)
    assert payload["results"][0]["path"] == str(
        Path(tmp_path.anchor) / "elsewhere" / "file.py"
    )


def test_emit_update_notice_writes_when_newer(monkeypatch, tmp_path):
    import vexor.services.mcp_service as mcp_service
    import vexor.services.system_service as system_service

    monkeypatch.delenv(mcp_service.ENV_NO_UPDATE_CHECK, raising=False)
    monkeypatch.setattr(
        system_service, "check_for_update", lambda current, **kw: "99.0.0"
    )
    stream = io.StringIO()
    mcp_service.emit_update_notice(stream)
    assert "99.0.0" in stream.getvalue()
    assert "vexor update --upgrade" in stream.getvalue()


def test_emit_update_notice_silent_when_current(monkeypatch):
    import vexor.services.mcp_service as mcp_service
    import vexor.services.system_service as system_service

    monkeypatch.delenv(mcp_service.ENV_NO_UPDATE_CHECK, raising=False)
    monkeypatch.setattr(
        system_service, "check_for_update", lambda current, **kw: None
    )
    stream = io.StringIO()
    mcp_service.emit_update_notice(stream)
    assert stream.getvalue() == ""


def test_emit_update_notice_disabled_by_env(monkeypatch):
    import vexor.services.mcp_service as mcp_service
    import vexor.services.system_service as system_service

    monkeypatch.setenv(mcp_service.ENV_NO_UPDATE_CHECK, "1")

    def _explode(current, **kw):
        raise AssertionError("update check must not run when disabled")

    monkeypatch.setattr(system_service, "check_for_update", _explode)
    stream = io.StringIO()
    mcp_service.emit_update_notice(stream)
    assert stream.getvalue() == ""


def test_emit_update_notice_swallows_errors(monkeypatch):
    import vexor.services.mcp_service as mcp_service
    import vexor.services.system_service as system_service

    monkeypatch.delenv(mcp_service.ENV_NO_UPDATE_CHECK, raising=False)

    def _boom(current, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(system_service, "check_for_update", _boom)
    stream = io.StringIO()
    mcp_service.emit_update_notice(stream)
    assert stream.getvalue() == ""
