"""MCP (Model Context Protocol) stdio server exposing Vexor to AI agents.

Implements the tools-only subset of MCP (initialize, ping, tools/list,
tools/call) as newline-delimited JSON-RPC 2.0 over stdio. Hand-rolled on
purpose: the official ``mcp`` SDK pulls in a large dependency tree
(httpx, anyio, pydantic, starlette, ...) for transports and features
that a tools-only stdio server does not need.
"""

from __future__ import annotations

import json

import sys
import threading
from pathlib import Path
from typing import Any, IO, Iterable, Mapping, Sequence, TextIO

from .. import __version__
from ..config import ENV_NO_UPDATE_CHECK
from ..modes import available_modes
from ..text import Messages
from ..utils import format_path, resolve_directory

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
LATEST_PROTOCOL_VERSION = PROTOCOL_VERSIONS[0]

JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603

MAX_TOP = 50
DEFAULT_TOP = 5

SEARCH_TOOL = "vexor_search"
INDEX_TOOL = "vexor_index"

_COMMON_TOOL_ARGUMENTS = frozenset(
    {
        "path",
        "mode",
        "include_hidden",
        "respect_gitignore",
        "recursive",
        "extensions",
        "exclude_patterns",
    }
)
_TOOL_ARGUMENTS = {
    SEARCH_TOOL: _COMMON_TOOL_ARGUMENTS | {"query", "top", "no_cache"},
    INDEX_TOOL: _COMMON_TOOL_ARGUMENTS,
}


class InvalidToolArguments(ValueError):
    """Raised when tool arguments fail structural validation."""


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if (
        not isinstance(value, str)
        and isinstance(value, Sequence)
        and all(isinstance(item, str) for item in value)
    ):
        return tuple(value)
    raise InvalidToolArguments(
        Messages.MCP_INVALID_ARGUMENTS.format(
            reason=f"'{field}' must be a list of strings"
        )
    )


def _bool_argument(value: Any, field: str, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise InvalidToolArguments(
        Messages.MCP_INVALID_ARGUMENTS.format(reason=f"'{field}' must be a boolean")
    )


def _mode_argument(value: Any) -> str:
    mode = value if value is not None else "auto"
    if not isinstance(mode, str) or mode not in available_modes():
        raise InvalidToolArguments(
            Messages.MCP_INVALID_ARGUMENTS.format(
                reason=f"'mode' must be one of: {', '.join(available_modes())}"
            )
        )
    return mode


def _top_argument(value: Any) -> int:
    if value is None:
        return DEFAULT_TOP
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_TOP
    ):
        raise InvalidToolArguments(
            Messages.MCP_INVALID_ARGUMENTS.format(
                reason=f"'top' must be an integer between 1 and {MAX_TOP}"
            )
        )
    return value


def _validate_tool_arguments(name: str, arguments: Mapping[str, Any]) -> None:
    unknown = [field for field in arguments if field not in _TOOL_ARGUMENTS[name]]
    unknown.sort(key=repr)
    if unknown:
        raise InvalidToolArguments(
            Messages.MCP_INVALID_ARGUMENTS.format(
                reason=Messages.MCP_ARGUMENTS_UNKNOWN.format(
                    names=", ".join(repr(field) for field in unknown)
                )
            )
        )


def _common_scan_properties(default_path: Path) -> dict[str, Any]:
    return {
        "path": {
            "type": "string",
            "description": Messages.MCP_ARG_PATH.format(path=default_path),
        },
        "mode": {
            "type": "string",
            "enum": available_modes(),
            "default": "auto",
            "description": Messages.MCP_ARG_MODE,
        },
        "include_hidden": {
            "type": "boolean",
            "default": False,
            "description": Messages.MCP_ARG_INCLUDE_HIDDEN,
        },
        "respect_gitignore": {
            "type": "boolean",
            "default": True,
            "description": Messages.MCP_ARG_RESPECT_GITIGNORE,
        },
        "recursive": {
            "type": "boolean",
            "default": True,
            "description": Messages.MCP_ARG_RECURSIVE,
        },
        "extensions": {
            "type": "array",
            "items": {"type": "string"},
            "description": Messages.MCP_ARG_EXTENSIONS,
        },
        "exclude_patterns": {
            "type": "array",
            "items": {"type": "string"},
            "description": Messages.MCP_ARG_EXCLUDE_PATTERNS,
        },
    }


_RESULT_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rank": {"type": "integer"},
        "score": {"type": "number"},
        "path": {"type": "string"},
        "absolute_path": {"type": "string"},
        "start_line": {"type": ["integer", "null"]},
        "end_line": {"type": ["integer", "null"]},
        "preview": {"type": ["string", "null"]},
    },
    "required": ["rank", "score", "path", "absolute_path"],
}

SEARCH_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "path": {"type": "string"},
        "backend": {"type": ["string", "null"]},
        "reranker": {"type": ["string", "null"]},
        "stale": {"type": "boolean"},
        "index_empty": {"type": "boolean"},
        "results": {"type": "array", "items": _RESULT_ITEM_SCHEMA},
    },
    "required": [
        "query",
        "path",
        "backend",
        "reranker",
        "stale",
        "index_empty",
        "results",
    ],
}

INDEX_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "mode": {"type": "string"},
        "status": {"type": "string", "enum": ["stored", "up_to_date", "empty"]},
        "files_indexed": {"type": "integer"},
    },
    "required": ["path", "mode", "status", "files_indexed"],
}


def build_tool_definitions(default_path: Path) -> list[dict[str, Any]]:
    """Return MCP tool definitions advertised by ``tools/list``."""
    scan_properties = _common_scan_properties(default_path)
    search_properties: dict[str, Any] = {
        "query": {
            "type": "string",
            "minLength": 1,
            "description": Messages.MCP_ARG_QUERY,
        },
        "top": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_TOP,
            "default": DEFAULT_TOP,
            "description": Messages.MCP_ARG_TOP,
        },
        "no_cache": {
            "type": "boolean",
            "default": False,
            "description": Messages.MCP_ARG_NO_CACHE,
        },
    }
    search_properties.update(scan_properties)
    return [
        {
            "name": SEARCH_TOOL,
            "description": Messages.MCP_TOOL_SEARCH_DESCRIPTION,
            "inputSchema": {
                "type": "object",
                "properties": search_properties,
                "required": ["query"],
                "additionalProperties": False,
            },
            "outputSchema": SEARCH_OUTPUT_SCHEMA,
        },
        {
            "name": INDEX_TOOL,
            "description": Messages.MCP_TOOL_INDEX_DESCRIPTION,
            "inputSchema": {
                "type": "object",
                "properties": dict(scan_properties),
                "required": [],
                "additionalProperties": False,
            },
            "outputSchema": INDEX_OUTPUT_SCHEMA,
        },
    ]


def _text_result(payload: Mapping[str, Any], *, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "content": [
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False)}
        ],
        "isError": is_error,
    }
    if not is_error:
        # Mirror the text payload as structured content (MCP 2025-06-18);
        # older clients ignore the extra field.
        result["structuredContent"] = payload
    return result


def _tool_error(tool: str, reason: str) -> dict[str, Any]:
    return _text_result(
        {"error": Messages.MCP_TOOL_FAILED.format(tool=tool, reason=reason)},
        is_error=True,
    )


class VexorMcpServer:
    """Stateful JSON-RPC message handler for one MCP session."""

    def __init__(
        self,
        *,
        default_path: Path | str | None = None,
        client: Any | None = None,
    ) -> None:
        self._client = client
        self.default_path = Path(default_path or Path.cwd()).resolve()
        self.protocol_version = LATEST_PROTOCOL_VERSION
        self.initialized = False

    def _get_client(self) -> Any:
        """Create the API client only when a tool is first invoked."""

        if self._client is None:
            from ..api import VexorClient

            self._client = VexorClient()
        return self._client

    # -- JSON-RPC plumbing -------------------------------------------------

    def handle_message(self, message: Any) -> dict[str, Any] | None:
        """Handle one decoded JSON-RPC message; return a response or None."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error_response(
                None, JSONRPC_INVALID_REQUEST, "Invalid JSON-RPC request."
            )
        method = message.get("method")
        has_id = "id" in message
        request_id = message.get("id")
        if not isinstance(method, str):
            # Responses to server-initiated requests (none exist) or malformed
            # frames: answer only when the peer expects a reply.
            if has_id:
                return _error_response(
                    request_id, JSONRPC_INVALID_REQUEST, "Invalid JSON-RPC request."
                )
            return None
        try:
            return self._dispatch(method, message.get("params"), has_id, request_id)
        except Exception as exc:  # pragma: no cover - defensive backstop
            if not has_id:
                return None
            return _error_response(request_id, JSONRPC_INTERNAL_ERROR, str(exc))

    def _dispatch(
        self,
        method: str,
        params: Any,
        has_id: bool,
        request_id: Any,
    ) -> dict[str, Any] | None:
        if not has_id:
            # Notifications (including unknown ones) never get a response.
            return None
        if method == "initialize":
            return _result_response(request_id, self._handle_initialize(params))
        if method == "ping":
            return _result_response(request_id, {})
        if method == "tools/list":
            return _result_response(
                request_id, {"tools": build_tool_definitions(self.default_path)}
            )
        if method == "tools/call":
            return self._handle_tools_call(params, request_id)
        return _error_response(
            request_id,
            JSONRPC_METHOD_NOT_FOUND,
            Messages.MCP_METHOD_NOT_FOUND.format(method=method),
        )

    def _handle_initialize(self, params: Any) -> dict[str, Any]:
        requested = ""
        if isinstance(params, dict):
            requested = str(params.get("protocolVersion") or "")
        self.protocol_version = (
            requested if requested in PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION
        )
        self.initialized = True
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": "vexor",
                "title": "Vexor",
                "version": __version__,
            },
            "instructions": Messages.MCP_SERVER_INSTRUCTIONS,
        }

    # -- Tools -------------------------------------------------------------

    def _handle_tools_call(self, params: Any, request_id: Any) -> dict[str, Any]:
        if not isinstance(params, dict):
            return _error_response(
                request_id,
                JSONRPC_INVALID_PARAMS,
                Messages.MCP_INVALID_ARGUMENTS.format(reason="params must be an object"),
            )
        name = params.get("name")
        if not isinstance(name, str):
            return _error_response(
                request_id,
                JSONRPC_INVALID_PARAMS,
                Messages.MCP_INVALID_ARGUMENTS.format(
                    reason=Messages.MCP_TOOL_NAME_INVALID
                ),
            )
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            return _error_response(
                request_id,
                JSONRPC_INVALID_PARAMS,
                Messages.MCP_INVALID_ARGUMENTS.format(
                    reason="arguments must be an object"
                ),
            )
        handlers = {SEARCH_TOOL: self._tool_search, INDEX_TOOL: self._tool_index}
        handler = handlers.get(name)
        if handler is None:
            return _error_response(
                request_id,
                JSONRPC_INVALID_PARAMS,
                Messages.MCP_UNKNOWN_TOOL.format(name=name),
            )
        try:
            _validate_tool_arguments(name, arguments)
            return _result_response(request_id, handler(arguments))
        except InvalidToolArguments as exc:
            return _error_response(request_id, JSONRPC_INVALID_PARAMS, str(exc))

    def _resolve_scan_arguments(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        raw_path = arguments.get("path")
        if raw_path is not None and not isinstance(raw_path, str):
            raise InvalidToolArguments(
                Messages.MCP_INVALID_ARGUMENTS.format(reason="'path' must be a string")
            )
        if raw_path:
            candidate = Path(raw_path).expanduser()
            path = candidate if candidate.is_absolute() else self.default_path / candidate
        else:
            path = self.default_path
        return {
            "path": path,
            "mode": _mode_argument(arguments.get("mode")),
            "include_hidden": _bool_argument(
                arguments.get("include_hidden"), "include_hidden"
            ),
            "respect_gitignore": _bool_argument(
                arguments.get("respect_gitignore"), "respect_gitignore", default=True
            ),
            "recursive": _bool_argument(
                arguments.get("recursive"), "recursive", default=True
            ),
            "extensions": _string_list(arguments.get("extensions"), "extensions")
            or None,
            "exclude_patterns": _string_list(
                arguments.get("exclude_patterns"), "exclude_patterns"
            )
            or None,
        }

    def _tool_search(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise InvalidToolArguments(
                Messages.MCP_INVALID_ARGUMENTS.format(
                    reason="'query' must be a non-empty string"
                )
            )
        top = _top_argument(arguments.get("top"))
        no_cache = _bool_argument(arguments.get("no_cache"), "no_cache")
        scan = self._resolve_scan_arguments(arguments)
        try:
            directory = resolve_directory(scan["path"])
            response = self._get_client().search(
                query.strip(),
                path=directory,
                top=top,
                mode=scan["mode"],
                include_hidden=scan["include_hidden"],
                respect_gitignore=scan["respect_gitignore"],
                recursive=scan["recursive"],
                extensions=scan["extensions"],
                exclude_patterns=scan["exclude_patterns"],
                no_cache=no_cache,
            )
        except Exception as exc:
            return _tool_error(SEARCH_TOOL, str(exc))
        return _text_result(
            {
                "query": query.strip(),
                "path": str(response.base_path),
                "backend": response.backend,
                "reranker": response.reranker,
                "stale": response.is_stale,
                "index_empty": response.index_empty,
                "results": [
                    {
                        "rank": rank,
                        "score": round(float(result.score), 4),
                        "path": format_path(result.path, response.base_path),
                        "absolute_path": str(result.path),
                        "start_line": result.start_line,
                        "end_line": result.end_line,
                        "preview": result.preview,
                    }
                    for rank, result in enumerate(response.results, start=1)
                ],
            }
        )

    def _tool_index(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        scan = self._resolve_scan_arguments(arguments)
        try:
            directory = resolve_directory(scan["path"])
            result = self._get_client().index(
                directory,
                mode=scan["mode"],
                include_hidden=scan["include_hidden"],
                respect_gitignore=scan["respect_gitignore"],
                recursive=scan["recursive"],
                extensions=scan["extensions"],
                exclude_patterns=scan["exclude_patterns"],
            )
        except Exception as exc:
            return _tool_error(INDEX_TOOL, str(exc))
        return _text_result(
            {
                "path": str(directory),
                "mode": scan["mode"],
                "status": result.status.value,
                "files_indexed": result.files_indexed,
            }
        )


def _result_response(request_id: Any, result: Mapping[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _write_line(stream: IO[Any], text: str) -> None:
    data = text + "\n"
    try:
        stream.write(data.encode("utf-8"))
    except TypeError:
        stream.write(data)
    stream.flush()


def serve(server: VexorMcpServer, stdin: Iterable[Any], stdout: IO[Any]) -> None:
    """Run the newline-delimited JSON-RPC loop until stdin is exhausted."""
    for raw_line in stdin:
        if isinstance(raw_line, bytes):
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                _write_line(
                    stdout,
                    json.dumps(
                        _error_response(
                            None, JSONRPC_PARSE_ERROR, Messages.MCP_PARSE_ERROR
                        )
                    ),
                )
                continue
        else:
            line = raw_line
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            _write_line(
                stdout,
                json.dumps(
                    _error_response(None, JSONRPC_PARSE_ERROR, Messages.MCP_PARSE_ERROR)
                ),
            )
            continue
        response = server.handle_message(message)
        if response is not None:
            _write_line(stdout, json.dumps(response, ensure_ascii=False))


def emit_update_notice(stream: TextIO) -> None:
    """Write a one-line stderr notice when a newer release exists.

    Best-effort: the check is TTL-cached, uses a short network timeout, and
    stays silent on any failure. Never writes to stdout, which is reserved
    for the protocol.
    """
    try:
        from .system_service import check_for_update, update_check_enabled

        if not update_check_enabled():
            return
        latest = check_for_update(__version__)
        if latest:
            print(
                Messages.MCP_UPDATE_AVAILABLE.format(
                    latest=latest, current=__version__
                ),
                file=stream,
                flush=True,
            )
    except Exception:
        pass


def _start_update_notice_thread() -> None:
    thread = threading.Thread(
        target=emit_update_notice,
        args=(sys.stderr,),
        name="vexor-mcp-update-check",
        daemon=True,
    )
    thread.start()


def serve_stdio(default_path: Path | str | None = None) -> None:
    """Serve MCP over the process's real stdin/stdout."""
    server = VexorMcpServer(default_path=default_path)
    stdin = getattr(sys.stdin, "buffer", sys.stdin)
    stdout = getattr(sys.stdout, "buffer", sys.stdout)
    print(
        Messages.MCP_SERVER_READY.format(path=server.default_path),
        file=sys.stderr,
        flush=True,
    )
    _start_update_notice_thread()
    serve(server, stdin, stdout)
