"""Centralized user-facing text for Vexor CLI."""

from __future__ import annotations

class Styles:
    ERROR = "red"
    WARNING = "yellow"
    SUCCESS = "green"
    INFO = "dim"
    TITLE = "bold cyan"
    TABLE_HEADER = "bold magenta"


class Messages:
    APP_HELP = "Vexor – A vector-powered CLI for semantic search over filenames."
    HELP_QUERY = "Text used to semantically match file names."
    HELP_SEARCH_PATH = "Root directory whose cached index will be used."
    HELP_SEARCH_TOP = "Number of results to display."
    HELP_INCLUDE_HIDDEN = "Use the index built with hidden files included."
    HELP_INDEX_PATH = "Root directory to scan recursively for indexing."
    HELP_INDEX_INCLUDE = "Include hidden files and directories when building the index."
    HELP_SET_API_KEY = "Persist an API key in ~/.vexor/config.json."
    HELP_CLEAR_API_KEY = "Remove the stored API key."
    HELP_SET_MODEL = "Set the default embedding model."
    HELP_SET_BATCH = "Set the default batch size (0 = single request)."
    HELP_SHOW_CONFIG = "Show current configuration."

    ERROR_API_KEY_MISSING = (
        "Gemini API key is missing or still set to the placeholder. "
        "Configure it via `vexor config --set-api-key <token>` or an environment variable."
    )
    ERROR_API_KEY_INVALID = (
        "Gemini API key is invalid. Verify the stored token and try again."
    )
    ERROR_GENAI_PREFIX = "Gemini API request failed: "
    ERROR_NO_EMBEDDINGS = "Gemini API returned no embeddings."
    ERROR_EMPTY_QUERY = "Query text must not be empty."
    ERROR_BATCH_NEGATIVE = "Batch size must be >= 0"

    INFO_NO_FILES = "No files found in the selected directory."
    INFO_NO_RESULTS = "No matching files found."
    ERROR_INDEX_MISSING = (
        "No cached index found for {path}. Run `vexor index --path \"{path}\"` first."
    )
    INFO_INDEX_SAVED = "Index saved to {path}."
    INFO_INDEX_EMPTY = "Index contains no files."
    INFO_INDEX_UP_TO_DATE = "Index already matches the current directory; nothing to do."
    WARNING_INDEX_STALE = "Cached index for {path} appears outdated; run `vexor index --path \"{path}\"` to refresh."
    INFO_INDEX_RUNNING = "Indexing files under {path}..."
    INFO_API_SAVED = "API key saved."
    INFO_API_CLEARED = "API key cleared."
    INFO_MODEL_SET = "Default model set to {value}."
    INFO_BATCH_SET = "Default batch size set to {value}."
    INFO_CONFIG_SUMMARY = (
        "API key set: {api}\n"
        "Default model: {model}\n"
        "Default batch size: {batch}"
    )
    INFO_SEARCH_RUNNING = "Searching cached index under {path}..."

    TABLE_TITLE = "Vexor semantic file search results"
    TABLE_HEADER_INDEX = "#"
    TABLE_HEADER_SIMILARITY = "Similarity"
    TABLE_HEADER_PATH = "File path"
    TABLE_BACKEND_PREFIX = "Backend: "
