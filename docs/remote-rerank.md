## Remote Rerank Design (SiliconFlow)

This document describes a future remote rerank integration. 

### Goals
- Add a new rerank strategy that calls a remote API.
- Use the existing 2x top-k candidate pool (rerank 2k, return k).
- Keep configuration explicit and simple.

### Non-goals
- No streaming support.
- No new cache layer for rerank responses.
- No change to indexing or embedding pipelines.

### API Targets
Compatible rerank endpoints share a simple request shape:
- SiliconFlow: `https://api.siliconflow.cn/v1/rerank`
- Knox: `https://api.knox.chat/v1/rerank`
- Jina: `https://api.jina.ai/v1/rerank`
- Authorization: `Bearer <token>`

Request body:
```
{
  "model": "BAAI/bge-reranker-v2-m3",
  "query": "Apple",
  "documents": ["apple", "banana", "fruit", "vegetable"]
}
```

Response shape (SiliconFlow/Jina-style):
```
{
  "id": "<string>",
  "results": [
    {
      "document": {"text": "<string>"},
      "index": 123,
      "relevance_score": 123
    }
  ],
  "tokens": {"input_tokens": 123, "output_tokens": 123}
}
```

Response shape (Knox-style):
```
{
  "object": "list",
  "data": [
    {"index": 0, "relevance_score": 0.43},
    {"index": 1, "relevance_score": 0.42}
  ],
  "model": "rerank-2.5",
  "usage": {"total_tokens": 26}
}
```

### Config
Add a new rerank option and sub-config:
```
{
  "rerank": "remote",
  "remote_rerank": {
    "base_url": "https://api.siliconflow.cn/v1/rerank",
    "api_key": "YOUR_KEY",
    "model": "BAAI/bge-reranker-v2-m3"
  }
}
```

Required fields:
- remote_rerank.base_url
- remote_rerank.api_key
- remote_rerank.model

### CLI
Add support in `vexor config`:
- `--rerank remote`
- `--set-remote-rerank-url <URL>`
- `--set-remote-rerank-model <MODEL>`
- `--set-remote-rerank-api-key <KEY>`
- `--clear-remote-rerank`

Validation:
- If `--rerank remote` is set, the required remote_rerank fields must exist.
- If required fields are missing, raise `typer.BadParameter`.

### Rerank Flow
1) Build semantic results as today.
2) Candidate pool = min(total, 2 * top_k).
3) Build `documents` from each candidate using the existing document builder
   (path + preview).
4) POST to `remote_rerank.base_url` with model, query, documents.
5) Use `results[*].index` or `data[*].index` to reorder candidates.
6) Return top_k results to the user.

### Output
The header should include the reranker:
- `Backend: <...> | Reranker: remote`

### Error Handling
- HTTP errors or timeouts should surface as user-facing errors.
- If the remote API returns a partial list, keep remaining results in original
  order and append them after the reranked ones.

### Security
- API key stored in config.json under `remote_rerank.api_key`.
- If omitted, `VEXOR_REMOTE_RERANK_API_KEY` is used at runtime.
- Ensure error messages never print the key.

### Testing Plan
- Unit test: remote rerank reorders by index and respects top_k.
- Integration test: config validation errors when required fields missing.
- Integration test: rerank header includes `Reranker: remote`.

### Future Extensions
- Allow alternative remote endpoints by changing base_url.
- Optional fallback strategy if remote rerank fails (semantic-only).
