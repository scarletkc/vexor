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

### API Target
SiliconFlow rerank endpoint:
- POST https://api.siliconflow.cn/v1/rerank
- Authorization: Bearer <token>

Request body:
```
{
  "model": "BAAI/bge-reranker-v2-m3",
  "query": "Apple",
  "documents": ["apple", "banana", "fruit", "vegetable"]
}
```

Response shape:
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

### Config
Add a new rerank option and sub-config:
```
{
  "rerank": "remote",
  "remote_rerank": {
    "base_url": "https://api.siliconflow.cn/v1/rerank",
    "api_key": "YOUR_KEY",
    "model": "BAAI/bge-reranker-v2-m3",
    "timeout_s": 10
  }
}
```

Required fields:
- remote_rerank.base_url
- remote_rerank.api_key
- remote_rerank.model

Optional fields:
- remote_rerank.timeout_s (default 10s)

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
5) Use `results[*].index` + `relevance_score` to reorder candidates.
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
- Ensure error messages never print the key.

### Testing Plan
- Unit test: remote rerank reorders by index and respects top_k.
- Integration test: config validation errors when required fields missing.
- Integration test: rerank header includes `Reranker: remote`.

### Future Extensions
- Allow alternative remote endpoints by changing base_url.
- Optional fallback strategy if remote rerank fails (semantic-only).
