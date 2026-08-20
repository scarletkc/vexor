# Collections Python API

A collection is a named, persistent set of caller-owned text records. Use one
when the source of truth is a database, message store, or another system that
already has stable record IDs and can send inserts and updates to Vexor. Use
`vexor index` when the source is a filesystem tree and Vexor should discover and
refresh files itself.

Collections are currently available through the Python API. They store records
in `collections.db`, separately from file indexes.

## Get a handle

Call `VexorClient(...).collection("name")` to get a `CollectionHandle`. Getting
the handle does not touch the database; the named collection is created by its
first successful non-empty write.

The handle uses the client's resolved embedding configuration unless you pass
overrides to `collection(...)`. The provider, model, and vector dimension are
pinned when the first write creates the collection. Any later write **or search**
configured with a different provider, model, or dimension raises an error that
tells you to recreate the collection instead of mixing incompatible vectors.
Reads are checked too: two models can share a vector width while embedding into
unrelated spaces, so a dimension match alone would let a query score vectors it
has nothing to do with and return a ranking that means nothing.

## Write records

`upsert_many(records)` accepts a sequence of mappings. Each record has:

- `id`: a non-empty caller-owned identifier. Vexor stores its string form.
- `text`: non-empty text to embed and search.
- `metadata`: an optional flat mapping used for filtering.

Metadata values may be `str`, `int`, `float`, `bool`, `datetime`, or `None`.
Keys must be non-empty strings. Lists, dictionaries, and other nested values are
rejected at write time rather than stored in a form no filter could ever match.

One asymmetry to know about: a `datetime` is stored as both an ISO 8601 string
and epoch seconds, so range filters on it work as expected, but reading a record
back returns the ISO **string**, not a `datetime` object. Parse it yourself if
you need the type: `datetime.fromisoformat(record.metadata["sent_at"])`.
Stored `datetime` values are returned as ISO 8601 strings when records are read.

An upsert replaces the record's metadata wholesale. When its text has not
changed, Vexor keeps the existing embedding and BM25 postings instead of
embedding it again, but still rewrites the metadata. The returned
`UpsertReport` reports `written`, `embedded`, and `skipped` counts.

## Filter records

Pass `filters` to `search()`. A bare value is equality shorthand; an operator
mapping provides the other comparisons.

| Operator | Meaning |
| --- | --- |
| `eq` | Equal to the value |
| `ne` | Not equal to the value, including records where the key is absent |
| `in` | Equal to any value in a non-empty list, tuple, set, or frozenset |
| `nin` | Equal to none of the values, including records where the key is absent |
| `gt`, `gte`, `lt`, `lte` | Number, boolean, or datetime range comparison |
| `exists` | Require the key to be present (`True`) or absent (`False`) |

All filter keys and all operators within them are ANDed. V1 has no OR. Because
`ne` and `nin` include records without the key, combine a negative condition
with `exists: True` when the key itself is required.

Filters resolve to record IDs before query scoring. An unknown key therefore
produces no matches for a positive condition, and a record excluded by a filter
cannot re-enter through dense or hybrid ranking.

## Search and scale

`search(query, *, top_k=10, filters=None, rerank="off")` returns
`RecordResult` objects with `id`, `text`, `metadata`, and `score`. Collection
search accepts only `rerank="off"` for dense similarity or `rerank="hybrid"`
to fuse dense and BM25 scores. Filters resolve before either mode scores
candidates.

Dense scoring is a brute-force matrix product over the filtered candidate set;
collections do not have an approximate nearest-neighbor index. Search cost
therefore tracks the size of one filtered slice. Without a filter, that slice
is the entire collection, so use selective metadata such as a chat, tenant, or
time boundary when the application naturally has one.

V1 stores one vector per record and does not chunk long text. Text that exceeds
the embedding provider's limit surfaces the provider error; Vexor does not
silently truncate it.

## Worked example: database-backed chat history

The application keeps the messages in its database and uses the message or turn
ID as the collection record ID. `chat_id` and `sent_at` form a selective slice
before relevance scoring:

```python
from datetime import datetime, timezone

from vexor import VexorClient

with VexorClient() as client:
    messages = client.collection("chat-history")
    report = messages.upsert_many(
        [
            {
                "id": "message-1042",
                "text": "The deployment failed because the token had expired.",
                "metadata": {
                    "chat_id": "chat-7",
                    "sent_at": datetime(2026, 8, 20, 9, 30, tzinfo=timezone.utc),
                },
            },
            {
                "id": "message-1043",
                "text": "Rotating the token fixed the deployment.",
                "metadata": {
                    "chat_id": "chat-7",
                    "sent_at": datetime(2026, 8, 20, 9, 35, tzinfo=timezone.utc),
                },
            },
        ]
    )

    results = messages.search(
        "Why did the deployment fail?",
        top_k=5,
        filters={
            "chat_id": "chat-7",
            "sent_at": {"gte": datetime(2026, 8, 20, tzinfo=timezone.utc)},
        },
        rerank="hybrid",
    )

    print(report.embedded, report.skipped)
    for result in results:
        print(result.id, result.score, result.text)
```

`get()` / `get_many()` read records by ID, `delete()` / `delete_many()` remove
them, `count()` reports the collection size, and `info()` returns the pinned
embedding contract. `drop()` deletes the named collection and all of its
records.
