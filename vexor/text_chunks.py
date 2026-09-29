"""Sliding windows over extracted text."""


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    size = max(int(chunk_size), 1)
    stride = max(size - max(int(overlap), 0), 1)
    chunks: list[str] = []
    start = 0
    length = len(normalized)
    while start < length:
        window = normalized[start : start + size].strip()
        if window:
            chunks.append(window)
        if start + size >= length:
            break
        start += stride
    return chunks
