import re
from dataclasses import dataclass
from typing import List

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    chunk_text: str


def semantic_chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[TextChunk]:
    """
    Splits text into overlapping, sentence-aware chunks without severing words.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly smaller than chunk_size")

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    sentences = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(cleaned) if s.strip()]
    chunks: List[TextChunk] = []
    current_parts: List[str] = []
    current_len = 0

    for sentence in sentences:
        # Handle single ultra-long sentences exceeding chunk_size by word-boundary slicing
        if len(sentence) > chunk_size:
            if current_parts:
                chunks.append(TextChunk(chunk_index=len(chunks), chunk_text=" ".join(current_parts)))
                current_parts = []
                current_len = 0
            words = sentence.split(" ")
            buf: List[str] = []
            buf_len = 0
            for word in words:
                added_len = len(word) + (1 if buf else 0)
                if buf and buf_len + added_len > chunk_size:
                    chunks.append(TextChunk(chunk_index=len(chunks), chunk_text=" ".join(buf)))
                    # Retain trailing words up to chunk_overlap
                    overlap_buf: List[str] = []
                    overlap_len = 0
                    for prev_w in reversed(buf):
                        if overlap_len + len(prev_w) + 1 <= chunk_overlap:
                            overlap_buf.insert(0, prev_w)
                            overlap_len += len(prev_w) + 1
                        else:
                            break
                    buf = overlap_buf
                    buf_len = sum(len(w) for w in buf) + max(0, len(buf) - 1)
                buf.append(word)
                buf_len += len(word) + (1 if len(buf) > 1 else 0)
            if buf:
                current_parts = buf
                current_len = buf_len
            continue

        added_len = len(sentence) + (1 if current_parts else 0)
        if current_parts and current_len + added_len > chunk_size:
            chunks.append(TextChunk(chunk_index=len(chunks), chunk_text=" ".join(current_parts)))

            # Build overlap tail from trailing sentences
            overlap_parts: List[str] = []
            overlap_len = 0
            for prev_sent in reversed(current_parts):
                if overlap_len + len(prev_sent) + 1 <= chunk_overlap:
                    overlap_parts.insert(0, prev_sent)
                    overlap_len += len(prev_sent) + 1
                else:
                    break
            current_parts = overlap_parts
            current_len = sum(len(p) for p in current_parts) + max(0, len(current_parts) - 1)

        current_parts.append(sentence)
        current_len += len(sentence) + (1 if len(current_parts) > 1 else 0)

    if current_parts:
        chunks.append(TextChunk(chunk_index=len(chunks), chunk_text=" ".join(current_parts)))

    return chunks
