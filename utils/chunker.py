def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split a long string into overlapping chunks of words.

    Args:
        text:       The raw text to split (e.g. extracted from a PDF page).
        chunk_size: How many words per chunk.
        overlap:    How many words from the end of one chunk to repeat at
                    the start of the next, so context isn't lost at boundaries.

    Returns:
        A list of non-empty string chunks.
    """
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap   # slide forward, keeping `overlap` words

    return chunks
