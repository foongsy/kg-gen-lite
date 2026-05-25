"""Text chunking with non-Latin (CJK) support.

Non-Latin text support (step 1 of 3):
    NLTK's sentence tokenizer is trained on Latin-script corpora and has no
    word-boundary signal for Chinese/Japanese/Korean text.  This module
    detects CJK-dominant input and falls back to punctuation-based splitting
    using the sentence-ending characters common to those scripts.
"""

import argparse
import re
import nltk


_CJK_SENTENCE_ENDING = re.compile(
    r"([。！？…；\n]+(?:\s*))"  # Chinese/Japanese sentence terminators
    r"|([।\n]+(?:\s*))"         # Devanagari (Hindi, etc.) Danda
)

# Unicode ranges used for CJK detection (mirrors llm_deduplicate._CJK_RANGES)
_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0xF900, 0xFAFF),
    (0x3040, 0x309F),
    (0x30A0, 0x30FF),
    (0xAC00, 0xD7AF),
)


def _is_cjk_char(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _is_cjk_text(text: str, threshold: float = 0.10) -> bool:
    """Return True if more than *threshold* fraction of characters are CJK."""
    if not text:
        return False
    cjk_count = sum(1 for c in text if _is_cjk_char(ord(c)))
    return cjk_count / len(text) > threshold


def _split_cjk_sentences(text: str) -> list[str]:
    """Split CJK text on sentence-ending punctuation.

    Re-attaches the terminator to the preceding sentence so each chunk
    includes its trailing punctuation (mirrors how NLTK returns sentences).
    """
    parts = _CJK_SENTENCE_ENDING.split(text)
    sentences: list[str] = []
    current = ""
    for part in parts:
        if part is None:
            continue
        if _CJK_SENTENCE_ENDING.fullmatch(part):
            current += part
            if current.strip():
                sentences.append(current.strip())
            current = ""
        else:
            current += part
    if current.strip():
        sentences.append(current.strip())
    return sentences or [text]


# Ensure NLTK punkt tokenizer is available for Latin-script text.
def _ensure_nltk_resource(resource_path, resource_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(resource_name, quiet=True)


_ensure_nltk_resource("tokenizers/punkt", "punkt")
_ensure_nltk_resource("tokenizers/punkt_tab", "punkt_tab")


def chunk_text(text: str, max_chunk_size: int = 500) -> list[str]:
    """Chunk text by sentence, respecting a maximum chunk size.

    For CJK-dominant text, splits on CJK sentence-ending punctuation
    instead of NLTK's Latin-trained tokeniser.
    Falls back to character-based chunking if a single CJK sentence is too
    large, and to word-based chunking (with single-word splitting) for Latin.

    Args:
        text: The text to chunk.
        max_chunk_size: Maximum length (in characters) of any chunk.
            Must be a positive integer.

    Returns:
        A list of text chunks, each at most *max_chunk_size* characters.

    Raises:
        ValueError: If *max_chunk_size* is not a positive integer.
    """
    if not isinstance(max_chunk_size, int) or max_chunk_size <= 0:
        raise ValueError(f"max_chunk_size must be a positive integer, got {max_chunk_size!r}")

    if _is_cjk_text(text):
        return _chunk_by_sentences(
            _split_cjk_sentences(text),
            max_chunk_size,
            cjk=True,
        )
    else:
        return _chunk_by_sentences(
            nltk.sent_tokenize(text),
            max_chunk_size,
            cjk=False,
        )


def _chunk_by_sentences(
    sentences: list[str], max_chunk_size: int, cjk: bool
) -> list[str]:
    """Accumulate *sentences* into chunks bounded by *max_chunk_size* chars."""
    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        separator = "" if cjk else " "
        if len(current_chunk) + len(sentence) + len(separator) <= max_chunk_size:
            current_chunk += sentence + separator
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            if len(sentence) > max_chunk_size:
                # Sentence too large — split by character (CJK) or word (Latin)
                if cjk:
                    for start in range(0, len(sentence), max_chunk_size):
                        chunks.append(sentence[start : start + max_chunk_size])
                else:
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(word) > max_chunk_size:
                            # Flush any pending chunk before splitting the word
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                                temp_chunk = ""
                            for start in range(0, len(word), max_chunk_size):
                                chunks.append(word[start : start + max_chunk_size])
                        elif len(temp_chunk) + len(word) + 1 <= max_chunk_size:
                            temp_chunk += word + " "
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word + " "
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
            else:
                current_chunk = sentence + separator

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Chunk large text into smaller pieces while respecting sentence boundaries."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the input text file. If omitted, reads from stdin.",
        default=None,
    )
    parser.add_argument(
        "--max_chunk_size",
        type=int,
        help="Maximum chunk size in characters (default=500).",
        default=500,
    )
    args = parser.parse_args()

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        import sys
        text = sys.stdin.read()

    result_chunks = chunk_text(text, max_chunk_size=args.max_chunk_size)
    for i, chunk in enumerate(result_chunks, start=1):
        print(f"--- Chunk {i} (length {len(chunk)}): ---")
        print(chunk)
        print()


if __name__ == "__main__":
    main()
