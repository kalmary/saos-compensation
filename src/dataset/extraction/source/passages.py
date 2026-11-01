"""Source selection, sentence batching and offset mappings."""

import re

from . import text as api
from .text import sentence_fragments


def fragments(text, limit):
    # Split at paragraphs and spaces while preserving source offsets; the model determines meaning.
    """
    Yield bounded paragraph fragments with original source offsets.

    args:
        text: Normalized Polish source text.
        limit: Maximum batch or fragment size.

    returns:
        Iterator of source fragments.
    """
    for line in re.finditer(r"[^\n]+", text):
        start = line.start()
        while start < line.end():
            end = min(start + limit, line.end())
            if end < line.end():
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            yield {"start": start, "end": end, "text": text[start:end]}
            start = end
            while start < line.end() and text[start].isspace():
                start += 1



def feature_input(text, config):
    """
    Select source text for factual passage classification.

    args:
        text: Normalized Polish source text.
        config: Effective pipeline or acquisition settings.

    returns:
        Selected source text, offsets and selection status.
    """
    reasons = list(api.REASONS.finditer(text))
    body_start = reasons[0].end() if len(reasons) == 1 else 0
    facts = re.search(
        r"^sąd(?:\s+\w+){0,3}\s+ustalił(?:\s+i\s+zważył)?(?:(?:,\s*|\s+)co następuje|\s+(?:następujący\s+)?stan faktyczny)?\s*[:.]?\s*$"
        r"|^(?:ustalenia faktyczne|stan faktyczny)\s*[:.]?\s*$",
        text[body_start:],
        re.IGNORECASE | re.MULTILINE,
    )
    start = body_start + facts.end() if facts else body_start
    reasoning = re.search(
        r"^sąd(?:\s+\w+){0,3}\s+zważył[^\n]*$",
        text[start:],
        re.IGNORECASE | re.MULTILINE,
    )
    end = start + reasoning.start() if reasoning else len(text)
    return {
        "text": text[start:end],
        "start": start,
        "end": end,
        "status": "court factual findings"
        if facts
        else "reasoning body; factual heading unavailable",
        "body_start": body_start,
    }



def source_blocks(text, limit):
    """
    Pack adjacent source sentences into bounded blocks with exact offsets.

    args:
        text: Normalized judgment text with paragraph boundaries.
        limit: Positive maximum block length in characters.

    returns:
        Iterator of source blocks; overlong sentences are split at spaces.
    """
    if limit <= 0:
        raise ValueError("Block character limit must be positive")
    start, end = None, None
    for paragraph in fragments(text, max(1, len(text))):
        for sentence in sentence_fragments(paragraph):
            for part in fragments(sentence["text"], limit):
                part_start = sentence["start"] + part["start"]
                part_end = sentence["start"] + part["end"]
                if start is not None and part_end - start > limit:
                    yield {"text": text[start:end], "start": start, "end": end}
                    start = None
                if start is None:
                    start = part_start
                end = part_end
    if start is not None:
        yield {"text": text[start:end], "start": start, "end": end}



def extraction_source(text, passages):
    # Each extraction input line maps to a complete source sentence, preserving negation and subject.
    """
    Assemble extraction input and map its sentences to source locations.

    args:
        text: Normalized Polish source text.
        passages: Accepted factual source fragments.

    returns:
        Assembled text and its source mapping.
    """
    sentences = {}
    for passage in passages:
        for paragraph in fragments(passage["text"], max(1, len(passage["text"]))):
            paragraph["start"] += passage["start"]
            paragraph["end"] += passage["start"]
            for sentence in sentence_fragments(paragraph):
                if text[sentence["start"] : sentence["end"]] != sentence["text"]:
                    raise ValueError("Feature passage does not match its source")
                sentences[(sentence["start"], sentence["end"])] = sentence
    parts, mapping, position = [], [], 0
    for sentence in sorted(sentences.values(), key=lambda row: row["start"]):
        parts.append(sentence["text"])
        mapping.append(
            {
                **sentence,
                "input_start": position,
                "input_end": position + len(sentence["text"]),
            }
        )
        position += len(sentence["text"]) + 1
    return "\n".join(parts), mapping



def extraction_batches(text, passages, limit):
    """
    Yield bounded factual batches with source mappings.

    args:
        text: Normalized Polish source text.
        passages: Accepted factual source fragments.
        limit: Maximum batch or fragment size.

    returns:
        Iterator of extraction text and source mappings.
    """
    _, sentences = extraction_source(text, passages)
    batch, size = [], 0
    for sentence in sentences:
        if batch and size + len(sentence["text"]) + 1 > limit:
            yield extraction_source(text, batch)
            batch, size = [], 0
        batch.append(sentence)
        size += len(sentence["text"]) + 1
    if batch:
        yield extraction_source(text, batch)
