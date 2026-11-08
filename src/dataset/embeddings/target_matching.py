"""Embedding-based selection of award and request sentences."""

import json
import math
import re
from functools import lru_cache, partial

from ..extraction.source.text import MONEY, sentence_fragments
from .encoding import create_embeddings, validate_settings as validate_encoder_settings

TARGET_CLASSES = ("judgment", "request")
EXCLUDED_CLASSES = ("prior_payment", "procedural_costs", "mixed", "other_payments")
PATTERN_CLASSES = TARGET_CLASSES + EXCLUDED_CLASSES


def validate_settings(settings, patterns):
    """
    Validate embedding settings, keyword groups and pattern classes before model loading.

    args:
        settings: Embedding model, runtime settings and similarity threshold.
        patterns: Nonempty target and exclusion phrase lists.

    returns:
        None.
    """
    required = {
        "model",
        "revision",
        "device",
        "batch_size",
        "max_length",
        "minimum_cosine_similarity",
        "candidate_keywords",
    }
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(f"targets.yaml embedding requires {sorted(required)}")
    validate_encoder_settings(settings)
    threshold = settings["minimum_cosine_similarity"]
    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("minimum_cosine_similarity must be between 0 and 1")
    keywords = settings["candidate_keywords"]
    if not isinstance(keywords, dict) or set(keywords) != {*PATTERN_CLASSES, "shared"}:
        raise ValueError(
            f"candidate_keywords requires {sorted((*PATTERN_CLASSES, 'shared'))} groups"
        )
    if not any(keywords.values()) or any(
        not isinstance(words, list)
        or any(not isinstance(word, str) or not word.strip() for word in words)
        for words in keywords.values()
    ):
        raise ValueError(
            "candidate_keywords groups must contain word stems, with at least one nonempty group"
        )
    if not isinstance(patterns, dict) or set(patterns) != set(PATTERN_CLASSES):
        raise ValueError(f"targets.yaml patterns requires {list(PATTERN_CLASSES)} lists")
    for label, phrases in patterns.items():
        if (
            not isinstance(phrases, list)
            or not phrases
            or any(not isinstance(p, str) or not p.strip() for p in phrases)
        ):
            raise ValueError(f"Target patterns for {label} must be nonempty strings")
        for phrase in phrases:
            prepare_text(phrase)


def eligible_classes(text, keywords):
    """Determine target and competing exclusion roles from local word stems.

    args:
        text: Original fragment or expanded contiguous source span.
        keywords: Target, shared and exclusion word-prefix groups.

    returns:
        Eligible classes, preserving role-specific precedence over shared stems.
    """
    matched = {
        group for group, words in keywords.items()
        if words and re.search(
            r"\b(?:" + "|".join(re.escape(word) for word in words) + r")\w*",
            text, re.IGNORECASE,
        )
    }
    classes = tuple(label for label in TARGET_CLASSES if label in matched)
    if not classes and "shared" in matched:
        classes = TARGET_CLASSES
    return classes + tuple(label for label in EXCLUDED_CLASSES if label in matched)


def candidate_sentences(text, keywords):
    """Select keyword-bearing comma fragments whose full sentence contains PLN evidence.

    args:
        text: Full normalized judgment text.
        keywords: Target, exclusion and shared word-prefix groups from targets.yaml.

    returns:
        Seed fragments with exact source offsets, sentence bounds and eligible classes.
        A seed need not contain money; expansion may recover it from the same sentence.
    """
    candidates = []
    for line in re.finditer(r"[^\n]+", text):
        if not MONEY.search(line[0]):
            continue
        chunk = {"text": line[0], "start": line.start(), "end": line.end()}
        for sentence in sentence_fragments(chunk):
            if not MONEY.search(sentence["text"]):
                continue
            for fragment in sentence_fragments(sentence, split_commas=True):
                classes = eligible_classes(fragment["text"], keywords)
                if classes:
                    candidates.append({
                        **fragment, "classes": classes,
                        "sentence_start": sentence["start"], "sentence_end": sentence["end"],
                    })
    return candidates


def prepare_text(text):
    """
    Neutralize example amounts and replace template slots with Polish role words.

    args:
        text: Pattern template or original source sentence.

    returns:
        Text used only for embedding; the original source remains unchanged.
    """
    replacements = {
        "defendant": "pozwanego",
        "claimant": "powoda",
        "amount": "100 zł",
        "previous_amount": "100 zł",
        "additional_amount": "100 zł",
    }
    for key in re.findall(r"\{([^{}]+)\}", text):
        if key not in replacements:
            raise ValueError(f"Unknown target-pattern placeholder: {key}")
        text = text.replace("{" + key + "}", replacements[key])
    return MONEY.sub("100 zł", text)


class TargetMatcher:
    """Cache pattern embeddings and rank source sentences by cosine similarity."""

    def __init__(self, settings, patterns, encoder=None):
        """
        Encode pattern templates once for reuse across judgments.

        args:
            settings: Embedding configuration and minimum similarity.
            patterns: Target and exclusion templates.
            encoder: Optional existing sentence encoder.

        returns:
            None.
        """
        validate_settings(settings, patterns)
        self.settings = settings
        self.patterns = [
            (label, phrase) for label, phrases in patterns.items() for phrase in phrases
        ]
        self.encode = encoder.encode if encoder is not None else partial(
            create_embeddings, settings=settings
        )
        self.pattern_vectors = self.encode(
            [prepare_text(phrase) for _, phrase in self.patterns]
        )

    def score(self, candidates):
        """Score a batch using unchanged pattern competition and exclusion tie priority.

        args:
            candidates: Nonempty source spans with eligible classes.

        returns:
            Pairs of winning class and source span enriched with similarity/pattern.
        """
        from .cosine_similarity import cosine_similarity

        vectors = self.encode([prepare_text(row["text"]) for row in candidates])
        scores = cosine_similarity(
            vectors, self.pattern_vectors, device=self.settings["device"]
        ).cpu()
        ranked = []
        for row, similarities in zip(candidates, scores, strict=True):
            best = {}
            for index, (label, phrase) in enumerate(self.patterns):
                if label not in row["classes"]:
                    continue
                score = float(similarities[index])
                if label not in best or score > best[label][0]:
                    best[label] = (score, phrase)
            label = max(best, key=lambda key: (best[key][0], key in EXCLUDED_CLASSES))
            score, phrase = best[label]
            ranked.append((label, {**row, "similarity": score, "pattern": phrase}))
        return ranked

    def grow(self, text, seed, label):
        """Grow a money-free target seed right first, then left within its sentence.

        args:
            text: Original normalized judgment for exact contiguous source spans.
            seed: Scored money-free fragment; final monetary spans must exceed the threshold.
            label: Target role that must remain the winner during expansion.

        returns:
            First accepted span with PLN evidence, or None. Money-free growth must
            improve similarity; monetary growth may decrease it but must exceed the
            threshold. Accepted left growth retries the right side before continuing.
        """
        sentence = {
            "text": text[seed["sentence_start"]:seed["sentence_end"]],
            "start": seed["sentence_start"], "end": seed["sentence_end"],
        }
        fragments = list(sentence_fragments(sentence, split_commas=True))
        left = right = next(
            index for index, part in enumerate(fragments)
            if part["start"] == seed["start"]
        )
        current = seed
        while True:
            for new_left, new_right in ((left, right + 1), (left - 1, right)):
                if new_left < 0 or new_right >= len(fragments):
                    continue
                start, end = fragments[new_left]["start"], fragments[new_right]["end"]
                span = text[start:end]
                candidate = {
                    **current, "text": span, "start": start, "end": end,
                    "classes": eligible_classes(span, self.settings["candidate_keywords"]),
                }
                winner, expanded = self.score([candidate])[0]
                if winner != label:
                    continue
                if MONEY.search(span):
                    if expanded["similarity"] > self.settings["minimum_cosine_similarity"]:
                        return expanded
                    continue
                if expanded["similarity"] > current["similarity"]:
                    left, right, current = new_left, new_right, expanded
                    break
            else:
                return None

    def match(self, text):
        """Select strongest eligible seeds and grow money-free target fragments.

        args:
            text: Full normalized judgment text.

        returns:
            First successful highest-ranked seed per class, with final contiguous
            source offsets and similarity. Exclusion seeds are never grown into targets.
        """
        candidates = candidate_sentences(text, self.settings["candidate_keywords"])
        result = dict.fromkeys(PATTERN_CLASSES)
        if not candidates:
            return result
        ranked = sorted(
            self.score(candidates), key=lambda match: match[1]["similarity"], reverse=True
        )
        for label, row in ranked:
            if result[label] is not None:
                continue
            if MONEY.search(row["text"]):
                if row["similarity"] <= self.settings["minimum_cosine_similarity"]:
                    continue
            else:
                if label not in TARGET_CLASSES:
                    continue
                row = self.grow(text, row, label)
                if row is None:
                    continue
            result[label] = row
        return result


@lru_cache(maxsize=1)
def cached_matcher(settings_json, patterns_json):
    """
    Reuse one model and pattern matrix for an unchanged configuration.

    args:
        settings_json: Serialized embedding configuration.
        patterns_json: Serialized pattern lists.

    returns:
        Initialized target matcher.
    """
    return TargetMatcher(json.loads(settings_json), json.loads(patterns_json))


def match_targets(text, config):
    """
    Locate target roles and competing exclusion contexts using configured embeddings.

    args:
        text: Full normalized judgment.
        config: Effective pipeline configuration.

    returns:
        Best matching source sentence per target or exclusion class.
    """
    if not candidate_sentences(text, config["embedding"]["candidate_keywords"]):
        return dict.fromkeys(PATTERN_CLASSES)
    matcher = cached_matcher(
        json.dumps(config["embedding"], sort_keys=True),
        json.dumps(config["target_patterns"], sort_keys=True),
    )
    return matcher.match(text)
