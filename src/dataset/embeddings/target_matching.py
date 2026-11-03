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


def candidate_sentences(text, keywords):
    """
    Select sentence or comma-delimited clauses with PLN amounts and relevant stems.

    args:
        text: Full normalized judgment text.
        keywords: Target, exclusion and shared word-prefix groups from targets.yaml.

    returns:
        Candidate sentences with unchanged source offsets and eligible classes.
    """
    core = {
        group: re.compile(
            r"\b(?:" + "|".join(re.escape(word) for word in words) + r")\w*",
            re.IGNORECASE,
        )
        if words
        else re.compile(r"(?!)")
        for group, words in keywords.items()
    }
    sentences = []
    for line in re.finditer(r"[^\n]+", text):
        if not MONEY.search(line[0]) or not any(
            pattern.search(line[0]) for pattern in core.values()
        ):
            continue
        for sentence in sentence_fragments(
            {"text": line[0], "start": line.start(), "end": line.end()},
            split_commas=True,
        ):
            if not MONEY.search(sentence["text"]):
                continue
            classes = tuple(
                label
                for label in TARGET_CLASSES
                if core[label].search(sentence["text"])
            )
            if not classes and core["shared"].search(sentence["text"]):
                classes = TARGET_CLASSES
            classes += tuple(
                label
                for label in EXCLUDED_CLASSES
                if label in core and core[label].search(sentence["text"])
            )
            if classes:
                sentences.append({**sentence, "classes": classes})
    return sentences


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

    def match(self, text):
        """
        Assign each candidate its highest-scoring eligible class above the strict threshold.

        args:
            text: Full normalized judgment text.

        returns:
            Best matches per target/exclusion class, with source offsets and scores.
        """
        from .cosine_similarity import cosine_similarity

        sentences = candidate_sentences(text, self.settings["candidate_keywords"])
        result = dict.fromkeys(PATTERN_CLASSES)
        if not sentences:
            return result
        vectors = self.encode([prepare_text(row["text"]) for row in sentences])
        scores = cosine_similarity(
            vectors, self.pattern_vectors, device=self.settings["device"]
        ).cpu()
        for row, similarities in zip(sentences, scores, strict=True):
            best = {}
            for index, (label, phrase) in enumerate(self.patterns):
                if label not in row["classes"]:
                    continue
                score = float(similarities[index])
                if label not in best or score > best[label][0]:
                    best[label] = (score, phrase)
            label = max(best, key=lambda key: (best[key][0], key in EXCLUDED_CLASSES))
            score, phrase = best[label]
            if score > self.settings["minimum_cosine_similarity"] and (
                result[label] is None or score > result[label]["similarity"]
            ):
                result[label] = {**row, "similarity": score, "pattern": phrase}
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
