"""Embedding-based selection of award and request sentences."""

import json
import math
import re
from functools import lru_cache, partial
from itertools import batched

from ..extraction.source.text import MONEY, sentence_fragments
from .encoding import create_embeddings, validate_settings as validate_encoder_settings

TARGET_CLASSES = ("judgment", "request")
EXCLUDED_CLASSES = ("prior_payment", "procedural_costs", "mixed", "other_payments")
PATTERN_CLASSES = TARGET_CLASSES + EXCLUDED_CLASSES + ("judgment-denial",)


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
    # Denial screening uses every sentence, so it has no keyword group.
    keyword_groups = {*TARGET_CLASSES, *EXCLUDED_CLASSES, "shared"}
    if not isinstance(keywords, dict) or set(keywords) != keyword_groups:
        raise ValueError(
            f"candidate_keywords requires {sorted(keyword_groups)} groups"
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
    Grow each sentence to the first comma-delimited part containing a PLN amount.

    args:
        text: Full normalized judgment text.
        keywords: Target, exclusion and shared word-prefix groups from targets.yaml.

    returns:
        At most one keyword-eligible monetary prefix per sentence, in source order.
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
            {"text": line[0], "start": line.start(), "end": line.end()}
        ):
            for part in sentence_fragments(sentence, split_commas=True):
                prefix = text[sentence["start"]:part["end"]]
                if not MONEY.search(prefix):
                    continue
                classes = tuple(
                    label for label in TARGET_CLASSES if core[label].search(prefix)
                )
                if not classes and core["shared"].search(prefix):
                    classes = TARGET_CLASSES
                classes += tuple(
                    label
                    for label in EXCLUDED_CLASSES
                    if label in core and core[label].search(prefix)
                )
                if classes:
                    sentences.append(
                        {
                            "text": prefix,
                            "start": sentence["start"],
                            "end": part["end"],
                            "classes": classes,
                        }
                    )
                break
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

    def match_denial(self, text):
        """Check full sentences for denial before any monetary candidate selection.

        args:
            text: Full normalized judgment, including sentences without PLN or keywords.

        returns:
            Strongest denial match in the first matching batch, or None.
        """
        from .cosine_similarity import cosine_similarity

        indices = [
            index for index, (label, _) in enumerate(self.patterns)
            if label == "judgment-denial"
        ]
        costs = [
            index for index, (label, _) in enumerate(self.patterns)
            if label == "procedural_costs"
        ]
        sentences = (
            sentence
            for line in re.finditer(r"[^\n]+", text)
            for sentence in sentence_fragments(
                {"text": line[0], "start": line.start(), "end": line.end()}
            )
        )
        for batch in batched(sentences, self.settings["batch_size"]):
            vectors = self.encode([prepare_text(row["text"]) for row in batch])
            scores = cosine_similarity(
                vectors, self.pattern_vectors[indices + costs], device=self.settings["device"]
            ).cpu()
            values, patterns = scores[:, :len(indices)].max(dim=1)
            cost_values = scores[:, len(indices):].max(dim=1).values
            values = values.masked_fill(values <= cost_values, -1)
            best = int(values.argmax())
            score = float(values[best])
            if score > self.settings["minimum_cosine_similarity"]:
                return {
                    **batch[best], "similarity": score,
                    "pattern": self.patterns[indices[int(patterns[best])]][1],
                }
        return None

    def match(self, text):
        """
        Rank the first monetary prefixes across sentences using a strict threshold.

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
    Locate denial evidence before monetary target and exclusion contexts.

    args:
        text: Full normalized judgment.
        config: Effective pipeline configuration.

    returns:
        Denial evidence, or best monetary matches per target/exclusion class.
    """
    if not text.strip():
        return dict.fromkeys(PATTERN_CLASSES)
    matcher = cached_matcher(
        json.dumps(config["embedding"], sort_keys=True),
        json.dumps(config["target_patterns"], sort_keys=True),
    )
    denial = matcher.match_denial(text)
    if denial is not None:
        return {**dict.fromkeys(PATTERN_CLASSES), "judgment-denial": denial}
    return matcher.match(text)
