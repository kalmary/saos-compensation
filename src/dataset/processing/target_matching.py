"""Embedding-based selection of award and request sentences."""

import json
import math
import re
from functools import lru_cache
from pathlib import Path

from .text import MONEY, sentence_fragments

ROOT = Path(__file__).resolve().parents[3]
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
        raise ValueError(f"embedding.yaml requires {sorted(required)}")
    for key in ("model", "revision"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(f"Invalid embedding {key}")
    if settings["device"] not in ("cpu", "mps", "cuda"):
        raise ValueError("Embedding device must be cpu, mps or cuda")
    for key in ("batch_size", "max_length"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"Embedding {key} must be a positive integer")
    if settings["max_length"] < 4:
        raise ValueError(
            "Embedding max_length must accommodate special tokens and text"
        )
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
        raise ValueError(f"target_pattern.yaml requires {list(PATTERN_CLASSES)} lists")
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
    Select sentences containing both a numeric PLN amount and a target-related word stem.

    args:
        text: Full normalized judgment text.
        keywords: Target, exclusion and shared word-prefix groups from embedding.yaml.

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
            {"text": line[0], "start": line.start(), "end": line.end()}
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


class SentenceEncoder:
    """Mean-pooled multilingual embeddings with complete long-sentence coverage."""

    def __init__(self, settings):
        """
        Load the pinned tokenizer and model into the selected PyTorch device.

        args:
            settings: Validated embedding configuration.

        returns:
            None.
        """
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.settings = settings
        device = settings["device"]
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Embedding CUDA was requested but is unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Embedding MPS was requested but is unavailable")
        options = {
            "revision": settings["revision"],
            "cache_dir": ROOT / "embedding_weights",
        }
        self.tokenizer = AutoTokenizer.from_pretrained(settings["model"], **options)
        self.model = (
            AutoModel.from_pretrained(
                settings["model"], use_safetensors=True, **options
            )
            .to(device)
            .eval()
        )
        if settings["max_length"] > self.model.config.max_position_embeddings:
            raise ValueError("Embedding max_length exceeds the model context")

    def encode(self, texts):
        """
        Batch all token windows and pool them back into one vector per input sentence.

        args:
            texts: Nonempty list of texts to encode.

        returns:
            Float32 embedding matrix on the configured device.
        """
        import torch

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.settings["max_length"],
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        mapping = tokens.pop("overflow_to_sample_mapping")
        device = self.settings["device"]
        sums = torch.zeros((len(texts), self.model.config.hidden_size), device=device)
        counts = torch.zeros((len(texts), 1), device=device)
        with torch.inference_mode():
            for start in range(0, len(mapping), self.settings["batch_size"]):
                end = start + self.settings["batch_size"]
                batch = {
                    key: value[start:end].to(device) for key, value in tokens.items()
                }
                output = self.model(**batch).last_hidden_state.float()
                mask = batch["attention_mask"].unsqueeze(-1).float()
                indices = mapping[start:end].to(device)
                sums.index_add_(0, indices, (output * mask).sum(dim=1))
                counts.index_add_(0, indices, mask.sum(dim=1))
        return sums / counts.clamp_min(1)


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
        self.encoder = encoder if encoder is not None else SentenceEncoder(settings)
        self.pattern_vectors = self.encoder.encode(
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
        vectors = self.encoder.encode([prepare_text(row["text"]) for row in sentences])
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
