"""Embedding-based selection of award and request sentences."""

import json
import math
import re
from functools import lru_cache
from pathlib import Path

from .text import MONEY, sentence_fragments

ROOT = Path(__file__).resolve().parents[3]


def validate_settings(settings, patterns):
    """
    Validate embedding settings and the two pattern classes before model loading.

    args:
        settings: Embedding model, runtime settings and similarity threshold.
        patterns: Nonempty judgment and request phrase lists.

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
    if not isinstance(patterns, dict) or set(patterns) != {"judgment", "request"}:
        raise ValueError("target_pattern.yaml requires judgment and request lists")
    for label, phrases in patterns.items():
        if (
            not isinstance(phrases, list)
            or not phrases
            or any(not isinstance(p, str) or not p.strip() for p in phrases)
        ):
            raise ValueError(f"Target patterns for {label} must be nonempty strings")
        for phrase in phrases:
            prepare_text(phrase)


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
            patterns: Judgment and request templates.
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
        Select the highest-scoring eligible sentence for each class; a tie abstains.

        args:
            text: Full normalized judgment text.

        returns:
            Judgment/request matches with source offsets and scores, or None per class.
        """
        from .cosine_similarity import cosine_similarity

        sentences = [
            sentence
            for line in re.finditer(r"[^\n]+", text)
            for sentence in sentence_fragments(
                {"text": line[0], "start": line.start(), "end": line.end()}
            )
        ]
        result = {"judgment": None, "request": None}
        if not sentences:
            return result
        vectors = self.encoder.encode([prepare_text(row["text"]) for row in sentences])
        scores = cosine_similarity(
            vectors, self.pattern_vectors, device=self.settings["device"]
        ).cpu()
        for row, similarities in zip(sentences, scores, strict=True):
            best = {}
            for index, (label, phrase) in enumerate(self.patterns):
                score = round(float(similarities[index]), 7)
                if label not in best or score > best[label][0]:
                    best[label] = (score, phrase)
            if best["judgment"][0] == best["request"][0]:
                continue
            label = max(best, key=lambda key: best[key][0])
            score, phrase = best[label]
            if score >= self.settings["minimum_cosine_similarity"] and (
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
    Locate judgment and optional request sentences using configured embeddings.

    args:
        text: Full normalized judgment.
        config: Effective pipeline configuration.

    returns:
        Best matching source sentences for both classes.
    """
    matcher = cached_matcher(
        json.dumps(config["embedding"], sort_keys=True),
        json.dumps(config["target_patterns"], sort_keys=True),
    )
    return matcher.match(text)
