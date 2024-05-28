#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any
from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.identify_language import IdentifyLanguageModel
from cs336_data.mask_pii import mask_emails, mask_phone_numbers, mask_ips
from cs336_data.harmful_content import NSFWModel, ToxicSpeechModel
from cs336_data.quality_filter import gopher_quality_filter
from cs336_data.deduplication import exact_line_deduplication, minhash_deduplication
from cs336_data.quality_classifier import load_and_predict


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text_from_html_bytes(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    model = IdentifyLanguageModel()
    return model.predict(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    model = NSFWModel()
    return model.predict(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    model = ToxicSpeechModel()
    return model.predict(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    label, score = load_and_predict(text)
    label = 'cc' if label == 'neg' else 'wiki'
    return label, score


def run_gopher_quality_filter(text: str) -> bool:
    return gopher_quality_filter(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    return exact_line_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    minhash_deduplication(
        in_paths=input_files,
        num_hashes=num_hashes,
        num_bands=num_bands,
        ngrams=ngrams,
        out_dir=output_directory,
        jaccard_threshold=jaccard_threshold,
    )
