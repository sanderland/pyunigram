import logging
import os
import sys
import time
from collections import Counter

from datasets import load_dataset


def create_logger(tag: str, verbose: bool = True):
    default_fields = logging.getLogRecordFactory()
    t0 = time.perf_counter()

    # https://stackoverflow.com/questions/63056270/python-logging-time-since-start-in-seconds
    def record_factory(*args, **kwargs):
        record = default_fields(*args, **kwargs)
        record.uptime = time.perf_counter() - t0
        record.level_nocaps = record.levelname.lower()
        return record

    logging.setLogRecordFactory(record_factory)
    logger = logging.getLogger(tag)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter(f"[%(uptime)6.1fs][{tag}] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_texts(dataset_name):
    """Load texts from the specified dataset."""
    if dataset_name.endswith("300mb"):
        dataset = load_dataset(
            "sanderland/monolingual-tokenizer-data",
            data_files=[f"{dataset_name}.txt"],
            split="train",
            streaming=True,
        )
        texts = [text for item in dataset for text in item["text"]]
    elif dataset_name == "swift":
        with open("../script_bpe/tests/data/taylorswift.txt", "r", encoding="utf-8") as f:
            texts = [f.read()]
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [item["text"] for item in dataset]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return texts


def compute_compression_stats(tokenizer, texts):
    """Compute compression statistics for a tokenizer and texts."""
    total_bytes = 0
    total_tokens = 0
    total_chars = 0
    for text in texts:
        text_bytes = len(text.encode("utf-8"))
        tokens = tokenizer.encode(text)
        total_bytes += text_bytes
        total_tokens += len(tokens)
        total_chars += len(text)
    bytes_per_token = total_bytes / total_tokens if total_tokens else 0
    chars_per_token = total_chars / total_tokens if total_tokens else 0
    return {
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "bytes_per_token": bytes_per_token,
        "chars_per_token": chars_per_token,
    }


def token_length_distribution(tokenizer):
    """Return token length distribution and examples."""
    token_lengths = [len(token) for token in tokenizer.vocab]
    length_counts = Counter(token_lengths)
    examples = {length: [token for token in tokenizer.vocab if len(token) == length][:100] for length in length_counts}
    return length_counts, examples


def save_tokenizer(tokenizer, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    tokenizer.save(outfile)
