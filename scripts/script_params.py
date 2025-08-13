import sys

import numpy as np
from utils import compute_compression_stats, load_texts, train_tokenizer


def test_pruning_influence(dataset_name="eng_latn_300mb", vocab_size=8000, pruning_percentages=None, sample_size=1000):
    if pruning_percentages is None:
        pruning_percentages = np.linspace(0.025, 0.5, 5)  # 2.5% to 50% pruning
    texts = load_texts(dataset_name)
    eval_texts = texts[:sample_size]
    results = []
    for i, pruning in enumerate(pruning_percentages):
        print(f"\n\033[1;32mTesting pruning_percentage={pruning:.3f}\033[0m")
        tokenizer = train_tokenizer(
            texts,
            vocab_size,
            pruning_shrinking_factor=1 - pruning,
            verbose=(i == 0),  # Only verbose for the first iteration,
        )
        stats = compute_compression_stats(tokenizer, eval_texts)
        results.append(
            {
                "pruning_percentage": pruning,
                "vocab_size": len(tokenizer.vocab),
                "bytes_per_token": stats["bytes_per_token"],
                "chars_per_token": stats["chars_per_token"],
            }
        )
        print(
            f"  Vocab size: {len(tokenizer.vocab)}, Bytes/token: {stats['bytes_per_token']:.4f}, Chars/token: {stats['chars_per_token']:.4f}"
        )
    return results


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "eng_latn_300mb"
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    test_pruning_influence(dataset, vocab_size)
