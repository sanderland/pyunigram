import os
from py_unigram.gemini import train_unigram_model
from collections import Counter
from datasets import load_dataset

def load_texts(dataset_name):
    """Load texts from the specified dataset."""
    if dataset_name.endswith('300mb'):
        dataset = load_dataset(
            "sanderland/monolingual-tokenizer-data",
            data_files=[f"{dataset_name}.txt"],
            split="train",
            streaming=True,
        )
        texts = [text for item in dataset for text in item['text']]
        breakpoint()  # For debugging purposes
    elif dataset_name == "swift":
        with open('../script_bpe/tests/data/taylorswift.txt', 'r', encoding='utf-8') as f:
            texts = [f.read()]
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [item['text'] for item in dataset]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return texts

def train_tokenizer(texts, vocab_size, **kwargs):
    """Train a unigram tokenizer."""
    tokens = train_unigram_model(
        corpus=texts,
        vocab_size=vocab_size,
        **kwargs,
    )
    return UnigramTokenizer(tokens)

def compute_compression_stats(tokenizer, texts):
    """Compute compression statistics for a tokenizer and texts."""
    total_bytes = 0
    total_tokens = 0
    total_chars = 0
    for text in texts:
        text_bytes = len(text.encode('utf-8'))
        tokens = tokenizer.encode(text)
        total_bytes += text_bytes
        total_tokens += len(tokens)
        total_chars += len(text)
    bytes_per_token = total_bytes / total_tokens if total_tokens else 0
    chars_per_token = total_chars / total_tokens if total_tokens else 0
    return {
        'total_bytes': total_bytes,
        'total_tokens': total_tokens,
        'total_chars': total_chars,
        'bytes_per_token': bytes_per_token,
        'chars_per_token': chars_per_token,
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
