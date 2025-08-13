# heavily vibe coded comparison script for pyunigram vs hf and sentencepiece
import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import regex as re
import sentencepiece as spm
from datasets import load_dataset
from tabulate import tabulate
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from py_unigram.pretokenize import pretokenize_corpus
from py_unigram.train import train_unigram

# Example sentences to compare tokenizers
EXAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "I love programming in Python",
    "Taylor Alison Swift is an American singer songwriter",
]


@dataclass
class VocabularyStats:
    total_size: int = 0
    one_char_tokens: int = 0
    special_tokens: int = 0
    regular_tokens: int = 0
    tokens: Set[str] = field(default_factory=set)


@dataclass
class TokenizationExample:
    original: str
    tokens: List[str]
    token_count: int


@dataclass
class TokenizerResult:
    name: str
    token_count: int
    compression_ratio: float
    vocab_stats: Optional[VocabularyStats] = None
    examples: List[TokenizationExample] = field(default_factory=list)


def get_texts(dataset_name):
    if dataset_name == "wikitext":
        return [text for text in load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"] if text.strip()]
    elif dataset_name == "swift":
        with open("./tests/data/swift_clean.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def train_hf_tokenizer(texts, vocab_size=20000) -> TokenizerResult:
    """Train and test a Hugging Face Unigram tokenizer.

    Args:
        texts: List of text samples to test compression on
        vocab_size: Target vocabulary size

    Returns:
        TokenizerResult with token count and compression ratio
    """
    print("\nTraining Hugging Face Unigram Tokenizer...")
    tokenizer = Tokenizer(models.Unigram(byte_fallback=False))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="\u2581",
        prepend_scheme="always",
    )
    tokenizer.decoder = decoders.Metaspace(
        replacement="\u2581",
        prepend_scheme="always",
    )
    # tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # tokenizer.decoder = decoders.ByteLevel()
    # tokenizer.post_processor = processors.ByteLevel()
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    # Train the tokenizer
    tokenizer.train_from_iterator((text for text in texts if text.strip()), trainer=trainer)

    # Get vocabulary statistics
    vocab = tokenizer.get_vocab()
    special_tokens = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}

    # Get all tokens by decoding their IDs
    tokens = {token: tokenizer.decode([token_id]) for token, token_id in vocab.items()}

    vocab_stats = VocabularyStats(
        total_size=len(vocab),
        one_char_tokens=sum(1 for token in tokens.values() if len(token) == 1),
        tokens=set(tokens.values()),
    )

    # Count special tokens
    vocab_stats.special_tokens = sum(1 for t in tokens.values() if t in special_tokens)
    vocab_stats.regular_tokens = len(vocab) - vocab_stats.special_tokens

    # Calculate token counts and compression
    total_bytes = 0
    total_tokens = 0

    for text in texts:
        if not text.strip():
            continue
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(tokenizer.encode(text).ids)

    compression_ratio = total_bytes / total_tokens if total_tokens > 0 else 0

    # Tokenize example sentences
    examples = []
    for sentence in EXAMPLE_SENTENCES:
        encoding = tokenizer.encode(sentence)
        examples.append(TokenizationExample(original=sentence, tokens=encoding.tokens, token_count=len(encoding.ids)))

    return TokenizerResult(
        name="* Hugging Face Unigram",
        token_count=total_tokens,
        compression_ratio=compression_ratio,
        vocab_stats=vocab_stats,
        examples=examples,
    )


def train_sentencepiece_tokenizer(texts, vocab_size=20000) -> TokenizerResult:
    """Train and test a SentencePiece Unigram tokenizer.

    Args:
        texts: List of text samples to test compression on
        vocab_size: Target vocabulary size

    Returns:
        TokenizerResult with token count and compression ratio
    """
    print("\nTraining SentencePiece Unigram Tokenizer...")
    corpus_file = "sp_corpus.txt"
    model_prefix = "sp_unigram"

    # Write corpus to file
    with open(corpus_file, "w", encoding="utf-8") as f:
        for text in texts:
            if text.strip():
                f.write(text + "\n")

    # Train the model
    spm.SentencePieceTrainer.train(
        f"--input={corpus_file} --model_prefix={model_prefix} --vocab_size={vocab_size} "
        "--model_type=unigram --character_coverage=1.0 --byte_fallback=false "
        "--normalization_rule_name=identity --remove_extra_whitespaces=false --add_dummy_prefix=true "
        "--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --minloglevel=1"
    )

    # Load the trained model
    sp_processor = spm.SentencePieceProcessor()
    sp_processor.load(f"{model_prefix}.model")

    # Get vocabulary statistics
    special_tokens = {"<unk>", "<s>", "</s>", "<pad>"}
    vocab_stats = VocabularyStats(total_size=sp_processor.GetPieceSize(), one_char_tokens=0, tokens=set())

    # Process all tokens
    for i in range(sp_processor.GetPieceSize()):
        token = sp_processor.IdToPiece(i)
        vocab_stats.tokens.add(token.replace("\u2581", " "))

        # Check for special tokens
        if token in special_tokens or (token.startswith("<") and token.endswith(">")):
            vocab_stats.special_tokens += 1

        # Check for single character tokens
        if len(token) == 1:
            vocab_stats.one_char_tokens += 1

    vocab_stats.regular_tokens = vocab_stats.total_size - vocab_stats.special_tokens

    # Calculate token counts and compression
    total_bytes = 0
    total_tokens = 0

    for text in texts:
        if not text.strip():
            continue
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(sp_processor.encode_as_ids(text))

    compression_ratio = total_bytes / total_tokens if total_tokens > 0 else 0

    # Tokenize example sentences
    examples = []
    for sentence in EXAMPLE_SENTENCES:
        tokens = sp_processor.encode_as_pieces(sentence)
        examples.append(TokenizationExample(original=sentence, tokens=tokens, token_count=len(tokens)))

    result = TokenizerResult(
        name="* SentencePiece Unigram",
        token_count=total_tokens,
        compression_ratio=compression_ratio,
        vocab_stats=vocab_stats,
        examples=examples,
    )

    # Clean up
    if os.path.exists(corpus_file):
        os.remove(corpus_file)
    if os.path.exists(f"{model_prefix}.model"):
        os.remove(f"{model_prefix}.model")
    if os.path.exists(f"{model_prefix}.vocab"):
        os.remove(f"{model_prefix}.vocab")

    return result


SPACES_PRETOK_REGEX = r" ?\p{L}+|\s+|[^\s\p{L}]+"
SPACES_PRETOK_PATTERN = re.compile(SPACES_PRETOK_REGEX)


def train_pyunigram_tokenizer(texts, name="PyUnigram", *, pretokenization: str = "spaces", **kwargs) -> TokenizerResult:
    """Train and test a PyUnigram tokenizer.

    Args:
        texts: List of text samples to test compression on
        vocab_size: Target vocabulary size

    Returns:
        TokenizerResult with token count and compression ratio
    """
    print("\nTraining PyUnigram...")

    # Get pretokens (text chunks with frequencies)
    if pretokenization == "spaces":
        pretokens = pretokenize_corpus(texts, regex_pattern=SPACES_PRETOK_REGEX)
    elif pretokenization == "gpt2":
        pretokens = pretokenize_corpus(texts)
    else:
        raise ValueError(f"Unknown pretokenization: {pretokenization}")
    print(f"Loaded {len(pretokens):,} unique pretokens, total {sum(pretokens.values()):,} pretokens")

    kwargs = {"vocab_size": 20000, "max_token_len": 16, "initial_vocab_factor": 4, **kwargs}
    # Train the model
    model, stats = train_unigram(pretokens=pretokens, **kwargs, verbose=True)

    # Get vocabulary statistics
    vocab_stats = VocabularyStats(
        total_size=len(model.tokens_by_id),
        one_char_tokens=sum(
            1
            for token in model.tokens_by_id.values()
            if len(token.text) == 1 or (len(token.text) == 3 and token.text.startswith("â"))
        ),  # Handle special chars
        tokens=set(token.text for token in model.tokens_by_id.values()),
    )

    # Tokenize example sentences
    examples = []
    for sentence in EXAMPLE_SENTENCES:
        token_ids = model.encode(sentence)
        token_strings = [model.tokens_by_id[id].text for id in token_ids]
        examples.append(TokenizationExample(original=sentence, tokens=token_strings, token_count=len(token_strings)))

    return TokenizerResult(
        name=name,
        token_count=stats["total_tokens"],
        compression_ratio=stats["bytes/token"],
        vocab_stats=vocab_stats,
        examples=examples,
    )


def calculate_similarity_matrix(tokenizers: List[TokenizerResult]) -> Dict[str, Dict[str, float]]:
    """Calculate Jaccard similarity matrix between all tokenizers."""
    jacard = {t.name: {t2.name: 0.0 for t2 in tokenizers if t2.name.startswith("*")} for t in tokenizers}

    for t1 in tokenizers:
        for t2 in tokenizers:
            if t2.name in jacard[t1.name]:
                intersection = len(t1.vocab_stats.tokens & t2.vocab_stats.tokens)
                union = len(t1.vocab_stats.tokens | t2.vocab_stats.tokens)
                jacard[t1.name][t2.name] = (intersection / union) * 100 if union else 0

    return jacard


def print_results(results: List[TokenizerResult], show_examples: bool = True):
    """Print the tokenizer comparison results and examples.

    Args:
        results: List of TokenizerResult objects to compare
        show_examples: Whether to show detailed tokenization examples
    """
    # Print overall statistics
    print("\n" + "=" * 80)
    print("TOKENIZER COMPARISON - OVERALL STATISTICS")
    print("=" * 80)

    # Prepare data for statistics table
    stats_data = []
    for result in sorted(results, key=lambda x: x.compression_ratio, reverse=True):
        total = result.vocab_stats.total_size
        one_char = f"{result.vocab_stats.one_char_tokens:,} ({result.vocab_stats.one_char_tokens / total:.1%})"
        special = f"{result.vocab_stats.special_tokens:,} ({result.vocab_stats.special_tokens / total:.1%})"
        regular = f"{result.vocab_stats.regular_tokens:,} ({result.vocab_stats.regular_tokens / total:.1%})"

        stats_data.append(
            [
                result.name,
                f"{result.token_count:,}",
                f"{result.compression_ratio:.4f}",
                f"{total:,}",
                one_char,
                special,
                regular,
            ]
        )

    # Print statistics table
    print(
        "\n"
        + tabulate(
            stats_data,
            headers=[
                "Tokenizer",
                "Tokens",
                "Compression",
                "Vocab Size",
                "1-Char Tokens",
                "Special Tokens",
                "Regular Tokens",
            ],
            tablefmt="grid",
            stralign="right",
            numalign="right",
        )
    )

    # Print vocabulary similarity matrix
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("VOCABULARY SIMILARITY MATRIX (Jaccard %)")
        print("=" * 80)

        jacard = calculate_similarity_matrix(results)
        similarity = [{"name": k, **cols} for k, cols in jacard.items()]
        # Print similarity table
        print("\n" + tabulate(similarity, headers="keys", tablefmt="grid", stralign="right", numalign="right"))

    # Print tokenization examples if requested
    if show_examples and results and results[0].examples:
        print("\n" + "=" * 80)
        print("TOKENIZER COMPARISON - EXAMPLE TOKENIZATIONS")
        print("=" * 80)

        # For each example sentence
        for i, sentence in enumerate(EXAMPLE_SENTENCES):
            print(f"\n\n{'=' * 40} Example {i + 1} {'=' * 40}")
            print(f"\nOriginal: {sentence}")
            print(f"Length: {len(sentence)} chars, {len(sentence.encode('utf-8'))} bytes\n")

            # Prepare data for tabulation
            example_data = []
            for result in results:
                if i < len(result.examples):
                    example = result.examples[i]
                    example_data.append(
                        [
                            result.name,
                            " ".join(str(token) for token in example.tokens) if example.tokens else "",
                            example.token_count,
                            f"{len(sentence.encode('utf-8')) / example.token_count:.2f}"
                            if example.token_count > 0
                            else "N/A",
                        ]
                    )

            if example_data:  # Only print if we have data
                # Print the tokenization table
                print(
                    "\n"
                    + tabulate(
                        example_data,
                        headers=["Tokenizer", "Tokens", "# Tokens", "Compression"],
                        tablefmt="grid",
                        maxcolwidths=[20, 60, 10, 12],
                        stralign="left",
                        numalign="right",
                    )
                )

                # Print token counts comparison
                print("\nToken counts (sorted):")
                counts_data = [
                    [r.name, r.examples[i].token_count]
                    for r in sorted(
                        [r for r in results if i < len(r.examples)], key=lambda x: x.examples[i].token_count
                    )
                ]
                print(
                    tabulate(
                        counts_data,
                        headers=["Tokenizer", "Token Count"],
                        tablefmt="simple_grid",
                        stralign="right",
                        numalign="right",
                    )
                )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare different tokenizers.")
    parser.add_argument("--no-examples", action="store_true", help="Do not show tokenization examples")
    parser.add_argument(
        "--dataset", type=str, default="swift", choices=["swift", "wikitext"], help="Dataset to use for training"
    )
    parser.add_argument("--vocab-size", type=int, default=512, help="Vocabulary size for the tokenizers")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load dataset and prepare texts
    try:
        texts = get_texts(args.dataset)
        print(f"Loaded {len(texts)} texts, total {sum(len(text) for text in texts):,} chars")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Train and evaluate each tokenizer
    tokenizers = [
        train_pyunigram_tokenizer(texts, "* PyUnigram (spaces)", vocab_size=args.vocab_size),
        train_pyunigram_tokenizer(
            texts, "PyUnigram shrink slow (spaces)", vocab_size=args.vocab_size, pruning_shrinking_factor=0.95
        ),
        train_pyunigram_tokenizer(
            texts, "PyUnigram no m-step removals (spaces)", vocab_size=args.vocab_size, m_step_low_count_threshold=0
        ),
        train_pyunigram_tokenizer(texts, "PyUnigram (gpt2)", vocab_size=args.vocab_size, pretokenization="gpt2"),
        train_hf_tokenizer(texts, args.vocab_size),
        train_sentencepiece_tokenizer(texts, args.vocab_size),
    ]

    # Print results with or without examples based on the flag
    print_results(tokenizers, show_examples=not args.no_examples)


if __name__ == "__main__":
    main()
