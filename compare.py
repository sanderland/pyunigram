import os
import json
import sentencepiece as spm
from dataclasses import dataclass
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import sys
from py_unigram.pretokenize import pretokenize_corpus


# Add the project root to Python path to enable local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import human tokenizer implementation
from py_unigram.human.train import train_unigram
from py_unigram.human.model import UnigramModel, Token

@dataclass
class TokenizerResult:
    name: str
    token_count: int
    compression_ratio: float

def load_wikitext_dataset():
    """Load and return the wikitext dataset."""
    print("Loading wikitext training corpus...")
    return load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def get_training_texts(dataset):
    """Get all training texts as a list of strings."""
    return [text for text in dataset["text"] if text.strip()]


def train_hf_tokenizer(dataset, texts, vocab_size=20000) -> TokenizerResult:
    """Train and test a Hugging Face Unigram tokenizer.
    
    Args:
        dataset: The training dataset (unused, kept for API consistency)
        texts: List of text samples to test compression on
        vocab_size: Target vocabulary size
        
    Returns:
        TokenizerResult with token count and compression ratio
    """
    print("\nTraining Hugging Face Unigram Tokenizer...")
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    # Train the tokenizer
    tokenizer.train_from_iterator(
        (text for text in texts if text.strip()),
        trainer=trainer
    )
    
    # Calculate token counts and compression
    total_bytes = 0
    total_tokens = 0
    
    for text in texts:
        if not text.strip():
            continue
        total_bytes += len(text.encode('utf-8'))
        total_tokens += len(tokenizer.encode(text).ids)
    
    compression_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    
    return TokenizerResult(
        name="Hugging Face Unigram",
        token_count=total_tokens,
        compression_ratio=compression_ratio
    )

def train_sentencepiece_tokenizer(dataset, texts, vocab_size=20000) -> TokenizerResult:
    """Train and test a SentencePiece Unigram tokenizer.
    
    Args:
        dataset: The training dataset (unused, kept for API consistency)
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
        f'--input={corpus_file} --model_prefix={model_prefix} --vocab_size={vocab_size} '
        '--model_type=unigram --character_coverage=1.0 --byte_fallback=true '
        '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --minloglevel=1'
    )
    
    # Load the trained model
    sp_processor = spm.SentencePieceProcessor()
    sp_processor.load(f"{model_prefix}.model")
    
    # Calculate token counts and compression
    total_bytes = 0
    total_tokens = 0
    
    for text in texts:
        if not text.strip():
            continue
        total_bytes += len(text.encode('utf-8'))
        total_tokens += len(sp_processor.encode_as_ids(text))
    
    compression_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    
    # Clean up temporary files
    for ext in ['.model', '.vocab', '_corpus.txt']:
        try:
            os.remove(f"{model_prefix}{ext}")
        except FileNotFoundError:
            pass
    
    return TokenizerResult(
        name="SentencePiece Unigram",
        token_count=total_tokens,
        compression_ratio=compression_ratio
    )

def train_human_tokenizer(dataset, texts, vocab_size=20000) -> TokenizerResult:
    """Train and test a human-implemented Unigram tokenizer.
    
    Args:
        dataset: The training dataset (for pretokens)
        texts: List of text samples to test compression on
        vocab_size: Target vocabulary size
        
    Returns:
        TokenizerResult with token count and compression ratio
    """
    print("\nTraining Human Unigram Tokenizer...")
    
    # Get pretokens (text chunks with frequencies)
    pretokens = pretokenize_corpus(dataset)
    print(f"Loaded {len(pretokens):,} unique pretokens, total {sum(pretokens.values()):,} pretokens")
    
    # Define required characters (all unique chars in the dataset)
    required_chars = list(set("".join(pretokens.keys())))
    print(f"Found {len(required_chars)} required characters")
    
    # Train the model
    model = train_unigram(
        pretokens=pretokens,
        vocab_size=vocab_size,
        max_token_len=16,
        initial_vocab_factor=4,
        required_chars=required_chars,
        verbose=True
    )
    
    # Calculate token counts and compression
    total_bytes = 0
    total_tokens = 0
    
    for text in texts:
        if not text.strip():
            continue
        total_bytes += len(text.encode('utf-8'))
        total_tokens += len(model.encode(text))
    
    compression_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    
    return TokenizerResult(
        name="Human Unigram",
        token_count=total_tokens,
        compression_ratio=compression_ratio
    )

def print_results(results: List[TokenizerResult]):
    """Print the tokenizer comparison results."""
    print("\n--- Tokenizer Comparison Results ---")
    print(f"{'Tokenizer':<25} {'Tokens':>12} {'Compression':>12}")
    print("-" * 50)
    
    for result in sorted(results, key=lambda x: x.compression_ratio, reverse=True):
        print(f"{result.name:<25} {result.token_count:>12,} {result.compression_ratio:>12.2f}")
    
    print("-" * 50)

def main():
    """Main function to run the tokenizer comparison."""
    # Configuration
    vocab_size = 20000
    
    # Load dataset and prepare texts
    dataset = load_wikitext_dataset()
    texts = get_training_texts(dataset)[:10]
    
    print(f"Loaded {len(texts)} texts, total {sum(len(text) for text in texts):,} chars")

    # Train and evaluate each tokenizer
    tokenizers = [
        train_human_tokenizer(dataset, texts, vocab_size),
        train_hf_tokenizer(dataset, texts, vocab_size),
        train_sentencepiece_tokenizer(dataset, texts, vocab_size),
    ]
    
    # Print results
    print_results(tokenizers)

if __name__ == "__main__":
    main()
