import os, sys
from datasets import load_dataset
from minunigram import train_unigram_model

def main(dataset_name="eng_latn_300mb", vocab_size=8000):
    # Load the dataset
    print("  🔄 Processing dataset...")
    if dataset_name == "eng_latn_300mb":
        print("\033[1;34m📚 Loading English dataset...\033[0m")
        corpus_name = "eng_latn_300mb"
        dataset = load_dataset(
            "sanderland/monolingual-tokenizer-data",
            data_files=[f"{corpus_name}.txt"],
            split="train",
            streaming=True,
        )
        texts = [text for item in dataset for text in item['text'].split('\n') if text.strip()]
        del dataset
    elif dataset_name == "swift":
        print("Loading taylor swift wiki page")
        with open('../script_bpe/tests/data/taylorswift.txt', 'r', encoding='utf-8') as f:
            texts = [f.read()]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    print(f"  ✨ Successfully loaded dataset")
    
    # Train tokenizer
    print("\n\033[1;32m🎯 Training Unigram Tokenizer\033[0m")
    tokenizer = train_unigram_model(
        corpus=texts,
        vocab_size=vocab_size,
        initial_vocab_size_factor=4,  # Start with 4x the target size
        max_piece_len=16,
        verbose=True,  # Enable detailed progress logging
    )
    
    # Evaluate compression
    print("\n\033[1;36m📊 Evaluating Compression Metrics\033[0m")
    total_bytes = 0
    total_tokens = 0
    total_chars = 0
    
    # Sample some texts for evaluation
    eval_texts = texts[:1000]  # Use first 1000 texts
    print(f"  🔍 Analyzing {len(eval_texts):,} sample texts...")
    
    for text in eval_texts:
        text_bytes = len(text.encode('utf-8'))
        tokens = tokenizer.encode(text)
        total_bytes += text_bytes
        total_tokens += len(tokens)
        total_chars += len(text)
    
    bytes_per_token = total_bytes / total_tokens
    chars_per_token = total_chars / total_tokens
    
    print("\n\033[1;33m📈 Tokenization Statistics:\033[0m")
    print(f"  📦 Vocabulary size: \033[1m{len(tokenizer.vocab):,}\033[0m tokens")
    print(f"  💾 Total bytes processed: \033[1m{total_bytes:,}\033[0m")
    print(f"  🔤 Total tokens generated: \033[1m{total_tokens:,}\033[0m")
    print(f"  📝 Total characters: \033[1m{total_chars:,}\033[0m")

    # Print stats on token lengths in vocab
    print("\n\033[1;34m🔢 Token Length Distribution in Vocabulary:\033[0m")
    from collections import Counter
    token_lengths = [len(token) for token in tokenizer.vocab]
    length_counts = Counter(token_lengths)
    for length in sorted(length_counts):
        print(f"  Length {length}: \033[1m{length_counts[length]:,}\033[0m tokens")
        # Print first 100 examples of this length
        examples = [token for token in tokenizer.vocab if len(token) == length][:100]
        if examples:
            print(f"    Examples: {', '.join(repr(e) for e in examples)}")

    print("\n\033[1;32m🎯 Efficiency Metrics:\033[0m")
    print(f"  ⚡ Bytes per token: \033[1m{bytes_per_token:.4f}\033[0m")
    print(f"  📏 Characters per token: \033[1m{chars_per_token:.4f}\033[0m")
    
    # Save the tokenizer
    print("\n\033[1;35m💾 Saving Tokenizer Model\033[0m")
    os.makedirs("models", exist_ok=True)
    outfile = f"models/{dataset_name}_{vocab_size}.json"
    tokenizer.save(outfile)
    print(f"\033[1;32m✨ Done! Tokenizer saved to {outfile}\033[0m")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
