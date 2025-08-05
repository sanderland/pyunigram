import sys

from utils import compute_compression_stats, load_texts, save_tokenizer, token_length_distribution, train_tokenizer


def main(dataset_name="eng_latn_300mb", vocab_size=8000):
    print("  🔄 Processing dataset...")
    texts = load_texts(dataset_name)
    print("  ✨ Successfully loaded dataset")

    print("\n\033[1;32m🎯 Training Unigram Tokenizer\033[0m")
    tokenizer = train_tokenizer(texts, vocab_size, verbose=True)

    print("\n\033[1;36m📊 Evaluating Compression Metrics\033[0m")
    eval_texts = texts[:1000]
    print(f"  🔍 Analyzing {len(eval_texts):,} sample texts...")
    stats = compute_compression_stats(tokenizer, eval_texts)

    print("\n\033[1;33m📈 Tokenization Statistics:\033[0m")
    print(f"  📦 Vocabulary size: \033[1m{len(tokenizer):,}\033[0m tokens")
    print(f"  💾 Total bytes processed: \033[1m{stats['total_bytes']:,}\033[0m")
    print(f"  🔤 Total tokens generated: \033[1m{stats['total_tokens']:,}\033[0m")
    print(f"  📝 Total characters: \033[1m{stats['total_chars']:,}\033[0m")

    print("\n\033[1;34m🔢 Token Length Distribution in Vocabulary:\033[0m")
    length_counts, examples = token_length_distribution(tokenizer)
    for length in sorted(length_counts):
        print(f"  Length {length}: \033[1m{length_counts[length]:,}\033[0m tokens")
        if examples[length]:
            print(f"    Examples: {', '.join(repr(e) for e in examples[length])}")

    print("\n\033[1;32m🎯 Efficiency Metrics:\033[0m")
    print(f"  ⚡ Bytes per token: \033[1m{stats['bytes_per_token']:.4f}\033[0m")
    print(f"  📏 Characters per token: \033[1m{stats['chars_per_token']:.4f}\033[0m")

    print("\n\033[1;35m💾 Saving Tokenizer Model\033[0m")
    outfile = f"models/{dataset_name}_{vocab_size}.json"
    save_tokenizer(tokenizer, outfile)
    print(f"\033[1;32m✨ Done! Tokenizer saved to {outfile}\033[0m")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
