# 1. Install necessary libraries
!pip install datasets tokenizers sentencepiece

import os
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import sentencepiece as spm

# 2. Load a Hugging Face Dataset
# We'll use a small subset of the wikitext dataset for demonstration purposes.
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Create an iterator for the dataset
def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

# --- Hugging Face Tokenizers ---

# 3. Train a Hugging Face Unigram Tokenizer
print("Training Hugging Face Unigram Tokenizer...")

# Instantiate a tokenizer with a Unigram model
hf_tokenizer = Tokenizer(models.Unigram())

# Set the pre-tokenizer to ByteLevel, which is used by GPT-2
hf_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

# Instantiate a trainer
# We set a vocabulary size and define the special tokens
trainer = trainers.UnigramTrainer(
    vocab_size=20000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Train the tokenizer
hf_tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# Save the tokenizer
hf_tokenizer_path = "hf_unigram_tokenizer.json"
hf_tokenizer.save(hf_tokenizer_path)
print(f"Hugging Face tokenizer trained and saved to {hf_tokenizer_path}")


# --- SentencePiece ---

# 4. Train a SentencePiece Unigram Tokenizer
print("\nTraining SentencePiece Unigram Tokenizer...")

# SentencePiece trains from a file, so we'll write our dataset to a text file.
corpus_file = "wikitext_corpus.txt"
with open(corpus_file, "w", encoding="utf-8") as f:
    for text_batch in get_training_corpus():
        for text in text_batch:
            f.write(text + "\n")

# Train the SentencePiece model
spm.SentencePieceTrainer.train(
    f'--input={corpus_file} --model_prefix=sp_unigram --vocab_size=20000 '
    '--model_type=unigram '
    '--character_coverage=1.0 '
    '--byte_fallback=true ' # This helps in handling any byte sequence, similar to ByteLevel
    '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
    '--pad_piece=[PAD] --unk_piece=[UNK] --bos_piece=[CLS] --eos_piece=[SEP]'
)

sp_model_path = "sp_unigram.model"
print(f"SentencePiece tokenizer trained and model saved to {sp_model_path}")


# --- Compression Test ---

# 5. Compare Compression
print("\n--- Compression Test ---")

# Load the trained tokenizers
hf_tokenizer = Tokenizer.from_file(hf_tokenizer_path)
sp = spm.SentencePieceProcessor()
sp.load(sp_model_path)

# Sample text for testing
test_text = "This is a sample sentence to test the compression of the two tokenizers. They should behave similarly due to the same pre-tokenization."

# Tokenize with Hugging Face Tokenizer
hf_tokens = hf_tokenizer.encode(test_text).tokens
hf_token_ids = hf_tokenizer.encode(test_text).ids

# Tokenize with SentencePiece
sp_tokens = sp.encode_as_pieces(test_text)
sp_token_ids = sp.encode_as_ids(test_text)

# Calculate compression ratio (bytes per token)
original_bytes = len(test_text.encode('utf-8'))
hf_compression_ratio = original_bytes / len(hf_token_ids)
sp_compression_ratio = original_bytes / len(sp_token_ids)

# Print results
print(f"\nOriginal Text: '{test_text}'")
print(f"Original Text Length (bytes): {original_bytes}\n")

print("--- Hugging Face Tokenizer ---")
print(f"Tokens: {hf_tokens}")
print(f"Number of tokens: {len(hf_tokens)}")
print(f"Compression Ratio (bytes/token): {hf_compression_ratio:.2f}\n")


print("--- SentencePiece Tokenizer ---")
print(f"Tokens: {sp_tokens}")
print(f"Number of tokens: {len(sp_tokens)}")
print(f"Compression Ratio (bytes/token): {sp_compression_ratio:.2f}")

# Clean up the created file
os.remove(corpus_file)