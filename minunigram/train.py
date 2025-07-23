import math
import regex as re
from collections import Counter
from collections.abc import Iterable

from .model import InternalModel
from .lattice import Lattice
from .tokenizer import UnigramTokenizer, GPT2_PRE_TOKENIZER_REGEX
from .model import UNK_PENALTY

EPSILON = 1e-7

def train_unigram_model(
    corpus: Iterable[str],
    vocab_size: int,
    pre_tokenizer_regex: str = GPT2_PRE_TOKENIZER_REGEX,
    unk_token: str = "<unk>",
    initial_vocab_size_factor: int = 10,
    max_piece_len: int = 16,
    required_chars: list[str] = None,
    verbose: bool = False,
) -> UnigramTokenizer:
    """
    Trains a Unigram model from a text corpus using Expectation-Maximization (EM).
    """
    # 1. Seed Vocabulary Initialization
    required_chars = {s: i for i, s in enumerate(required_chars)} if required_chars else {}
    if verbose:
        print("\033[1;36m🔍 Phase 1: Initializing Seed Vocabulary\033[0m")
        print(f"  📊 Target vocab size: {vocab_size}")
        print(f"  🌱 Initial size factor: {initial_vocab_size_factor}x")
        print(f"  📏 Max piece length: {max_piece_len}")
        print(f"  🧩 Base vocabulary: {len(required_chars):,} tokens")

    seed_vocab_size = vocab_size * initial_vocab_size_factor
    substring_counts = Counter()
    text_count = 0
    for text in corpus:
        text_count += 1
        pretokens = re.findall(pre_tokenizer_regex, text)
        for pretoken in pretokens:
            for i in range(len(pretoken)):
                for j in range(i + 1, min(i + 1 + max_piece_len, len(pretoken) + 1)):
                    substring_counts[pretoken[i:j]] += 1
        if verbose and text_count % 10000 == 0:
            print(f"  📝 Processed {text_count:,} texts, found {len(substring_counts):,} unique substrings")

    char_counts = Counter(c for text in corpus for c in text)
    seed_vocab = {token: count for token, count in char_counts.items()}
    for s in required_chars:
        if s not in seed_vocab:
            seed_vocab[s] = 1 # log is unhappy with 0 counts, so use 1
    
    for token, count in substring_counts.most_common():
        if len(seed_vocab) >= seed_vocab_size:
            break
        if token not in seed_vocab:
            seed_vocab[token] = count

    scores = {token: math.log(count / sum(seed_vocab.values())) for token, count in seed_vocab.items()}
    
    if verbose:
        print(f"\n\033[1;32m🚀 Starting with {len(seed_vocab):,} seed tokens\033[0m")
    
    # 2. Expectation-Maximization (EM) Loop
    num_em_steps = 4
    for i in range(num_em_steps):
        if verbose:
            print(f"\n\033[1;35m⚡ EM Iteration {i+1}/{num_em_steps}\033[0m")
        
        expected_counts = Counter()
        model = InternalModel(scores, unk_token=unk_token)
        text_count = 0
        for text in corpus:
            text_count += 1
            lattice = Lattice(text)
            model.populate_nodes(lattice)
            lattice.populate_marginal(expected_counts)
            if verbose and text_count % 10000 == 0:
                print(f"  💫 Processed {text_count:,} texts")
        
        total_expected_count = sum(expected_counts.values())
        for token in scores:
            count = expected_counts.get(token, 0)
            scores[token] = math.log((count + EPSILON) / total_expected_count)        
        # 3. Vocabulary Pruning Loop
        if len(scores) > vocab_size:
            model = InternalModel(scores, unk_token=unk_token)
            prunable_tokens = [t for t in scores if len(t) > 1]
            
            # If we can't prune enough tokens to reach target size, break out
            if len(scores) - len(prunable_tokens) > vocab_size:
                break

            # Calculate losses for prunable tokens
            losses = {}
            for token in prunable_tokens:
                # Simplified loss: frequency of the token in Viterbi paths
                freq = sum(1 for text in corpus for t, _ in model.encode_optimized(text) if t == token)
                if token in substring_counts:
                    # Boost frequency based on substring count
                    freq += substring_counts[token] / max(substring_counts.values())
                losses[token] = freq

            # Sort and prune tokens
            sorted_tokens = sorted(losses.keys(), key=lambda k: losses[k])
            num_to_prune = min(len(sorted_tokens), max(1, len(scores) - vocab_size))
            if verbose:
                print(f"  ✂️  Pruning {num_to_prune:,} tokens to reach target size")
            for i in range(num_to_prune):
                if sorted_tokens[i] in scores:
                    del scores[sorted_tokens[i]]
            if verbose:
                print(f"  📦 Current vocabulary size: {len(scores):,}")

    final_vocab = {unk_token: 0}
    final_scores = {0: min(scores.values()) - UNK_PENALTY if scores else -UNK_PENALTY}
    for token, i in required_chars.items():
        final_vocab[token] = i + 1
        final_scores[i + 1] = scores[token]

    # Create the final vocabulary mapping
    non_base_tokens = {k:v for k,v in scores.items() if k not in required_chars}
    for i, token in enumerate(sorted(non_base_tokens.keys(), key=lambda k: (-scores[k], k))):
        token_id = len(final_vocab)
        final_vocab[token] = token_id
        final_scores[token_id] = scores[token]

    tokenizer_data = {
        'metadata': {'description': 'Unigram Model trained with custom script.'},
        'vocab': final_vocab,
        'scores': final_scores,
        'pre_tokenizer_regex': pre_tokenizer_regex,
        'unk_token': unk_token
    }
    
    return UnigramTokenizer(config=tokenizer_data)
