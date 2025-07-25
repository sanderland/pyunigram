import math
import regex as re
from collections import Counter
from collections.abc import Iterable
from scipy.special import digamma # Required for the faithful M-step

from .model import InternalModel
from .lattice import Lattice
from .tokenizer import GeminiUnigramTokenizer, GPT2_PRE_TOKENIZER_REGEX
from .model import UNK_PENALTY

def train_unigram_model(
    corpus: Iterable[str],
    vocab_size: int,
    pre_tokenizer_regex: str = GPT2_PRE_TOKENIZER_REGEX,
    unk_token: str = "<unk>",
    initial_vocab_size_factor: int = 4,
    max_piece_len: int = 16,
    pruning_shrinking_factor: float = 0.75,
    num_em_sub_iterations: int = 2,
    dirichlet_alpha: float = 1.0,
    required_chars: list[str] = None,
    verbose: bool = False,
) -> GeminiUnigramTokenizer:
    """
    A port of the SentencePiece Unigram model training logic, incorporating the VB/digamma M-step and sub-EM iterations.
    """
    pruning_percentage: float = 1 - pruning_shrinking_factor

    if verbose: print("\033[1;36mPhase 1: Pre-tokenization and Substring Counting\033[0m")

    pretokenizer = re.compile(pre_tokenizer_regex)
    pretokens = Counter(pt for text in corpus for pt in pretokenizer.findall(text))
    if verbose: print(f"  - Found {len(pretokens):,} unique and {sum(pretokens.values()):,} total pretokens, total length {sum(len(pt) * count for pt, count in pretokens.items()):,} characters.")
    del corpus

    substring_counts = Counter()
    for pretoken, count in pretokens.items():
        for i in range(len(pretoken)):
            for j in range(i + 1, min(i + 1 + max_piece_len, len(pretoken) + 1)):
                substring_counts[pretoken[i:j]] += count

    seed_vocab_size = int(vocab_size * initial_vocab_size_factor)
    scores = {token: math.log(count) for token, count in substring_counts.most_common(seed_vocab_size)}
    
    protected_tokens = set(required_chars or [])
    char_counts = Counter(c for pretoken in pretokens for c in pretoken)
    for char, count in char_counts.items():
        protected_tokens.add(char)
        if char not in scores: scores[char] = math.log(count)

    if verbose: print(f"\n\033[1;32mPhase 2: Starting with {len(scores):,} seed tokens. Target: {vocab_size}\033[0m")
    
    # --- MAIN TRAINING LOOP ---
    while len(scores) > vocab_size:
        # The E-M steps are run multiple times between each pruning step.
        for sub_iter in range(num_em_sub_iterations):
            if verbose: print(f"\n\033[1;35mEM Sub-Iteration {sub_iter + 1}/{num_em_sub_iterations} on {len(scores):,} tokens...\033[0m")
            
            # E-Step: Calculate expected frequencies
            expected_counts = Counter()
            model = InternalModel(scores, unk_token=unk_token)
            for pretoken, count in pretokens.items():
                lattice = Lattice(pretoken)
                model.populate_nodes(lattice)
                lattice.populate_marginal(expected_counts, count)

            # M-Step: Update scores using Variational Bayes with digamma.
            dead_tokens = {token for token, count in expected_counts.items() if count < 0.5 and token not in protected_tokens}
            if dead_tokens and sub_iter > 0:  # Only prune after the first sub-iteration
                if verbose: print(f"  - Removing {len(dead_tokens)} dead tokens with expected count < 0.5")
                for token in dead_tokens:
                    if token in scores: del scores[token]
            else:
                if verbose: print(f"  - No dead tokens to prune, lowest expected count is {min(expected_counts.values()):.2f}")

            # This is the Variational Bayes update rule.
            # Bayesianified/DPified EM algorithm: https://cs.stanford.edu/~pliang/papers/tutorial-acl2007-talk.pdf            
            total_expected_count = sum(expected_counts.values())
            log_total = digamma(total_expected_count + len(scores) * dirichlet_alpha)
            for token in scores:
                scores[token] = digamma(expected_counts[token] + dirichlet_alpha) - log_total
        
        # --- Pruning Step (after all sub-iterations are complete) ---
        if verbose: print(f"\n\033[1;36mCalculating loss to prune from {len(scores):,} to {vocab_size}...\033[0m")
        
        prunable_tokens = [token for token in scores if token not in protected_tokens]
        if not prunable_tokens: break

        losses = {}
        model = InternalModel(scores, unk_token=unk_token)
        for token_to_prune in prunable_tokens:
            scores_without_token = scores.copy(); del scores_without_token[token_to_prune]
            model_without_token = InternalModel(scores_without_token, unk_token=unk_token)
            loss = 0
            for pretoken, count in pretokens.items():
                if token_to_prune in pretoken:
                    _, _, score_with = model.encode_optimized(pretoken)
                    _, _, score_without = model_without_token.encode_optimized(pretoken)
                    loss += (score_with - score_without) * count
            losses[token_to_prune] = loss

        num_to_prune = max(1, min(len(scores) - vocab_size, int(len(scores) * pruning_percentage)))
        num_to_prune = min(num_to_prune, len(losses))
        
        sorted_tokens_by_loss = sorted(losses.keys(), key=lambda k: losses[k])
        
        if verbose: print(f"  - Pruning {num_to_prune} tokens with the lowest loss...")
        
        for i in range(num_to_prune):
            if sorted_tokens_by_loss[i] in scores:
                del scores[sorted_tokens_by_loss[i]]

    if verbose: print(f"\n\033[1;32mPhase 3: Finalizing Vocabulary\033[0m")
    
    # Finalize vocabulary and scores
    final_vocab = {unk_token: 0}
    final_scores = {0: -30.0}
    
    next_id = 1
    for token in sorted(list(protected_tokens)):
        if token in scores:
            final_vocab[token] = next_id
            final_scores[next_id] = scores.pop(token)
            next_id += 1

    for token in sorted(scores.keys(), key=lambda k: (-scores[k], k)):
        final_vocab[token] = next_id
        final_scores[next_id] = scores[token]
        next_id += 1

    tokenizer_data = {
        'metadata': {'settings': dict(pruning_percentage=pruning_percentage, vocab_size=vocab_size, initial_vocab_size_factor=initial_vocab_size_factor, max_piece_len=max_piece_len, num_em_sub_iterations=num_em_sub_iterations, dirichlet_alpha=dirichlet_alpha)},
        'vocab': final_vocab,
        'scores': final_scores,
        'pre_tokenizer_regex': pre_tokenizer_regex,
        'unk_token': unk_token,
    }
    
    return UnigramTokenizer(config=tokenizer_data)