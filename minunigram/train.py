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
    initial_vocab_size_factor: int = 4,
    max_piece_len: int = 16,
    pruning_percentage: float = 0.10,
    required_chars: list[str] = None,
    verbose: bool = False,
) -> UnigramTokenizer:
    """
    A faithful port of the SentencePiece Unigram model training logic.
    This implementation prioritizes correctness and faithfulness to the original C++
    trainer's methodology over simplified heuristics.
    """
    if verbose:
        print("\033[1;36m🔍 Phase 1: Pre-tokenization and Substring Counting\033[0m")

    pretokenizer = re.compile(pre_tokenizer_regex)
    pretokens = Counter(pt for text in corpus for pt in pretokenizer.findall(text))
    del corpus # Free memory

    if verbose:
        print(f"  📊 Found {len(pretokens):,} unique pretokens.")

    substring_counts = Counter()
    for pretoken, count in pretokens.items():
        for i in range(len(pretoken)):
            for j in range(i + 1, min(i + 1 + max_piece_len, len(pretoken) + 1)):
                substring_counts[pretoken[i:j]] += count

    seed_vocab_size = vocab_size * initial_vocab_size_factor
    seed_vocab = {token: count for token, count in substring_counts.most_common(seed_vocab_size)}
    
    # Ensure all required characters are present.
    char_counts = Counter(c for pretoken in pretokens for c in pretoken)
    for char, count in char_counts.items():
        if char not in seed_vocab: seed_vocab[char] = count
    if required_chars:
        for char in required_chars:
            if char not in seed_vocab: seed_vocab[char] = 1

    scores = {token: math.log(count / sum(seed_vocab.values())) for token, count in seed_vocab.items()}
    
    if verbose:
        print(f"\n\033[1;32m🌱 Initialized with {len(scores):,} seed tokens. Target: {vocab_size}\033[0m")
    
    # Main EM and Pruning Loop
    # This structure is faithful to the original, where EM and pruning happen
    # in each iteration until the target vocab size is reached.
    while len(scores) > vocab_size:
        # --- E-M Step (Expectation-Maximization) ---
        if verbose: print(f"\n\033[1;35m⚡ Performing EM on {len(scores):,} tokens...\033[0m")
        
        expected_counts = Counter()
        model = InternalModel(scores, unk_token=unk_token)
        for pretoken, count in pretokens.items():
            lattice = Lattice(pretoken)
            model.populate_nodes(lattice)
            lattice.populate_marginal(expected_counts, count)
        
        total_expected_count = sum(expected_counts.values())
        if total_expected_count == 0: break
        for token in scores:
            scores[token] = math.log((expected_counts.get(token, 0) + EPSILON) / total_expected_count)
        
        # --- Pruning Step ---
        # Tokens required by the user or single-character tokens are protected from pruning.
        protected_tokens = set(required_chars or []) | {token for token in scores if len(token) == 1}
        prunable_tokens = [token for token in scores if token not in protected_tokens]

        # Calculate loss for each prunable token.
        # Loss is defined as the drop in total corpus log-likelihood if the token were removed.
        # This is a faithful, readable implementation of that concept.
        if verbose: print(f"  📉 Calculating loss for {len(prunable_tokens):,} prunable tokens...")
        
        losses = {}
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

        # Determine the number of tokens to prune using a percentage of the excess.
        num_excess_tokens = len(scores) - vocab_size
        num_to_prune =  max(1, min(num_excess_tokens, int(len(scores) * pruning_percentage)))
        
        # Prune the tokens with the lowest loss (i.e., the least useful ones).
        sorted_tokens_by_loss = sorted(losses.keys(), key=lambda k: losses[k])
        
        for i in range(num_to_prune):
            if sorted_tokens_by_loss[i] in scores:
                del scores[sorted_tokens_by_loss[i]]
        if verbose: print(f"  ✂️  Pruned {num_to_prune} tokens with the lowest loss -> {len(scores):,} remaining.")

    # Finalize vocabulary and scores
    final_vocab = {unk_token: 0}
    final_scores = {0: min(scores.values()) - UNK_PENALTY if scores else -UNK_PENALTY}
    
    next_id = 1
    # Add required characters first to ensure they get low IDs if desired.
    if required_chars:
        for char in sorted(list(set(required_chars))):
            if char in scores:
                final_vocab[char] = next_id
                final_scores[next_id] = scores.pop(char)
                next_id += 1

    # Add remaining tokens sorted by score
    for token in sorted(scores.keys(), key=lambda k: (-scores[k], k)):
        final_vocab[token] = next_id
        final_scores[next_id] = scores[token]
        next_id += 1

    tokenizer_data = {
        'metadata': {'description': 'Unigram Model trained with custom script.'},
        'vocab': final_vocab,
        'scores': final_scores,
        'pre_tokenizer_regex': pre_tokenizer_regex,
        'unk_token': unk_token,
    }
    
    return UnigramTokenizer(config=tokenizer_data)