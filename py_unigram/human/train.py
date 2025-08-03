import heapq
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from scipy.special import digamma

from .model import Lattice, Token, UnigramModel


def make_initial_vocab(
    pretokens: dict[str, int],
    required_tokens: set[str],
    num_tokens: int,
    max_token_length: int = 16,
) -> list[Token]:

    substring_freq = Counter()
    for pretoken, freq in pretokens.items():
        for i in range(len(pretoken)): # SentencePiece uses suffix array, this is simpler but more mem intensive
            for j in range(i + 1, min(len(pretoken) + 1, i + max_token_length + 1)):
                substring_freq[pretoken[i:j]] += freq

    for token in required_tokens:
        substring_freq[token] = max(substring_freq.get(token, 0), 1)  # Ensure required tokens are always included
    all_tokens = [ (freq * len(token), token) for token, freq in substring_freq.items()]
    # TODO: length 1 vs locked vs required_tokens
    selected_tokens = heapq.nlargest(
        num_tokens,
        all_tokens,
        key=lambda item: (len(item[1])==1 or item[1] in required_tokens, item[0])
    )
    log_sum_scores = math.log(sum(score for score, _ in selected_tokens))
    return [Token(text, i, math.log(score) - log_sum_scores, locked=text in required_tokens) for i, (score, text) in enumerate(selected_tokens)]

def run_e_step(
    model: UnigramModel,
    pretokens: dict[str, int],
) -> tuple[list[float], float, int]:
    """Performs the Expectation step of the EM algorithm for Unigram."""
    expected_count = defaultdict(float)
    objective = total_tokens = 0
    total_pretoken_freq = sum(freq for _, freq in pretokens.items())

    if total_pretoken_freq == 0:
        return expected_count, total_obj, total_tokens

    for pretoken, freq in pretokens.items():
        lattice = model.make_lattice(pretoken)
        z, token_prob = lattice.calc_marginal()
        if math.isnan(z): # should not happen
            print(f"      ⚠️ Warning: NaN likelihood for pretoken {pretoken} with freq={freq}.")
            continue
        for token_id, prob in token_prob.items():
            expected_count[token_id] += prob * freq
        viterbi_path, _ = lattice.viterbi()
        num_tokens_in_sentence = len(viterbi_path)
        total_tokens += num_tokens_in_sentence * freq
        objective -= (z * freq) / total_pretoken_freq

    return expected_count, objective, total_tokens

def run_m_step(
    model: UnigramModel,
    expected_counts: list[float],
    dp_smoothing: bool = True,
    k_expected_frequency_threshold = 0.5, # TODO: scale with corpus size
    verbose: bool = False,
):
    """Performs the Maximization step of the EM algorithm for Unigram.
        expected_counts: Expected frequency for each token from E-step
        dp_smoothing: If True, use digamma-based sparsity (like SentencePiece).
                      If False, use standard maximum likelihood estimation.
    """
    prev_tokens = model.tokens
    # Filter infrequent pieces.
    model.tokens = [
        t for t in prev_tokens
        if expected_counts[t.id] >= k_expected_frequency_threshold or t.locked
    ]
    
    if verbose:
        print(f"    🔍 Filtering: {len(prev_tokens) - len(model.tokens)} tokens out due to < {k_expected_frequency_threshold} expected frequency.")

    total_freq = sum(expected_counts[t.id] for t in model.tokens)
    if dp_smoothing: # SentencePiece-style: digamma transform with implicit alpha=0 for sparsity bias
        log_total = digamma(total_freq)
        for t in model.tokens:
            t.log_prob = digamma(expected_counts[t.id]) - log_total
    else: # Standard maximum likelihood estimation
        for t in model.tokens:
            t.log_prob = math.log(expected_counts[t.id] / total_freq)

def prune_tokens(
    model: UnigramModel,
    pretokens: dict[str, int],
    vocab_size: int,
    shrinking_factor: float = 0.75,
    verbose: bool = False,
) -> list[Token]:
    """
    Prunes tokens based on their importance to the model using Viterbi-based pruning.
    This is a Python port of the C++ PruneSentencePieces function from SentencePiece.
    
    Args:
        model: The UnigramModel containing current tokens
        pretokens: Dictionary of pretoken frequencies
        vocab_size: Target vocabulary size
        shrinking_factor: Factor to reduce vocabulary by (default: 0.75)
        verbose: Whether to print progress information
        
    Returns:
        List of Token objects after pruning
    """
    # Calculate target size based on vocab size and shrinking factor
    target_size = max(
        vocab_size,
        int(len(model.tokens) * shrinking_factor)
    )
    if verbose:
        print(f"    ✂️  Pruning tokens using Viterbi-based pruning from {len(model.tokens)} to {target_size} tokens...")
    
    # Initialize data structures for tracking token pruning
    always_keep = [True] * len(model.tokens)  # Whether each token must be kept
    alternatives = [[] for _ in model.tokens]  # Alternative segmentations for each token

    # First, segments the current tokens to know how each token is resegmented if removed
    # To do so, we take the second best segmentation of token[i].
    # alternatives[i] stores the sequence of second best tokens.
    for i, token in enumerate(model.tokens):
        if token.locked:  # Skip locked tokens (must be kept)
            always_keep[i] = True
            continue
            
        lattice = model.make_lattice(token.text)
        best_path, _ = lattice.viterbi(allow_single_token=True)
        
        if len(best_path) >= 2:  # Can safely remove this token if its Viterbi path is split
            always_keep[i] = False
        else:  # Try to find alternative segmentation without single-token path
            alt_path, _ = lattice.viterbi(allow_single_token=False)
            if alt_path: # Found alternative segmentation
                always_keep[i] = False
                alternatives[i] = [t.id for t in alt_path]
            else: # No alternative segmentation found, must keep
                always_keep[i] = True

    # Second, segments all sentences to compute likelihood
    # with a unigram language model. inverted[i] stores
    # the set of sentence index where the tokens[i] appears.
    token_count = [0.0] * len(model.tokens)
    inverted = [[] for _ in model.tokens]
    vsum = sum(pretokens.values())

    for pretoken, count in pretokens.items():
        lattice = model.make_lattice(pretoken)
        viterbi_path, _ = lattice.viterbi()
        for token in viterbi_path:
            token_count[token.id] += count
            inverted[token.id].append(pretoken)

    total_count = sum(token_count)
    log_total = math.log(total_count)
    candidates = []
    new_tokens = []

    # Finally, computes how likely the LM likelihood is reduced if the token[i] is removed from the vocabulary.
    # Since the exact computation of loss is difficult, we compute the loss approximately by assuming that all
    # token[i] in the sentences are replaced with alternatives[i] when token[i] is removed.
    for i, token in enumerate(model.tokens):
        if token_count[i] == 0 or not always_keep[i]:  # not found in Viterbi path. Can remove this entry safely.
            continue
        elif not alternatives[i] or token.locked: # no alternatives. Keeps this entry.
            new_tokens.append(token)
        else:  
            # The logprob with the token[i] = log(count[i] / total_count)
            logprob_token = math.log(token_count[i]) - log_total
            
            # After removing the token[i], its frequency freq[i] is re-assigned to alternatives.
            # new_sum = current_sum - freq[i] + freq[i] * alternatives[i].size()
            #         = current_sum + freq[i] * (alternatives[i] - 1)
            logsum_alt = math.log(total_count + token_count[i] * (len(alternatives[i]) - 1))
            
            # The frequencies of alternatives are increased by freq[i]
            logprob_alt = sum(
                math.log(token_count[alt_id] + token_count[i]) - logsum_alt
                for alt_id in alternatives[i]
            )
            # The frequency of token[i] = sum of pretoken freqs where token[i] appears
            token_i_freq = sum(pretokens[pretoken] for pretoken in inverted[i]) / vsum            
            # loss: the diff of likelihood after removing the token[i]
            loss = token_i_freq * (logprob_token - logprob_alt)
            candidates.append((i, loss))

    # reduce vocabulary to target_size
    candidates.sort(key=lambda x: x[1])
    for i, _ in candidates:
        if len(new_tokens) >= target_size:
            break
        new_tokens.append(model.tokens[i])

    if verbose:
        print(f"    Pruned {len(model.tokens) - len(new_tokens)} tokens -> current vocab size: {len(new_tokens)}")
    return new_tokens

def finalize_tokens(
    model: UnigramModel,
    vocab_size: int,
    verbose: bool = False,
) -> list[Token]:
    """Finalizes the vocabulary by ensuring required characters are included and keeping top pieces.
    
    This implements the same algorithm as the C++ version, ensuring required characters
    are included and keeping the highest scoring pieces up to the vocabulary size.
    
    Args:
        model: The UnigramModel containing current tokens
        vocab_size: Target vocabulary size
        verbose: Whether to print progress information
        
    Returns:
        List of Token objects representing the final vocabulary
    """
    if verbose:
        print("Finalizing vocabulary...")
        print(f"  Target vocabulary size: {vocab_size}")
    
    # Get all sentence pieces from the model
    final_tokens = {}
    
    min_score_penalty = 0.0
    MIN_SCORE_PENALTY_DELTA = 0.0001
    # add required tokens to final_tokens
    # TODO: Add penalty to avoid required pieces from having the same score?
    # TODO: check .locked etc
    for token in model.tokens:
        if token.locked:
            final_tokens[token.id] = token

    # Keep highest scoring tokens
    for token in sorted(model.tokens, key=lambda x: -x.log_prob):
        if token.id in final_tokens:
            continue
        if len(final_tokens) >= vocab_size:
            break
        final_tokens[token.id] = token
    
    if verbose:
        print(f"  Final vocabulary size: {len(final_tokens)}")
    
    return sorted(final_tokens.values(), key=lambda x: -x.log_prob)

# --- Main Training Function ---

def train_unigram(
    pretokens: dict[str, int],
    vocab_size: int = 8000,
    max_token_len: int = 16,
    initial_vocab_factor: int = 4,
    pruning_shrinking_factor: float = 0.75,
    dirichlet_alpha: float = 1.0,
    required_tokens: list[str] | None = None,
    max_iterations: int = 100,
    num_sub_iterations: int = 2,
    verbose: bool = False,
) -> UnigramModel:
    """Trains a Unigram tokenizer model."""
    if not pretokens:
        raise ValueError("Input 'pretokens' dictionary cannot be empty.")
    
    # initialize vocab and model
    required_tokens = set(required_tokens or [])
    vocab = make_initial_vocab(pretokens, required_tokens, vocab_size * initial_vocab_factor, max_token_len)
    if verbose:
        print(f"🌱 Generated {vocab_size} * {initial_vocab_factor} = {len(vocab)} initial tokens with max length {max_token_len} from {len(pretokens)} pretokens")
    model = UnigramModel(vocab)

    # EM Training Loop
    desired_vocab_size = int(vocab_size * 1.1)
    for iter in range(max_iterations):
        # Sub-EM Iterations
        for sub_iter in range(num_sub_iterations):
            if verbose:
                print(f"    🔄 EM Iteration {iter + 1}, Sub-iteration {sub_iter + 1}/{num_sub_iterations}")
            expected_count, objective, total_tokens = run_e_step(model, pretokens, verbose)
            run_m_step(model, expected_count, dirichlet_alpha, verbose)
            avg_tokens_per_piece = (1.0 * num_tokens / len(model)) if len(model) > 0 else 0
            if verbose:
                print(f"      📈 Updated model. Size: {len(model)}, Obj: {obj:.6f}, Tokens: {num_tokens}, Avg Tokens/Piece: {avg_tokens_per_piece:.2f}")

        # Check Stopping Condition
        current_size = len(model)
        if current_size <= desired_vocab_size:
            if verbose: print(f"    ✅ Desired vocab size ({desired_vocab_size}) reached or exceeded ({current_size}). Stopping EM iterations.")
            break

        # Pruning Step
        if verbose: print(f"    🔪 Pruning to shrink vocab towards {desired_vocab_size}...")
        pruned_tokens = prune_tokens(model, pretokens, vocab_size, pruning_shrinking_factor, verbose)
        model = UnigramModel(pruned_tokens)
        iteration += 1

        # Safety break
        if iteration > 100:
             if verbose: print("    ⛔ Max iterations (100) reached. Stopping.")
             break

    # Phase 4: Finalization
    if verbose: print("\n🏁 Phase 3: Finalizing Vocabulary")
    final_tokens = finalize_tokens(model, vocab_size, verbose)
    if verbose:
        print(f"  🏁 Final vocabulary size: {len(final_tokens)}")
        print("\n🎉 === Unigram Model Training Completed ===\n")
    return UnigramModel(final_tokens)
