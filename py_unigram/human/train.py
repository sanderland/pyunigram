import heapq
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from scipy.special import digamma

from .model import Lattice, Token, UnigramModel


def make_initial_vocab(
    pretokens: dict[str, int],
    required_chars: set[str],
    num_tokens: int,
    max_token_length: int = 16,
) -> list[Token]:

    substring_freq = Counter()
    for pretoken, freq in pretokens.items():
        for i in range(len(pretoken)): # SentencePiece uses suffix array, this is simpler but more mem intensive
            for j in range(i + 1, min(len(pretoken) + 1, i + max_token_length + 1)):
                substring_freq[pretoken[i:j]] += freq

    for char in required_chars:
        substring_freq[char] = max(substring_freq.get(char, 0), 1)  # Ensure required chars are always included
    all_tokens = [ (freq * len(piece), piece) for piece, freq in substring_freq.items()]

    # force length 1 tokens to be included
    selected_tokens = heapq.nlargest(
        num_tokens,
        all_tokens,
        key=lambda item: (item[0] + 1e15 * (len(item[1])==1), item[1])
    )
    log_sum_scores = math.log(sum(score for score, _ in selected_tokens))
    return [Token(text,               i,              math.log(score) - log_sum_scores) for i, (score, text) in enumerate(selected_tokens)]

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

def prune_pieces(
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
    if verbose:
        print("    ✂️  Pruning tokens using Viterbi-based pruning...")
    
    # Get current tokens from the model
    current_tokens = model.tokens
    if not current_tokens or vocab_size == 0:
        return current_tokens

    # Calculate target size based on vocab size and shrinking factor
    target_size = min(
        vocab_size,
        int(len(current_tokens) * shrinking_factor)
    )
    
    if verbose:
        print(f"    Pruning from {len(current_tokens)} to {target_size} tokens...")

    if len(current_tokens) <= target_size:
        if verbose:
            print(f"    ✓ No pruning needed (current: {len(current_tokens)}, target: {target_size})")
        return current_tokens

    # Initialize data structures for tracking token pruning
    always_keep = [True] * len(current_tokens)  # Whether each token must be kept
    alternatives = [[] for _ in current_tokens]  # Alternative segmentations for each token

    # First, segments the current tokens to know how each token is resegmented if removed
    # To do so, we take the second best segmentation of token[i].
    # alternatives[i] stores the sequence of second best tokens.
    for i, token in enumerate(current_tokens):
        if token.locked:  # Skip locked tokens (must be kept)
            always_keep[i] = True
            continue
            
        lattice = model.make_lattice(token.text)
        
        # Get best path
        best_path, _ = lattice.viterbi(allow_single_token=True)
        
        if len(best_path) >= 2:
            # Can safely remove this token if its Viterbi path is split
            always_keep[i] = False
        else:
            # Try to find alternative segmentation without single-token path
            alt_path, _ = lattice.viterbi(allow_single_token=False)
            if alt_path:
                # Found alternative segmentation
                always_keep[i] = False
                alternatives[i] = [t.id for t in alt_path]
            else:
                # No alternative segmentation found, must keep
                always_keep[i] = True

    # Second, segments all sentences to compute likelihood
    # with a unigram language model. inverted[i] stores
    # the set of sentence index where the tokens[i] appears.
    freq = [0.0] * len(current_tokens)
    inverted = [[] for _ in current_tokens]
    vsum = 0.0

    for pretoken, count in pretokens.items():
        lattice = model.make_lattice(pretoken)
        _, token_probs = lattice.calc_marginal()
        vsum += count
        
        for token_id, prob in token_probs.items():
            if 0 <= token_id < len(freq):
                freq[token_id] += prob * count
                inverted[token_id].append(pretoken)

    total_freq = sum(freq)
    log_total = math.log(total_freq)
    candidates = []
    new_tokens = []

    # Finally, computes how likely the LM likelihood is reduced if
    # the token[i] is removed from the vocabulary.
    # Since the exact computation of loss is difficult, we compute the
    # loss approximately by assuming that all token[i] in the sentences
    # are replaced with alternatives[i] when token[i] is removed.
    for i, token in enumerate(current_tokens):
        if freq[i] == 0 or not always_keep[i]:
            # not found in Viterbi path. Can remove this entry safely.
            continue
        elif not alternatives[i]:
            # no alternatives. Keeps this entry.
            new_tokens.append(token)
        else:
            # The frequency of token[i]
            F = sum(pretokens.get(pretoken, 0) for pretoken in inverted[i]) / vsum
            
            # The logprob with the token[i]
            logprob_token = math.log(freq[i]) - log_total
            
            # After removing the token[i], its frequency freq[i] is
            # re-assigned to alternatives.
            # new_sum = current_sum - freq[i] + freq[i] * alternatives[i].size()
            #         = current_sum + freq[i] * (alternatives[i] - 1)
            new_total = total_freq + freq[i] * (len(alternatives[i]) - 1)
            logsum_alt = math.log(new_total)
            
            # The frequencies of alternatives are increased by freq[i]
            logprob_alt = 0.0
            for alt_id in alternatives[i]:
                if 0 <= alt_id < len(freq):
                    logprob_alt += math.log(freq[alt_id] + freq[i]) - logsum_alt
            
            # loss: the diff of likelihood after removing the token[i]
            loss = F * (logprob_token - logprob_alt)
            candidates.append((i, loss))

    # Keeps trainer_spec_.shrinking_factor * tokens.size() pieces.
    # shrinking_factor is 0.75 by default.
    candidates.sort(key=lambda x: x[1])
    for i, _ in candidates:
        if len(new_tokens) >= target_size:
            break
        new_tokens.append(current_tokens[i])

    # Ensure we keep locked tokens that might have been pruned
    locked_tokens = [t for t in current_tokens if t.locked and t not in new_tokens]
    new_tokens.extend(locked_tokens)

    if verbose:
        print(f"    Pruned {len(current_tokens) - len(new_tokens)} tokens")
        print(f"    Final vocabulary size: {len(new_tokens)}")

    return new_tokens

def finalize_pieces(
    model: UnigramModel,
    vocab_size: int,
    required_chars: set[str],
    verbose: bool = False,
) -> list[Token]:
    """Finalizes the vocabulary by ensuring required characters are included and keeping top pieces.
    
    This implements the same algorithm as the C++ version, ensuring required characters
    are included and keeping the highest scoring pieces up to the vocabulary size.
    
    Args:
        model: The UnigramModel containing current tokens
        vocab_size: Target vocabulary size
        required_chars: Set of characters that must be included in the final vocabulary
        verbose: Whether to print progress information
        
    Returns:
        List of Token objects representing the final vocabulary
    """
    if verbose:
        print("Finalizing vocabulary...")
        print(f"  Target vocabulary size: {vocab_size}")
        print(f"  Number of required characters: {len(required_chars)}")
    
    # Get all sentence pieces from the model
    sentence_pieces = {token.text: math.exp(token.log_prob) for token in model.tokens}
    final_pieces = {}
    
    # required_chars_ must be included in the final sentencepieces.
    min_score_penalty = 0.0
    MIN_SCORE_PENALTY_DELTA = 0.0001
    
    # Sort required_chars for consistent ordering
    for char in sorted(required_chars):
        s = char  # In Python, we can work with Unicode strings directly
        if s in sentence_pieces:
            final_pieces[s] = sentence_pieces[s]
        else:
            # Add penalty to avoid required pieces from having the same score.
            # Since the required_chars is sorted, frequent pieces have less penalties.
            min_score = min(sentence_pieces.values()) if sentence_pieces else 0.0
            final_pieces[s] = min_score + min_score_penalty
            min_score_penalty += MIN_SCORE_PENALTY_DELTA
    
    # Then keeps sentencepieces with higher scores.
    # Sort sentence pieces by score in descending order
    for piece, score in sorted(sentence_pieces.items(), key=lambda x: -x[1]):
        if piece in final_pieces:
            continue
        if len(final_pieces) >= vocab_size:
            break
        final_pieces[piece] = score
    
    # Convert to Token objects, sorted by score in descending order
    final_tokens = []
    for i, (piece, score) in enumerate(sorted(final_pieces.items(), key=lambda x: -x[1])):
        token = Token(
            text=piece,
            id=i,
            log_prob=math.log(score) if score > 0 else float('-inf'),
            locked=piece in required_chars  # Lock required characters
        )
        final_tokens.append(token)
    
    if verbose:
        print(f"  Final vocabulary size: {len(final_tokens)}")
    
    return final_tokens

# --- Main Training Function ---

def train_unigram(
    pretokens: dict[str, int],
    vocab_size: int = 8000,
    max_token_len: int = 16,
    initial_vocab_factor: int = 4,
    pruning_shrinking_factor: float = 0.75,
    dirichlet_alpha: float = 1.0,
    required_chars: list[str] | None = None,
    max_iterations: int = 100,
    num_sub_iterations: int = 2,
    verbose: bool = False,
) -> list[tuple[str, float]]:
    """Trains a Unigram tokenizer model."""
    if not pretokens:
        raise ValueError("Input 'pretokens' dictionary cannot be empty.")
    
    # initialize vocab and model
    required_chars = set(required_chars or [])
    vocab = make_initial_vocab(pretokens, required_chars, vocab_size * initial_vocab_factor, max_token_len)
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
        pruned_pieces = prune_pieces(model, pretokens, vocab_size, pruning_shrinking_factor, verbose)
        model.set_pieces(pruned_pieces)
        iteration += 1

        # Safety break
        if iteration > 100:
             if verbose: print("    ⛔ Max iterations (100) reached. Stopping.")
             break

    # Phase 4: Finalization
    if verbose: print("\n🏁 Phase 3: Finalizing Vocabulary")
    final_pieces = finalize_pieces(model, vocab_size, required_chars, verbose)
    if verbose:
        print(f"  🏁 Final vocabulary size: {len(final_pieces)}")
        print("\n🎉 === Unigram Model Training Completed ===\n")
    return final_pieces
