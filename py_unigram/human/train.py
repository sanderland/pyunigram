import heapq  # For BoundedPriorityQueue logic
import math
from collections import Counter
from typing import Dict, List, Tuple

from scipy.special import digamma

from .model import Lattice, Token

# --- Core Training Functions ---

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



def _to_log_prob(items: list[tuple[str, float]]) -> None:
    """Converts frequencies/scores to log probabilities in-place. Mirrors C++ ToLogProb."""
    if not items:
        return
    total_count = sum(freq for _, freq in items)
    if total_count <= 0:
        log_prob = math.log(1.0 / len(items)) if items else 0.0
        for i in range(len(items)):
            items[i] = (items[i][0], log_prob)
        return
    log_total = math.log(total_count)
    for i in range(len(items)):
        piece, freq = items[i]
        if freq > 0:
            items[i] = (piece, math.log(freq) - log_total)
        else:
            items[i] = (piece, -1e38)

def run_e_step(
    model: TrainerModel,
    pretokens: dict[str, int],
    verbose: bool = False,
) -> tuple[list[float], float, int]:
    """
    Performs the Expectation step of the EM algorithm for Unigram.
    """
    piece_size = len(model)
    expected = [0.0] * piece_size
    total_obj = 0.0
    total_tokens = 0
    total_pretoken_freq = sum(freq for _, freq in pretokens.items())

    if total_pretoken_freq == 0:
        return expected, total_obj, total_tokens

    for pretoken_str, freq in pretokens.items():
        lattice = Lattice(pretoken_str)
        model.populate(lattice)
        Z = lattice.populate_marginal(freq, expected)
        if math.isnan(Z):
            if verbose: print(f"      ⚠️ Warning: NaN likelihood for pretoken freq={freq}. Skipping.")
            continue
        viterbi_path, _ = lattice.viterbi()
        num_tokens_in_sentence = len(viterbi_path)
        total_tokens += num_tokens_in_sentence * freq
        total_obj -= (Z * freq) / total_pretoken_freq

    return expected, total_obj, total_tokens

def run_m_step(
    model: TrainerModel,
    expected_counts: list[float],
    dirichlet_alpha: float = 1.0,
    verbose: bool = False,
) -> list[tuple[str, float]]:
    """
    Performs the Maximization step of the EM algorithm for Unigram.
    """
    current_pieces = model.pieces
    if len(current_pieces) != len(expected_counts):
        raise ValueError(f"Model pieces ({len(current_pieces)}) and expected counts ({len(expected_counts)}) size mismatch.")

    # Filter infrequent pieces (Bayesian/DP modification)
    k_expected_frequency_threshold = 0.5
    filtered_pieces_and_expected = [
        (current_pieces[i], expected_counts[i])
        for i in range(len(current_pieces))
        if expected_counts[i] >= k_expected_frequency_threshold
    ]
    if len(filtered_pieces_and_expected) < len(current_pieces) and verbose:
        removed_pieces = [(p, ec) for p, ec in zip(current_pieces, expected_counts, strict=False) if ec < k_expected_frequency_threshold]
        print(f"    🔍 Filtering: {len(current_pieces) - len(filtered_pieces_and_expected)} pieces filtered out due to low expected frequency.")
        for piece, ec in sorted(removed_pieces, key=lambda x: x[1]):
            print(f"      - Removed piece: {piece[0]!r} with expected count {ec:.4f} and score {piece[1]:.2f}")
    if not filtered_pieces_and_expected:
        if verbose: print("    ⚠️ Warning: No pieces survived filtering in M-step. Returning empty list.")
        return []

    # Prepare data for VB update (Digamma)
    pieces_for_update = [p for p, _ in filtered_pieces_and_expected]
    freqs_for_update = [f for _, f in filtered_pieces_and_expected]

    # Variational Bayes Update using Digamma
    sum_freq_plus_alpha = sum(freqs_for_update) + len(freqs_for_update) * dirichlet_alpha
    log_total = digamma(sum_freq_plus_alpha) if sum_freq_plus_alpha > 0 else float('-inf')
    new_pieces = []
    for i in range(len(pieces_for_update)):
        piece_str, _ = pieces_for_update[i]
        freq = freqs_for_update[i]
        adjusted_freq = freq + dirichlet_alpha
        if adjusted_freq > 0 and log_total != float('-inf'):
            new_score = digamma(adjusted_freq) - log_total
        else:
            new_score = -1e38
        new_pieces.append((piece_str, new_score))

    return new_pieces

def prune_pieces(
    model: TrainerModel,
    pretokens: dict[str, int],
    vocab_size: int,
    shrinking_factor: float = 0.75,
    verbose: bool = False,
) -> list[tuple[str, float]]:
    """
    Prunes the current set of pieces based on Viterbi analysis and alternative paths.
    """
    if verbose: print("    🔪 Pruning sentence pieces...")
    current_pieces = model.pieces
    piece_size = len(current_pieces)

    # Phase 1: Find alternative segmentations for each piece
    always_keep = [False] * piece_size
    alternatives: List[List[int]] = [[] for _ in range(piece_size)]
    for i in range(piece_size):
        piece_str, _ = current_pieces[i]
        lattice = Lattice(piece_str)
        model.populate(lattice)
        nbest_paths_and_scores = lattice.nbest(2)
        if len(nbest_paths_and_scores) == 1:
            always_keep[i] = True
        elif len(nbest_paths_and_scores) >= 2:
            path1_nodes, _ = nbest_paths_and_scores[0]
            if len(path1_nodes) >= 2:
                always_keep[i] = False
            elif len(path1_nodes) == 1:
                always_keep[i] = True
                path2_nodes_list, _ = nbest_paths_and_scores[1] if len(nbest_paths_and_scores) > 1 else ([], 0.0)
                alternatives[i] = [node.piece_id for node in path2_nodes_list]

    # Phase 2: Calculate Viterbi frequencies for all pieces across pretokens
    freq = [0.0] * piece_size
    vsum = 0.0
    for pretoken_str, pretoken_freq in pretokens.items():
        vsum += pretoken_freq
        lattice = Lattice(pretoken_str)
        model.populate(lattice)
        viterbi_path, _ = lattice.viterbi()
        for node in viterbi_path:
            if 0 <= node.piece_id < piece_size:
                freq[node.piece_id] += pretoken_freq

    # Phase 3 & 4: Calculate pruning loss and select pieces to keep
    candidates: List[Tuple[int, float]] = []
    new_sentencepieces: List[Tuple[str, float]] = []
    sum_freq = sum(freq)
    log_sum_freq = math.log(sum_freq) if sum_freq > 0 else float('-inf')
    for i in range(piece_size):
        if freq[i] == 0 or not always_keep[i]:
            continue
        elif not alternatives[i]:
            new_sentencepieces.append(current_pieces[i])
        else:
            F = (freq[i] / vsum) if vsum > 0 else 0.0
            logprob_sp = math.log(freq[i]) - log_sum_freq if freq[i] > 0 and log_sum_freq != float('-inf') else -1e38
            freq_i = freq[i]
            num_alternatives = len(alternatives[i])
            sum_freq_alt = sum_freq + freq_i * (num_alternatives - 1)
            log_sum_freq_alt = math.log(sum_freq_alt) if sum_freq_alt > 0 else float('-inf')
            logprob_alt = 0.0
            for alt_piece_id in alternatives[i]:
                new_freq_alt_piece = freq[alt_piece_id] + freq_i
                if new_freq_alt_piece > 0 and log_sum_freq_alt != float('-inf'):
                    logprob_alt += (math.log(new_freq_alt_piece) - log_sum_freq_alt)
                else:
                    logprob_alt += -1e38
            loss = F * (logprob_sp - logprob_alt) if F > 0 else 0.0
            candidates.append((i, loss))

    pruned_size = max(vocab_size, int(shrinking_factor * piece_size))
    candidates.sort(key=lambda x: x[1])
    for piece_index, loss in candidates:
        if len(new_sentencepieces) >= pruned_size:
            break
        new_sentencepieces.append(current_pieces[piece_index])

    if verbose: print(f"    ✅ Pruning done. Pieces after pruning: {len(new_sentencepieces)}")
    return new_sentencepieces

def finalize_pieces(
    model: TrainerModel,
    required_chars: set[str],
    vocab_size: int,
    meta_pieces: list[str] = None,
    verbose: bool = False,
) -> list[tuple[str, float]]:
    """
    Finalizes the vocabulary to the exact target size, ensuring required characters are present.
    """
    if meta_pieces is None:
        meta_pieces = ["<unk>"]
    if verbose: print("    🎯 Finalizing sentence pieces...")
    current_pieces = model.pieces
    final_pieces_dict: Dict[str, float] = {}

    # Ensure required characters are present
    min_score_penalty = 0.0
    k_min_score_penalty_delta = 0.0001
    sorted_required_chars = sorted(list(required_chars))
    model_min_score = getattr(model, 'min_score', -1e38)
    for char_str in sorted_required_chars:
        found = False
        for piece, score in current_pieces:
            if piece == char_str:
                final_pieces_dict[piece] = score
                found = True
                break
        if not found:
            final_pieces_dict[char_str] = model_min_score + min_score_penalty
            min_score_penalty += k_min_score_penalty_delta

    # Add other high-score pieces up to vocab_size
    vocab_size - len(meta_pieces)
    sorted_current = sorted(current_pieces, key=lambda x: x[1], reverse=True)
    for piece, score in sorted_current:
        if len(final_pieces_dict) >= vocab_size:
            break
        if piece not in final_pieces_dict:
            final_pieces_dict[piece] = score

    # Sort final pieces by score descending for output and adjust size
    final_pieces_sorted_list = sorted(final_pieces_dict.items(), key=lambda x: x[1], reverse=True)
    if len(final_pieces_sorted_list) > vocab_size:
        final_pieces_sorted_list = final_pieces_sorted_list[:vocab_size]

    if verbose: print(f"    ✅ Finalization done. Final pieces: {len(final_pieces_sorted_list)}")
    return final_pieces_sorted_list

# --- Main Training Function ---

def train_unigram(
    pretokens: dict[str, int],
    vocab_size: int = 8000,
    num_sub_iterations: int = 2,
    dirichlet_alpha: float = 1.0,
    pruning_shrinking_factor: float = 0.75,
    initial_vocab_factor: int = 4,
    max_piece_len: int = 16,
    required_chars: list[str] | None = None,
    meta_pieces: list[str] = None,
    verbose: bool = False,
) -> list[tuple[str, float]]:
    """
    Trains a Unigram tokenizer model.
    """
    if meta_pieces is None:
        meta_pieces = ["<unk>"]
    if verbose: print("🚀 === Unigram Model Training Started ===")
    if not pretokens:

        raise ValueError("Input 'pretokens' dictionary cannot be empty.")
    required_chars = set(required_chars or [])
    vocab = make_initial_vocab(pretokens, required_chars, vocab_size * initial_vocab_factor, max_piece_len)
    if verbose: print(f"🌱 Generated {vocab_size} * {initial_vocab_factor} = {len(vocab)} initial tokens with max length {max_piece_len} from {len(pretokens)} pretokens")

    # Phase 2: Initialize Model
    seed_piece_strings_for_init = {piece: 1 for piece, _ in seed_pieces}
    model = TrainerModel(seed_piece_strings_for_init, max_piece_length=max_piece_len)

    # Phase 3: EM Training Loop
    if verbose: print("\n🔁 Phase 2: EM Training Loop")
    desired_vocab_size = int(vocab_size * 1.1)
    if verbose: print(f"  🎯 Target intermediate vocab size: {desired_vocab_size}")
    iteration = 0
    while True:
        if verbose: print(f"\n  🔁 EM Iteration {iteration + 1}")
        # Sub-EM Iterations
        for sub_iter in range(num_sub_iterations):
            if verbose: print(f"    🔄 Sub-iteration {sub_iter + 1}/{num_sub_iterations}")
            expected_counts, obj, num_tokens = run_e_step(model, pretokens, verbose)
            new_pieces = run_m_step(model, expected_counts, dirichlet_alpha, verbose)
            model.set_pieces(new_pieces)
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
    final_pieces = finalize_pieces(model, required_chars_set, vocab_size, meta_pieces, verbose)
    if verbose:
        print(f"  🏁 Final vocabulary size: {len(final_pieces)}")
        print("\n🎉 === Unigram Model Training Completed ===\n")
    return final_pieces
