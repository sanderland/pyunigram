import heapq
import logging
import math
from collections import Counter, defaultdict

from scipy.special import digamma

from py_unigram.model import Token, UnigramModel
from py_unigram.utils import create_logger


def log_examples(logger, tokens_with_scores: list[tuple[Token, float]], score_label="score", n=5):
    tokens_with_scores.sort(key=lambda x: x[1])
    for i, (t, score) in enumerate(tokens_with_scores):
        if len(tokens_with_scores) > 2 * n:
            if i == n:
                logger.debug("   │  ├─ ...")
            if n < i < len(tokens_with_scores) - n:
                continue
        list_item = " ├─" if i < len(tokens_with_scores) - 1 else " └─"
        logger.debug(f"   │ {list_item} {repr(t.text):25}  {score_label} = {score:10.3g}")


def make_initial_vocab(
    logger: logging.Logger,
    pretokens: dict[str, int],
    required_tokens: set[str],
    num_tokens: int,
    max_token_length: int = 16,
) -> list[Token]:
    # TODO: does it make sense to have required tokens other than length 1?

    substring_freq = Counter()
    for pretoken, freq in pretokens.items():
        for i in range(len(pretoken)):  # SentencePiece uses suffix array, this is simpler but more mem intensive
            for j in range(i + 1, min(len(pretoken) + 1, i + max_token_length + 1)):
                substring_freq[pretoken[i:j]] += freq
    required_tokens |= {t for t in substring_freq if len(t) == 1}
    for token in required_tokens:
        substring_freq[token] = max(substring_freq.get(token, 0), 1)  # Ensure required tokens are always included
    all_tokens = [(freq * len(token), token) for token, freq in substring_freq.items()]

    selected_tokens = heapq.nlargest(num_tokens, all_tokens, key=lambda item: (item[1] in required_tokens, item[0]))
    log_sum_scores = math.log(sum(score for score, _ in selected_tokens))
    tokens = [
        Token(text, i, math.log(score) - log_sum_scores, required=text in required_tokens)
        for i, (score, text) in enumerate(selected_tokens)
    ]

    logger.info(f"🌱 Selected {num_tokens:,} initial tokens from {len(all_tokens):,} candidates")
    logger.debug(f"   ├─ Source: {len(pretokens):,} unique pretokens")
    logger.debug(f"   └─ Max length: {max_token_length}")

    return tokens


def run_e_step(
    logger: logging.Logger,
    model: UnigramModel,
    pretokens: dict[str, int],
) -> tuple[dict[int, float], float, int]:
    """Performs the Expectation step of the EM algorithm for Unigram."""
    expected_count = defaultdict(float)
    objective = total_tokens = 0
    total_pretoken_freq = sum(freq for _, freq in pretokens.items())

    if total_pretoken_freq == 0:
        return expected_count, objective, total_tokens

    for pretoken, freq in pretokens.items():
        lattice = model.make_lattice(pretoken)
        z, token_prob = lattice.calc_marginal()
        assert not math.isnan(z), f"NaN likelihood for pretoken {pretoken} with freq={freq}."
        for token_id, prob in token_prob.items():
            expected_count[token_id] += prob * freq
        viterbi_path, _ = lattice.viterbi()
        num_tokens_in_sentence = len(viterbi_path)
        total_tokens += num_tokens_in_sentence * freq
        objective -= (z * freq) / total_pretoken_freq

    return expected_count, objective, total_tokens


def run_m_step(
    logger: logging.Logger,
    model: UnigramModel,
    expected_count: dict[int, float],
    dp_smoothing: bool = True,
    k_expected_frequency_threshold=0.5,  # TODO: scale with corpus size?
) -> tuple[UnigramModel, int]:
    """Performs the Maximization step of the EM algorithm for Unigram.

    Args:
        expected_counts: Expected frequency for each token from E-step
        dp_smoothing: If True, use digamma-based sparsity (like SentencePiece).
                      If False, use standard maximum likelihood estimation.
    """
    # Filter infrequent pieces.
    filtered_tokens = [t for t in model.tokens if expected_count[t.id] >= k_expected_frequency_threshold or t.required]
    num_removed = len(model.tokens) - len(filtered_tokens)
    if num_removed > 0:
        filtered_ids = {t.id for t in filtered_tokens}
        removed_tokens = [(t, expected_count[t.id]) for t in model.tokens if t.id not in filtered_ids]
        logger.debug(
            f"   ├─ Removed {num_removed} low-frequency tokens below threshold {k_expected_frequency_threshold} - examples:"
        )
        log_examples(logger, removed_tokens, "expected count")

        model = UnigramModel(filtered_tokens)

    total_freq = sum(expected_count[t.id] for t in model.tokens)
    if dp_smoothing:  # SentencePiece-style: digamma transform with implicit alpha=0 for sparsity bias
        log_total = digamma(total_freq)
        for t in model.tokens:
            t.log_prob = digamma(expected_count[t.id]) - log_total
    else:  # Standard maximum likelihood estimation
        for t in model.tokens:
            t.log_prob = math.log(expected_count[t.id] / total_freq)

    return model, num_removed


def prune_tokens(
    logger: logging.Logger,
    model: UnigramModel,
    pretokens: dict[str, int],
    desired_vocab_size: int,
    shrinking_factor: float = 0.75,
    defensive: bool = False,
) -> tuple[UnigramModel, int]:
    # Calculate target size based on vocab size and shrinking factor
    num_non_required_tokens = sum(not t.required for t in model.tokens)
    shrink_n = int(num_non_required_tokens * (1 - shrinking_factor))
    target_size = max(desired_vocab_size, num_non_required_tokens - shrink_n)

    # 1. Count occurences of all tokens in optimal tokenization of all pretokens
    token_count = {t.id: 0.0 for t in model.tokens}
    for pretoken, count in pretokens.items():
        viterbi_path, _ = model.make_lattice(pretoken).viterbi()
        for token in viterbi_path:
            token_count[token.id] += count

    total_count = sum(token_count.values())
    log_total = math.log(total_count)

    # 2. consider which tokens we can remove and what the cost is
    candidates = []
    new_tokens = []
    unused_token_ids = set()
    for token in model.tokens:
        if token.required:  # never remove required tokens
            new_tokens.append(token)
            continue
        if token_count[token.id] == 0:  # never used in an optimal segmentation. includes split viterbi path.
            unused_token_ids.add(token.id)
            continue
        lattice = model.make_lattice(token.text)
        alt_path, _ = lattice.viterbi(allow_single_token=False)
        assert alt_path, f"Token {token.id} has no alternative segmentation"

        # Computes how the LM likelihood is reduced if the token is removed from the vocabulary.
        # Since the exact computation of loss is difficult, we compute the loss approximately by assuming that all
        # token[id] in the sentences are replaced with their second best segmentation (alt_path) when removed.

        # The logprob with the token[i] = log(count[i] / total_count)
        logprob_token = math.log(token_count[token.id]) - log_total
        # After removing the token[i], its frequency freq[i] is re-assigned to alternatives.
        # new_sum = current_sum - freq[i] + freq[i] * alternatives[i].size()
        #         = current_sum + freq[i] * (alternatives[i] - 1)
        logsum_alt = math.log(total_count + token_count[token.id] * (len(alt_path) - 1))
        # The frequencies of alternatives are increased by freq[i]
        logprob_alt = sum(math.log(token_count[alt.id] + token_count[token.id]) - logsum_alt for alt in alt_path)
        # loss: the diff of likelihood after removing the token[i]
        loss = (token_count[token.id] / total_count) * (logprob_token - logprob_alt)
        # NEW_FEATURE: if alternatives are already gone, optionally prevent removing this token
        defended = any(alt.id in unused_token_ids for alt in alt_path)
        candidates.append((token, loss, defended))

    # 3. Reduce vocabulary to target_size
    candidates.sort(key=lambda x: -x[1])
    defended_tokens = []
    for token, loss, defended in candidates:
        if len(new_tokens) < target_size:
            new_tokens.append(token)
        elif defensive and defended:
            defended_tokens.append((token, loss))
            new_tokens.append(token)

    new_token_ids = {t.id for t in new_tokens}
    pruned_tokens = [
        (token, loss)
        for token, loss, _ in candidates
        if token.id not in new_token_ids and token.id not in unused_token_ids
    ]

    logger.info(
        f"✂️  Pruning vocabulary from {len(model.tokens):,} to target {target_size:,} -> new vocab size {len(new_tokens):,}"
    )
    logger.debug(
        f"   ├─ Target size: {target_size:,} based on shrinking factor {shrinking_factor} * non required tokens {num_non_required_tokens:,} and desired vocab size {desired_vocab_size}"
    )
    if unused_token_ids:
        unused_tokens_info = [(model.tokens_by_id[tid], token_count[tid]) for tid in unused_token_ids]
        logger.debug(f"   ├─ Dropped {len(unused_tokens_info):,} tokens not in any optimal path")
        log_examples(logger, unused_tokens_info, "count")
    logger.debug(f"   ├─ Kept {len(new_tokens)} required tokens")
    if defended_tokens:
        logger.debug(f"   ├─ Defended {len(defended_tokens):,} tokens from being removed along with their alternatives")
        log_examples(logger, defended_tokens, "loss")

    logger.debug(f"   ├─ Pruned {len(pruned_tokens):,} tokens from {len(candidates):,} candidates")
    if pruned_tokens:
        log_examples(logger, pruned_tokens, "loss")
    if candidates:
        logger.debug(f"   └─ Candidates loss range: {candidates[0][1]:.4g} to {candidates[-1][1]:.4g}")
    else:
        logger.info("   └─ No candidates for pruning!")

    return UnigramModel(new_tokens), len(unused_token_ids), len(pruned_tokens), defended_tokens


def finalize_tokens(
    logger: logging.Logger,
    model: UnigramModel,
    vocab_size: int,
) -> tuple[UnigramModel, int]:
    """Finalizes the vocabulary based on scores from the EM iterations.
    This is called after the EM iterations, so we already have good scores."""
    # TODO: add penalty to avoid locked tokens from having the same score?

    final_tokens = {}
    # Add required tokens
    for token in model.tokens:
        if token.required:
            final_tokens[token.id] = token

    # Keep highest scoring tokens
    for token in sorted(model.tokens, key=lambda x: -x.log_prob):
        if token.id in final_tokens:
            continue
        if len(final_tokens) >= vocab_size:
            break
        final_tokens[token.id] = token

    removed_tokens = [(t, t.log_prob) for t in model.tokens if t.id not in final_tokens]
    new_model = UnigramModel(final_tokens.values())
    logger.info(f"✨ Finalizing vocabulary from {len(model.tokens):,} to target {vocab_size:,}")
    logger.info(f"   ├─ Removed {len(removed_tokens):,} tokens")
    log_examples(logger, removed_tokens, "logprob")
    logger.info(
        f"   └─ Kept {len(final_tokens):,} tokens with logprob range {new_model.tokens[0].log_prob:.4g} to {new_model.tokens[-1].log_prob:.4g}"
    )

    return new_model, len(model.tokens) - len(new_model.tokens)


# --- Main Training Function ---


def train_unigram(
    pretokens: dict[str, int],
    vocab_size: int = 8000,
    max_token_len: int = 16,
    initial_vocab_factor: int = 10,
    pre_final_vocab_factor: float = 1.1,
    pruning_shrinking_factor: float = 0.75,
    m_step_dp_smoothing: bool = True,
    m_step_low_count_threshold: float = 0.5,
    defensive_prune: bool = False,
    required_tokens: list[str] | None = None,
    max_iterations: int = 100,
    num_sub_iterations: int = 2,
    verbose: bool = False,
) -> tuple[UnigramModel, dict[str, int]]:
    """Trains a Unigram tokenizer model.
    Args:
        pretokens: Dictionary of pretokens as str -> count
        vocab_size: Target vocabulary size, including required tokens
        max_token_len: Maximum token length
        initial_vocab_factor: Initial vocab size is this x vocab_size
        pre_final_vocab_factor: Desired vocab size before finalization is this x vocab_size
        pruning_shrinking_factor: Shrink non-required part of vocab by this factor each iteration
        m_step_dp_smoothing: If True, use digamma-based sparsity for logprobs (like SentencePiece).
        m_step_low_count_threshold: Threshold for removing low-frequency tokens in M-step
        defensive_prune: If True, defend tokens from being removed if their alternative has been pruned already.
        required_tokens: List of tokens included in vocabulary (in addition to all characters in corpus).
        max_iterations: Maximum number of EM iterations
        num_sub_iterations: Number of sub-iterations per EM iteration
        verbose: Whether to print detailed information about removed tokens

    Returns:
        Tuple containing the trained UnigramModel and statistics
    """
    logger = create_logger("train_unigram", verbose)

    if not pretokens:
        raise ValueError("Input 'pretokens' dictionary cannot be empty.")

    # initialize vocab and model
    required_tokens = set(required_tokens or [])
    vocab = make_initial_vocab(logger, pretokens, required_tokens, vocab_size * initial_vocab_factor, max_token_len)
    total_pretokens = sum(pretokens.values())
    total_bytes = sum(len(pretoken.encode()) * freq for pretoken, freq in pretokens.items())
    desired_vocab_size = int(vocab_size * pre_final_vocab_factor)
    totals_removed = defaultdict(list)
    defended_token_ids = set()

    model = UnigramModel(vocab)

    # EM Training Loop
    for iter in range(max_iterations):
        # Sub-EM Iterations
        for sub_iter in range(num_sub_iterations):
            logger.info(f"🔄 EM Iteration {iter + 1}.{sub_iter + 1}. Model size {len(model.tokens):,}")
            expected_count, objective, total_tokens = run_e_step(logger, model=model, pretokens=pretokens)
            model, m_step_removed = run_m_step(
                logger=logger,
                model=model,
                expected_count=expected_count,
                dp_smoothing=m_step_dp_smoothing,
                k_expected_frequency_threshold=m_step_low_count_threshold,
            )
            totals_removed["M Step Low Count"].append(m_step_removed)
            avg_tokens_per_pretoken = 1.0 * total_tokens / total_pretokens
            logger.debug(f"   ├─ Objective: {objective:.4f}")
            logger.debug(f"   ├─ Total tokens: {total_tokens:,d}")
            logger.debug(f"   └─ Avg tokens/pretoken: {avg_tokens_per_pretoken:.4f}")

        # Check Stopping Condition
        current_size = len(model.tokens)
        if current_size <= desired_vocab_size:
            logger.info("🎯 Target vocabulary size for EM iterations reached")
            logger.debug(f"   ├─ Current: {current_size:,}")
            logger.debug(f"   └─ Target:  {desired_vocab_size:,}")
            break

        # Pruning Step
        model, num_unused, num_pruned, defended_tokens = prune_tokens(
            logger=logger,
            model=model,
            pretokens=pretokens,
            desired_vocab_size=desired_vocab_size,
            shrinking_factor=pruning_shrinking_factor,
            defensive=defensive_prune,
        )
        totals_removed["Prune/Zero Count"].append(num_unused)
        totals_removed["Prune/Loss"].append(num_pruned)
        defended_token_ids.update(t.id for t, _ in defended_tokens)

    # Finalization
    model, finalize_removed = finalize_tokens(logger=logger, model=model, vocab_size=vocab_size)
    totals_removed["Finalize"].append(finalize_removed)

    # Final e-step for stats
    expected_count, objective, total_tokens = run_e_step(logger, model=model, pretokens=pretokens)
    stats = {
        "objective": objective,
        "total_tokens": total_tokens,
        "tokens/pretoken": total_tokens / total_pretokens,
        "bytes/token": total_bytes / total_tokens,
    }
    num_defended = len(defended_token_ids)
    defended_in_final = [(t, t.log_prob) for t in model.tokens if t.id in defended_token_ids]
    logger.info(f"🎉 Training completed successfully! Objective: {stats['objective']:.4f}")
    logger.debug("  📊 Token Removal Statistics:")
    for key, value in totals_removed.items():
        logger.debug(f"   ├─ {key:<20} {sum(value):6,d} tokens" + (f" in steps {value}" if len(value) > 1 else ""))
    if defensive_prune:
        logger.debug(f"   ├─ Defended {num_defended:,} tokens from being removed along with their alternatives.")
        if defended_in_final:
            logger.debug(f"   ├─ {len(defended_in_final):,} defended tokens made it to the final vocabulary.")
            log_examples(logger, defended_in_final, "logprob")
        else:
            logger.debug("   ├─ No defended tokens made it to the final vocabulary.")
    logger.debug("  📊 Compression Statistics:")
    logger.debug(f"   ├─ Total tokens: {stats['total_tokens']:,d}")
    logger.debug(f"   └─ Avg tokens/pretoken: {stats['tokens/pretoken']:.4f}")

    return model, stats
