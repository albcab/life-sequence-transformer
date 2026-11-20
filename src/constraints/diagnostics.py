"""CPU reference diagnostics. Perplexity measures LM likelihood, not fidelity.

Exact KL/TV require normalized distributions on the same finite support.
Empirical zero counts must not be interpreted as proof of target zero mass.
"""
import math


def trajectory_metrics(product, initial_state, tokens, model_log_probs, proposal_log_probs):
    """Audit a completed trajectory and its per-token probability ledger.

    Log probabilities must come from the actual model/proposal used at each
    step, including EOL when emitted. The path likelihood ratio below is NOT
    a substitute for final SMC weights after resampling.
    """
    if not (len(tokens) == len(model_log_probs) == len(proposal_log_probs)) or not tokens:
        raise ValueError('Require equal nonempty token and log-probability lists')
    for value in [*model_log_probs, *proposal_log_probs]:
        if not math.isfinite(value) or value > 1e-7:
            raise ValueError('Sampled-token log probabilities must be finite and <= 0')
    state = initial_state
    for token in tokens:
        if state is None or product.accepting(state):
            state = None
            break
        state = product.step(state, token)
    log_p, log_q = sum(model_log_probs), sum(proposal_log_probs)
    nll = -log_p / len(tokens)
    return dict(accepted=state is not None and product.accepting(state),
                generated_tokens=len(tokens), model_log_probability=log_p,
                proposal_log_probability=log_q, path_log_likelihood_ratio=log_p-log_q,
                model_nll_per_token=nll,
                model_perplexity=math.exp(nll) if nll < 709 else math.inf)


def distribution_metrics(target, candidate):
    """Exact forward/reverse KL (nats) and TV on an enumerable support."""
    for distribution in (target, candidate):
        if any(not math.isfinite(p) or p < 0 for p in distribution.values()):
            raise ValueError('Probabilities must be finite and nonnegative')
        if not math.isclose(sum(distribution.values()), 1., abs_tol=1e-9):
            raise ValueError('Distributions must be normalized')
    def kl(p, q):
        total = 0.
        for key, mass in p.items():
            if mass:
                other = q.get(key, 0.)
                if not other:
                    return math.inf
                total += mass * math.log(mass / other)
        return total
    return dict(kl_target_candidate=kl(target, candidate),
                kl_candidate_target=kl(candidate, target),
                total_variation=.5 * sum(abs(target.get(k, 0)-candidate.get(k, 0))
                                         for k in target.keys() | candidate.keys()))
