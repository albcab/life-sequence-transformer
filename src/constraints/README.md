# Conditional-sampling prototype

`ProductConstraint(LifeGrammar(token_ids), EventInYear(event_ids, target_year))`
combines the existing life grammar with a finite event monitor. The event is
at least one explicitly supplied domain token in continuation year N.
`first_event=True` also excludes earlier events. Year 1 ends at the first
continuation EOY: it may be a partial calendar year. This does not implement
accumulated employment years or infer which tokens denote maternity.

`initial(observed_prefix)` advances the grammar through the observed history;
the monitor starts at the continuation boundary. `choices(state, budget)`
returns only transitions with an accepting suffix within the remaining domain
token budget, INCLUDING the candidate token. Both grammar and monitor must
accept at EOL. This prototype requires EOL; it does not implement the legacy
sampler's separate year-limit stopping rule.

Distance is exact bounded BFS over complete matcher configurations and monitor
states. Infinity means impossible within this budget. SearchLimitExceeded
means unknown and must not be converted into rejection. Full-vocabulary search
can be expensive. Domain-token distances MUST NOT be compared with BPE budgets.

`ConditionalSubwordDFA` connects this monitor to GPT-2 SMC. Its search operates
on complete BPE adapter states (including partial words and matcher states).
Domain-token distances are never used to prune BPE candidates. Conditional mode
requires joint acceptance at EOL and disables the legacy EOY stopping rule.
The target explicitly conditions on EOL within `max_new_tokens`; it does not
claim to target the unlimited-length distribution. Grammar-only runs retain
their existing behavior. Conditional caches are in-memory and property-specific;
the original persistent grammar cache is deliberately disabled for this adapter.

Use `python -m scripts.sample_conditional_lives --help` for the opt-in pilot
runner. It takes a JSON list of observed domain IDs, vocabulary TSV, explicit
event token names, target continuation year, and BPE budget. `--preflight-only`
checks feasibility without loading the LM or running generation. Sampling
requires a matching fine-tuned checkpoint and a fresh output directory.
Outputs include both tokenizations, token probability ledgers, independent
post-generation property audits, final weights, seed, and provenance hashes.
This runner uses its own outputs rather than the legacy economic evaluator.

Exact BFS may hit the state limit on real vocabularies even for short horizons.
A search-limit exception is an unresolved query, never evidence of
impossibility. This is a correctness prototype, not a scalable year-10 engine.
An LTLf compiler, accumulated employment-time semantics, efficient symbolic
reachability, and HMM probability lookahead remain future work.

## Regression and probability measurements

- Exhaustive small-grammar tests compare feasible choices against enumeration.
- Real LIFESEQUENCEDFA tests check partial matchers and joint terminal acceptance.
- trajectory_metrics audits completed sequences and aligned token scores.
- distribution_metrics provides exact KL in both directions and total variation
  for normalized finite distributions; missing candidate support gives infinite
  forward KL. It is not a reliable plug-in estimator for sparse life samples.
- GPT-2 DFA-SMC now records model_token_log_probs and proposal_token_log_probs,
  copied through ancestor resampling and frozen at termination. The experiment
  runner writes these plus sequence scores and per-token model NLL to samples.jsonl.
- The path sum log(p/q) is not the final SMC weight after resampling. Keep both.
- Perplexity is exp(mean token NLL). Compare within the same tokenizer/model,
  context/rolling policy, and termination convention. It is not a measure of
  conditional fidelity; do not compare BPE and life-token perplexity directly.

Run CPU regressions in the existing container environment:

    python -m unittest discover -s tests

## Full-vocabulary SMC baseline

`feasibility="local"` bypasses exhaustive BFS rather than pretending a failed
search proved impossibility. The product monitor still rejects invalid events,
and the final hard acceptance potential assigns zero weight to dead ends or
unfinished trajectories at the BPE cap. A population with no accepted particle
returns all-zero weights and `population_extinct=True`; it must be included in
failure denominators. Corrected local proposals are not finite-particle exact
samples and do not supply HMM lookahead.

`python -m scripts.run_full_conditional_study --help` runs matched model jobs
using a manifest of checkpoints and observed prefixes. Every attempt and final
weight is saved. Successful trajectories receive an independent domain-level
check; malformed scores or grammar violations stop the queue. Full vocabulary
means all life-domain words with the existing structural-special-token BPE
conventions, not every arbitrary GPT-2 token string.

The grammar's one-value-set matchers were corrected to consume exactly one
token (MONTH, TIPO, DUR, EOY, EOL). Earlier repeated-DUR outputs are not valid
under this correction. Both scalar and batch matchers use the corrected rule;
new results must not be treated as an unchanged-grammar rerun of old exports.

### Observation-window campaign

The large campaign sets `EventInYear(..., horizon_years=N+2)` and terminates
at the corresponding EOY, not at EOL. The observed prefix ends at an EOY, so
continuation years are complete calendar-year segments. Premature EOL is
excluded. The event is required in year N, but earlier events are not forbidden.
The target additionally includes completing this window within the BPE cap;
unfinished particles get zero final weight. The target is explicit in each
manifest. This is finite-window conditioning, not a completed lifetime.
