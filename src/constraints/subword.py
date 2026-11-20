"""Temporal monitoring on completed domain tokens, feasibility in BPE units."""
from collections import deque
from functools import lru_cache
from math import inf

from src.llm.dfa_smc import SubwordDFA
from src.models.dfa import evaluate_matcher, TOKEN_INDICES_DICT
from src.constraints.product import SearchLimitExceeded


class ConditionalSubwordDFA(SubwordDFA):
    conditional = True

    def __init__(self, tokenizer, token2index, monitor, *, max_states=10000, feasibility="exact"):
        if max_states < 1:
            raise ValueError('max_states must be positive')
        # Base persistent cache does not fingerprint the property. Never reuse it.
        super().__init__(tokenizer, token2index, cache_dir=False)
        self.monitor = monitor
        self.max_states = max_states
        self.year_boundary_states = {target for (_, matcher), target in self.dfa.transitions.items()
                                     if matcher is TOKEN_INDICES_DICT['eoy']}
        if feasibility not in ('exact', 'local'):
            raise ValueError('feasibility must be exact or local')
        self.feasibility = feasibility
        self.allow_particle_failure = feasibility == 'local'

    def initial(self, domain_prefix):
        base, word, needs_space = super().initial(domain_prefix)
        if any(node == 'qfinal' for node, _ in base):
            raise ValueError('Cannot condition a continuation after an observed EOL')
        return (frozenset(((node, self.monitor.initial()), prefix)
                         for node, prefix in base), word, needs_space)

    @lru_cache(maxsize=8192)
    def domain_next(self, configs, token):
        following = set()
        for (node, temporal), prefix in configs:
            updated = self.monitor.step(temporal, token)
            if updated is None:
                continue
            for matcher, target in self.dfa.transitions_by_state.get(node, ()):
                satisfied, extendable, fixable = evaluate_matcher(matcher, list(prefix) + [token])
                if satisfied:
                    if target != 'qfinal' or self.monitor.accepting(updated):
                        following.add(((target, updated), ()))
                    if extendable:
                        following.add(((node, updated), prefix + (token,)))
                elif fixable:
                    following.add(((node, updated), prefix + (token,)))
        return frozenset(following)

    def accepting(self, state):
        configs, word, _ = state
        return not word and any((node in self.year_boundary_states if self.monitor.horizon_years is not None
                                else node == 'qfinal') and self.monitor.accepting(temporal)
                                for (node, temporal), prefix in configs if not prefix)

    @lru_cache(maxsize=32768)
    def distance(self, state, budget):
        """Exact shortest accepting BPE suffix within budget; raises on exhaustion."""
        if budget < 0:
            raise ValueError('budget must be nonnegative')
        queue = deque([(state, 0)])
        seen = {state}
        while queue:
            current, depth = queue.popleft()
            if self.accepting(current):
                return depth
            if depth == budget:
                continue
            for nxt in self.choices(current).values():
                if nxt in seen:
                    continue
                if len(seen) >= self.max_states:
                    raise SearchLimitExceeded('BPE product reachability exceeded max_states; no sampling performed with an approximate mask')
                seen.add(nxt)
                queue.append((nxt, depth + 1))
        return inf

    @lru_cache(maxsize=32768)
    def choices_with_budget(self, state, remaining_tokens):
        if remaining_tokens < 0:
            raise ValueError('remaining_tokens must be nonnegative')
        if remaining_tokens == 0 or self.accepting(state):
            return {}
        if self.feasibility == 'local':
            # No approximate distances: retain all locally permitted proposals.
            # Final acceptance is applied as a hard SMC potential at the cap.
            return self.choices(state)
        return {token: nxt for token, nxt in self.choices(state).items()
                if self.distance(nxt, remaining_tokens - 1) < inf}
