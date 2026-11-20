"""Exact bounded reachability on a grammar/monitor product.

All transitions consume one *domain token*. This is not a BPE distance or a
probabilistic lookahead. Search exhaustion must never be treated as rejection.
"""
from collections import deque
from dataclasses import dataclass
from math import inf


class SearchLimitExceeded(RuntimeError):
    """Reachability is unknown because the configured search limit was reached."""


@dataclass(frozen=True)
class ProductState:
    grammar: object
    monitor: object


class ProductConstraint:
    def __init__(self, grammar, monitor):
        self.grammar = grammar
        self.monitor = monitor

    def initial(self, observed_prefix):
        # The event clock starts at the continuation boundary, not at birth/BOL.
        return ProductState(self.grammar.initial(observed_prefix), self.monitor.initial())

    def step(self, state, token):
        grammar = self.grammar.step(state.grammar, token)
        if grammar is None:
            return None
        monitor = self.monitor.step(state.monitor, token)
        if monitor is None:
            return None
        following = ProductState(grammar, monitor)
        if self.grammar.accepting(grammar) and not self.monitor.accepting(monitor):
            return None
        return following

    def accepting(self, state):
        terminal = (self.grammar.at_year_boundary(state.grammar)
                    if getattr(self.monitor, 'horizon_years', None) is not None
                    else self.grammar.accepting(state.grammar))
        return terminal and self.monitor.accepting(state.monitor)

    def distance(self, state, budget, *, max_states=100000):
        """Minimum accepting suffix length <= budget, else infinity.

        Infinity is relative to this budget, not proof of global impossibility.
        BFS includes full partial-matcher configurations. No static grammar
        distance is assumed. An exception signals resource exhaustion.
        """
        if budget < 0 or max_states < 1:
            raise ValueError('budget must be nonnegative and max_states positive')
        queue = deque([(state, 0)])
        seen = {state}
        while queue:
            current, depth = queue.popleft()
            if self.accepting(current):
                return depth
            if depth == budget:
                continue
            for token in self.grammar.tokens(current.grammar):
                nxt = self.step(current, token)
                if nxt is None or nxt in seen:
                    continue
                if len(seen) >= max_states:
                    raise SearchLimitExceeded('Product reachability exceeded max_states')
                seen.add(nxt)
                queue.append((nxt, depth + 1))
        return inf

    def choices(self, state, remaining_tokens, *, max_states=100000):
        """Feasible next tokens; remaining_tokens INCLUDES the next token."""
        if remaining_tokens < 0:
            raise ValueError('remaining_tokens must be nonnegative')
        if remaining_tokens == 0 or self.accepting(state):
            return {}
        result = {}
        for token in self.grammar.tokens(state.grammar):
            nxt = self.step(state, token)
            if nxt is not None and self.distance(nxt, remaining_tokens - 1,
                                                 max_states=max_states) < inf:
                result[token] = nxt
        return result


@dataclass(frozen=True)
class EventInYear:
    """At least one event in continuation year N; optional no earlier event.

    Year 1 is the current continuation segment up to its first EOY. If the
    prompt ends mid-year, this first segment is a partial year. Event identity
    is supplied explicitly by domain token IDs, not inferred from spelling.
    """
    event_tokens: frozenset
    target_year: int
    eoy: int = 3
    eol: int = 2
    first_event: bool = False
    horizon_years: int | None = None

    def __post_init__(self):
        object.__setattr__(self, 'event_tokens', frozenset(self.event_tokens))
        if self.horizon_years is not None and self.horizon_years < self.target_year:
            raise ValueError('horizon_years must cover target_year')
        if self.target_year < 1 or not self.event_tokens:
            raise ValueError('Specify a positive target_year and event tokens')
        if self.event_tokens & {self.eoy, self.eol}:
            raise ValueError('Event tokens must exclude EOY/EOL')

    def initial(self):
        return (1, False)

    def accepting(self, state):
        return state[1] and (self.horizon_years is None or state[0] > self.horizon_years)

    def step(self, state, token):
        year, seen = state
        if self.horizon_years is not None and (year > self.horizon_years or token == self.eol):
            return None
        if token in self.event_tokens:
            if self.first_event and year < self.target_year:
                return None
            seen = seen or year == self.target_year
        if token == self.eol and not seen:
            return None
        if token == self.eoy:
            if year == self.target_year and not seen:
                return None
            year = min(year + 1, (self.horizon_years or self.target_year) + 1)
        return (year, seen)
