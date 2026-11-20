"""Domain-token adapter retaining the existing life grammar's matcher state."""
from functools import lru_cache
from src.models.dfa import LIFESEQUENCEDFA, evaluate_matcher, TOKEN_INDICES_DICT


class LifeGrammar:
    def __init__(self, token_ids, dfa=None):
        self.dfa = dfa if dfa is not None else LIFESEQUENCEDFA()
        self.vocabulary = tuple(sorted(set(map(int, token_ids))))

    def initial(self, observed_prefix):
        state = frozenset({(self.dfa.initial_state, ())})
        for token in observed_prefix:
            state = self.step(state, int(token))
            if state is None:
                raise ValueError('Observed prefix is invalid under the life grammar')
        return state

    def at_year_boundary(self, state):
        targets = {target for (_, matcher), target in self.dfa.transitions.items()
                   if matcher is TOKEN_INDICES_DICT['eoy']}
        return any(node in targets and not prefix for node, prefix in state)

    def accepting(self, state):
        return ('qfinal', ()) in state

    @lru_cache(maxsize=8192)
    def step(self, state, token):
        following = set()
        for node, prefix in state:
            for matcher, target in self.dfa.transitions_by_state.get(node, ()):
                satisfied, extendable, fixable = evaluate_matcher(matcher, list(prefix) + [token])
                if satisfied:
                    following.add((target, ()))
                    if extendable:
                        following.add((node, prefix + (token,)))
                elif fixable:
                    following.add((node, prefix + (token,)))
        return frozenset(following) if following else None

    def tokens(self, state):
        if self.accepting(state):
            return ()
        return (token for token in self.vocabulary if self.step(state, token) is not None)
