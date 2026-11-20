import itertools
import math
import unittest
from src.constraints.product import ProductConstraint, EventInYear, SearchLimitExceeded
from src.constraints.life import LifeGrammar
from src.constraints.diagnostics import trajectory_metrics, distribution_metrics


class TinyGrammar:
    # 0 = ordinary event, 1 = target event, 2 = EOL, 3 = EOY.
    def initial(self, prefix): return 0
    def step(self, state, token): return None if state == 1 else (1 if token == 2 else 0)
    def accepting(self, state): return state == 1
    def tokens(self, state): return () if state == 1 else (0, 1, 2, 3)


class ProductTests(unittest.TestCase):
    def setUp(self):
        self.product = ProductConstraint(TinyGrammar(), EventInYear({1}, 2))
        self.initial = self.product.initial([])

    def accepts(self, state, suffix):
        for token in suffix:
            if state is None or self.product.accepting(state): return False
            state = self.product.step(state, token)
        return state is not None and self.product.accepting(state)

    def test_choices_match_exhaustive_suffix_oracle(self):
        # Every reachable prefix, including early events and closed years.
        for length in range(3):
            for prefix in itertools.product(range(4), repeat=length):
                state = self.initial
                for token in prefix:
                    state = self.product.step(state, token) if state else None
                if state is None or self.product.accepting(state): continue
                for budget in range(1, 5):
                    expected = set()
                    for size in range(1, budget+1):
                        for suffix in itertools.product(range(4), repeat=size):
                            if self.accepts(state, suffix): expected.add(suffix[0])
                    self.assertEqual(set(self.product.choices(state, budget)), expected)

    def test_joint_distance_and_terminal_acceptance(self):
        self.assertEqual(self.product.distance(self.initial, 3), 3)
        self.assertEqual(self.product.distance(self.initial, 2), math.inf)
        self.assertIsNone(self.product.step(self.initial, 2))
        self.assertTrue(self.accepts(self.initial, [3, 1, 2]))
        self.assertFalse(self.accepts(self.initial, [1, 3, 2]))
        self.assertFalse(self.accepts(self.initial, [3, 3, 1, 2]))

    def test_exhaustion_is_not_reported_as_impossibility(self):
        with self.assertRaises(SearchLimitExceeded):
            self.product.distance(self.initial, 4, max_states=1)

    def test_first_event_and_clock_origin(self):
        monitor = EventInYear({1}, 2, first_event=True)
        self.assertIsNone(monitor.step(monitor.initial(), 1))
        self.assertEqual(self.product.initial([3, 3]).monitor, (1, False))

    def test_real_grammar_partial_matchers_and_terminal(self):
        grammar = LifeGrammar([19, 138, 117, 2, 3])
        product = ProductConstraint(grammar, EventInYear({138}, 1))
        state = product.initial([5, 13, 19, 31, 11, 1])
        self.assertEqual(product.distance(state, 4), 4)
        for token in [19, 138, 117, 2]:
            state = product.step(state, token)
            self.assertIsNotNone(state)
        self.assertTrue(product.accepting(state))
        self.assertEqual(product.choices(state, 10), {})

    def test_probability_ledger_and_rejected_trajectory(self):
        report = trajectory_metrics(self.product, self.initial, [3, 1, 2],
                                    [math.log(.25)]*3, [math.log(.5)]*3)
        self.assertTrue(report['accepted'])
        self.assertAlmostEqual(report['model_perplexity'], 4)
        self.assertAlmostEqual(report['path_log_likelihood_ratio'], math.log(.5)*3)
        self.assertFalse(trajectory_metrics(self.product, self.initial, [3, 1],
                                           [-1]*2, [-1]*2)['accepted'])

    def test_exact_distribution_regression(self):
        target = {'a': .9, 'b': .1}
        self.assertEqual(distribution_metrics(target, target)['total_variation'], 0)
        mismatch = distribution_metrics(target, {'a': .5, 'b': .5})
        self.assertAlmostEqual(mismatch['total_variation'], .4)
        self.assertGreater(mismatch['kl_target_candidate'], 0)
        self.assertEqual(distribution_metrics(target, {'a': 1.})['kl_target_candidate'], math.inf)


if __name__ == '__main__': unittest.main()
