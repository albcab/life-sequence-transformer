import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch

from src.constraints.product import EventInYear, SearchLimitExceeded
from src.constraints.subword import ConditionalSubwordDFA
from src.llm.dfa_smc import sample


class Tokens:
    pieces = [' ', '[BOL]', ' [EOL]', ' [EOY]', ' MONTH_1', ' TIPO_10',
              ' DUR_1', ' TIPO_', '10']
    pad_token_id = 0
    def __len__(self): return len(self.pieces)
    def decode(self, ids, **kwargs): return ''.join(self.pieces[i] for i in ids)
    def convert_tokens_to_ids(self, token): return {'[BOL]':1, '[EOL]':2, '[EOY]':3}[token]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(n_positions=32)
    def forward(self, input_ids, **kwargs):
        logits = torch.zeros((*input_ids.shape, len(Tokens())))
        cache = ((torch.zeros(input_ids.shape[0], 1, 1, 1),) * 2,)
        return SimpleNamespace(logits=logits, past_key_values=cache)


class ConditionalSMCTests(unittest.TestCase):
    prefix = [5, 13, 19, 31, 11, 1]
    words = {'[BOL]':1, '[EOL]':2, '[EOY]':3, 'MONTH_1':19, 'TIPO_10':138, 'DUR_1':117}

    def adapter(self, year=1, **kwargs):
        return ConditionalSubwordDFA(Tokens(), self.words, EventInYear({138}, year), **kwargs)

    def test_bpe_budget_counts_split_token_and_terminal(self):
        adapter = self.adapter()
        initial = adapter.initial(self.prefix)
        self.assertEqual(adapter.distance(initial, 4), 4)
        self.assertEqual(adapter.distance(initial, 3), math.inf)
        month = adapter.choices_with_budget(initial, 4)[4]
        self.assertIn(5, adapter.choices_with_budget(month, 3))
        self.assertNotIn(7, adapter.choices_with_budget(month, 3))
        self.assertIn(7, adapter.choices_with_budget(month, 4))

    def test_property_blocks_early_eol(self):
        adapter = self.adapter(year=2)
        state = adapter.initial(self.prefix)
        for token in [4, 5, 6]: state = adapter.choices(state)[token]
        self.assertNotIn(2, adapter.choices(state))
        state = adapter.choices(state)[3]
        for token in [4, 7, 8, 6, 2]: state = adapter.choices(state)[token]
        self.assertTrue(adapter.accepting(state))

    def test_completed_samples_probability_ledger_and_legacy_year_stop_bypass(self):
        torch.manual_seed(19)
        adapter = self.adapter(year=2)
        # Legacy num_years=0 would stop at the first EOY before the event.
        rows = sample(Model(), Tokens(), adapter, [1], self.prefix, 0,
                      nb_particles=4, ess_threshold=1., max_new_tokens=8)
        for row in rows:
            self.assertEqual(row['end'], 'EOL')
            self.assertEqual(row['tokens'], [4, 5, 6, 3, 4, 5, 6, 2])
            self.assertEqual(len(row['model_token_log_probs']), 8)
            self.assertAlmostEqual(sum(row['model_token_log_probs']), -8*math.log(9), places=5)
            self.assertAlmostEqual(sum(row['proposal_token_log_probs']), 0., places=5)
            self.assertAlmostEqual(row['trajectory_log_weight'], -8*math.log(9), places=5)
        self.assertAlmostEqual(sum(row['weight'] for row in rows), 1.)

    def test_impossible_budget_and_search_limit_fail_before_forward(self):
        model = Model()
        with patch.object(model, 'forward', side_effect=AssertionError('No model call expected')):
            with self.assertRaises(ValueError):
                sample(model, Tokens(), self.adapter(), [1], self.prefix, 0, max_new_tokens=3)
            with self.assertRaises(SearchLimitExceeded):
                sample(model, Tokens(), self.adapter(max_states=1), [1], self.prefix, 0, max_new_tokens=4)

    def test_enumerated_proposal_correction_recovers_conditional_distribution(self):
        from src.constraints.diagnostics import distribution_metrics
        adapter = self.adapter()
        leaves = []
        def visit(state, tokens, draws, proposal_mass):
            if adapter.accepting(state):
                leaves.append((tuple(tokens), draws, proposal_mass))
                return
            choices = adapter.choices_with_budget(state, 5-len(tokens))
            for index, (token, nxt) in enumerate(choices.items()):
                visit(nxt, tokens+[token], draws+[index], proposal_mass/len(choices))
        visit(adapter.initial(self.prefix), [], [], 1.)
        self.assertGreater(len(leaves), 1)
        prior = {tokens: 9.**(-len(tokens)) for tokens, _, _ in leaves}
        normalizer = sum(prior.values())
        target = {tokens: p/normalizer for tokens, p in prior.items()}
        corrected, proposal = {}, {}
        for tokens, draws, mass in leaves:
            forced = iter(draws)
            with patch('torch.multinomial', side_effect=lambda *a, **k: torch.tensor([next(forced)])):
                row = sample(Model(), Tokens(), adapter, [1], self.prefix, 0,
                             nb_particles=1, max_new_tokens=5)[0]
            self.assertEqual(tuple(row['tokens']), tokens)
            self.assertAlmostEqual(math.exp(sum(row['proposal_token_log_probs'])), mass, places=6)
            corrected[tokens] = mass * math.exp(row['trajectory_log_weight'])
            proposal[tokens] = mass
        z = sum(corrected.values())
        corrected = {tokens: p/z for tokens, p in corrected.items()}
        self.assertLess(distribution_metrics(target, corrected)['total_variation'], 1e-6)
        self.assertGreater(distribution_metrics(target, proposal)['kl_target_candidate'], 1e-3)

    def test_property_caches_are_isolated(self):
        first, second = self.adapter(1), self.adapter(2)
        a, b = first.initial(self.prefix), second.initial(self.prefix)
        for token in [4,5,6]:
            a = first.choices(a)[token]
            b = second.choices(b)[token]
        self.assertIn(2, first.choices(a))
        self.assertNotIn(2, second.choices(b))
        self.assertIsNone(first.disk_cache)
        self.assertIsNone(second.disk_cache)


if __name__ == '__main__': unittest.main()
