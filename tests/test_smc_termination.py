"""Small CPU regressions for SMC termination; no checkpoint or data required."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.models.decode_only import GeneratorDecoderOnly
from src.models.dfa import LIFESEQUENCEDFA


class TinyDFA:
    state_distances = {"end": 1, "year": 2, "loop": 1, "qfinal": 0}

    def get_initial_conf(self, sequence, verbose=False):
        state = "qfinal" if sequence[-1] == 2 else {
            1: "end", 4: "year", 5: "loop",
        }[sequence[0]]
        return {(state, ())}


class TinyModel:
    smc_sample = GeneratorDecoderOnly.smc_sample
    num_outputs = 6

    def __init__(self):
        self.forward_sizes = []

    def __call__(self, batch):
        ids = batch["input_ids"]
        self.forward_sizes.append(len(ids))
        logits = torch.empty(len(ids), ids.shape[-1], self.num_outputs)
        for row in range(len(ids)):
            length = int(batch["padding_mask"][row].sum())
            first, last = int(ids[row, 0, 0]), int(ids[row, 0, length - 1])
            assert last != 2, "Finished particles must not enter the model"
            if first == 5:
                token, mass = 3, 0.6
            elif last == 4:
                token, mass = 3, 0.4
            else:
                token, mass = 2, (0.8 if first == 1 else 0.5)
            probs = torch.full((self.num_outputs,), (1 - mass) / 5)
            probs[token] = mass
            logits[row] = probs.log()
        return logits

    def validate_tokens_batched(self, indices, configurations, dfa, remaining_tokens):
        state = next(iter(configurations))[0]
        assert state != "qfinal", "Terminal states must not be advanced"
        token, target = {
            "end": (2, "qfinal"), "year": (3, "end"), "loop": (3, "loop"),
        }[state]
        return {token: {(target, (), dfa.state_distances[target])}}


def batch_for(prefixes):
    ids = torch.zeros(len(prefixes), 3, 8, dtype=torch.long)
    mask = torch.zeros(len(prefixes), 8, dtype=torch.long)
    for row, prefix in enumerate(prefixes):
        ids[row, 0, :len(prefix)] = torch.tensor(prefix)
        mask[row, :len(prefix)] = 1
    return {"input_ids": ids, "padding_mask": mask}


class SMCTerminationTests(unittest.TestCase):
    def sample(self, model, prefixes, **kwargs):
        with patch("src.models.decode_only.LIFESEQUENCEDFA", TinyDFA):
            return model.smc_sample(batch_for(prefixes), **kwargs)

    def test_eol_freezes_sequence_and_importance_weight(self):
        model = TinyModel()
        output, generated, weights, trajectory = self.sample(
            model, [[1], [4]], num_years=10, ess_threshold=0.1,
        )
        self.assertEqual(model.forward_sizes, [2, 1])
        self.assertEqual(output["padding_mask"].sum(1).tolist(), [2, 3])
        self.assertEqual(generated.sum(1).tolist(), [1, 2])
        # Early EOL mass .8 versus delayed EOL mass .4 * .5.
        torch.testing.assert_close(weights, torch.tensor([.8, .2], dtype=torch.float64))
        torch.testing.assert_close(trajectory.exp(), torch.tensor([.8, .2], dtype=torch.float64))

    def test_resampling_retains_finished_particles_and_their_done_flags(self):
        model = TinyModel()
        multinomial = torch.multinomial

        def draw(probs, num_samples, **kwargs):
            if num_samples == 3:
                torch.testing.assert_close(
                    probs, torch.tensor([.5, .25, .25], dtype=torch.float64),
                )
                return torch.tensor([0, 0, 1])
            return multinomial(probs, num_samples, **kwargs)

        with patch("torch.multinomial", side_effect=draw):
            output, _, weights, trajectory = self.sample(
                model, [[1], [4], [4]], num_years=10, ess_threshold=1,
            )
        self.assertEqual(model.forward_sizes, [3, 1])
        self.assertEqual(output["padding_mask"].sum(1).tolist(), [2, 2, 3])
        torch.testing.assert_close(weights, torch.tensor([.4, .4, .2], dtype=torch.float64))
        torch.testing.assert_close(trajectory.exp(), torch.tensor([.8, .8, .2], dtype=torch.float64))

    def test_already_terminated_prefix_is_not_advanced(self):
        model = TinyModel()
        output, generated, weights, trajectory = self.sample(
            model, [[1, 2]], num_years=10,
        )
        self.assertEqual(model.forward_sizes, [])
        self.assertEqual(int(generated.sum()), 0)
        self.assertEqual(int(output["padding_mask"].sum()), 2)
        self.assertEqual(float(weights[0]), 1)
        self.assertEqual(float(trajectory[0]), 0)

    def test_year_limit_still_applies_with_eol_configured(self):
        model = TinyModel()
        output, _, _, _ = self.sample(model, [[5]], num_years=0, ending_idx=2)
        self.assertEqual(model.forward_sizes, [1])
        self.assertEqual(int(output["padding_mask"].sum()), 2)

    def test_terminal_state_needs_no_outgoing_transition(self):
        dfa = LIFESEQUENCEDFA()
        dfa.transitions_by_state.pop("qfinal", None)
        result = GeneratorDecoderOnly.validate_tokens_batched(
            None, np.arange(6), {("qfinal", (), 0)}, dfa, 4,
        )
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
