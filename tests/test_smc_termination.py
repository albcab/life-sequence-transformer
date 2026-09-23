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

    def test_person_batches_match_separate_deterministic_populations(self):
        model = TinyModel()
        combined = self.sample(
            model, [[1], [4]], num_years=[10, 10], nb_particles=3,
        )
        self.assertEqual(model.forward_sizes, [6, 3])
        for user, prefix in enumerate([[1], [4]]):
            separate = self.sample(TinyModel(), [prefix] * 3, num_years=10)
            for key in ("input_ids", "padding_mask"):
                torch.testing.assert_close(combined[0][key][user*3:(user+1)*3], separate[0][key])
            for result in (1, 2, 3):
                torch.testing.assert_close(combined[result][user*3:(user+1)*3], separate[result])

    def test_person_specific_year_limits_and_partial_user_batch(self):
        model = TinyModel()
        output, _, weights, _ = self.sample(
            model, [[5], [5]], num_years=[0, 2], nb_particles=2,
        )
        self.assertEqual(model.forward_sizes, [4, 2, 2])
        self.assertEqual(output["padding_mask"].sum(1).tolist(), [2, 2, 4, 4])
        torch.testing.assert_close(weights.reshape(2, 2).sum(1), torch.ones(2, dtype=torch.float64))
        # The final loader batch may contain only one person.
        _, _, weights, _ = self.sample(TinyModel(), [[1]], num_years=[2], nb_particles=3)
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(float(weights.sum()), 1)

    def test_resampling_cannot_cross_people_or_reset_other_weights(self):
        class BranchDFA:
            state_distances = {"branch": 1, "middle": 2, "end": 1, "qfinal": 0}

            def get_initial_conf(self, sequence, verbose=False):
                return {("branch", ())}

        class BranchModel(TinyModel):
            def __call__(self, batch):
                ids = batch["input_ids"]
                self.forward_sizes.append(len(ids))
                logits = torch.empty(len(ids), ids.shape[-1], self.num_outputs)
                for row in range(len(ids)):
                    length = int(batch["padding_mask"][row].sum())
                    assert int(ids[row, 0, length - 1]) != 2
                    person = int(ids[row, 0, 0])
                    probs = torch.zeros(self.num_outputs)
                    if length == 1:
                        mass = .8 if person == 1 else .4
                        probs[2] = probs[3] = mass / 2
                    else:
                        mass = (.1 if person == 1 else .9) if length == 2 else .5
                        probs[3 if length == 2 else 2] = mass
                    probs[0] = 1 - mass
                    logits[row] = probs.log()
                return logits

            def validate_tokens_batched(self, indices, configurations, dfa, remaining_tokens):
                state = next(iter(configurations))[0]
                targets = {"branch": {2: "qfinal", 3: "middle"},
                           "middle": {3: "end"}, "end": {2: "qfinal"}}[state]
                return {token: {(target, (), dfa.state_distances[target])}
                        for token, target in targets.items()}

        model = BranchModel()
        first_choices = iter([0, 1, 0, 1])
        resampled = []

        def draw(probs, num_samples, **kwargs):
            if num_samples == 2:
                # Only person 0 has low ESS; select its finished ancestor twice.
                torch.testing.assert_close(probs, torch.tensor([10/11, 1/11], dtype=torch.float64))
                resampled.append(True)
                return torch.tensor([0, 0])
            return torch.tensor([next(first_choices) if len(probs) == 2 else 0])

        with patch("src.models.decode_only.LIFESEQUENCEDFA", BranchDFA), \
             patch("torch.multinomial", side_effect=draw):
            output, _, weights, trajectory = model.smc_sample(
                batch_for([[1], [4]]), num_years=[10, 10], nb_particles=2, ess_threshold=.9,
            )
        self.assertEqual(len(resampled), 1)
        self.assertEqual(model.forward_sizes, [4, 2, 1])
        self.assertEqual(output["input_ids"][:, 0, 0].tolist(), [1, 1, 4, 4])
        self.assertEqual(output["padding_mask"].sum(1).tolist(), [2, 2, 2, 4])
        torch.testing.assert_close(weights, torch.tensor([.5, .5, 1/1.45, .45/1.45], dtype=torch.float64))
        torch.testing.assert_close(trajectory.exp(), torch.tensor([.8, .8, .4, .18], dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
