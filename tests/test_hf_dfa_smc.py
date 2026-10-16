import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from src.llm.dfa_smc import SubwordDFA, sample


class Tokenizer:
    pieces = [' ', '[BOL]', ' [EOL]', ' [EOY]', ' MONTH_11', ' TIPO_10', ' DUR_12',
              ' DUR_2', ' MONTH_1', ' INCOME_1', 'distilled', ' DUR_', '12']
    pad_token_id = 0
    def __len__(self): return len(self.pieces)
    def decode(self, ids, **kw): return ''.join(self.pieces[i] for i in ids)
    def convert_tokens_to_ids(self, text): return {'[BOL]':1,'[EOL]':2,'[EOY]':3}[text]


class GrammarTests(unittest.TestCase):
    def test_duration_and_prefix_constraints(self):
        adapter=SubwordDFA(Tokenizer(), {'[BOL]':1,'[EOL]':2,'[EOY]':3,'MONTH_11':29,
            'MONTH_1':19,'TIPO_10':138,'DUR_12':128,'DUR_2':118,'INCOME_1':140}, cache_dir=False)
        state=adapter.initial([5,13,19,31,12,1])
        state=adapter.choices(state)[4]
        state=adapter.choices(state)[5]
        choices=adapter.choices(state)
        self.assertNotIn(6,choices)  # MONTH_11 cannot have DUR_12.
        self.assertNotIn(10,choices)
        self.assertNotIn(12,adapter.choices(choices[11]))
        state=choices[7]
        self.assertIn(3,adapter.choices(state))
        terminal=adapter.choices(state)[2]
        self.assertFalse(adapter.choices(terminal).get(4))


class FakeAdapter:
    eoy=3; eol=2; words={'[EOL]':2}
    def initial(self,prefix): return 0
    def choices(self,state):
        return {2:2,3:1} if state==0 else {2:2}


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.p=torch.nn.Parameter(torch.zeros(1));self.config=SimpleNamespace(n_positions=32)
    def forward(self,input_ids,**kw):
        probs=torch.tensor([.1,.1,.6,.2])
        logits=probs.log().expand(input_ids.shape[0],input_ids.shape[1],4)
        cache=((torch.zeros(input_ids.shape[0],1,1,1),torch.zeros(input_ids.shape[0],1,1,1)),)
        return SimpleNamespace(logits=logits,past_key_values=cache)


class WeightTests(unittest.TestCase):
    def test_terminal_weights_are_frozen(self):
        # First particle finishes immediately (.8); second takes a year then EOL (.8*.6).
        draws=iter([torch.tensor([0]),torch.tensor([1]),torch.tensor([0])])
        with patch('torch.multinomial',side_effect=lambda *a,**k:next(draws)):
            result=sample(FakeModel(),Tokenizer(),FakeAdapter(),[1],[1],10,nb_particles=2,ess_threshold=.01)
        self.assertEqual([r['tokens'] for r in result],[[2],[3,2]])
        self.assertAlmostEqual(result[0]['weight'],.625,places=6)
        self.assertAlmostEqual(result[1]['weight'],.375,places=6)

    def test_legacy_year_stop(self):
        draws=iter([torch.tensor([0]),torch.tensor([1])])
        with patch('torch.multinomial',side_effect=lambda *a,**k:next(draws)):
            result=sample(FakeModel(),Tokenizer(),FakeAdapter(),[1],[1],0,nb_particles=2)
        self.assertEqual([r['end'] for r in result],['EOL','year_limit'])

    def test_resampling_preserves_states_and_resets_only_incremental_weight(self):
        class Longer(FakeAdapter):
            def choices(self,state):
                return {2:3,3:1} if state==0 else ({3:2} if state==1 else {2:3})
        draws=iter([torch.tensor([0]),torch.tensor([1]),torch.tensor([0]),
                    torch.tensor([0,1]),torch.tensor([0])])
        with patch('torch.multinomial',side_effect=lambda *a,**k:next(draws)):
            result=sample(FakeModel(),Tokenizer(),Longer(),[1],[1],10,nb_particles=2,ess_threshold=1)
        self.assertEqual([r['tokens'] for r in result],[[2],[3,3,2]])
        self.assertEqual(result[0]['resamplings'],1)
        self.assertAlmostEqual(result[0]['weight'],.625,places=6)
        self.assertAlmostEqual(result[1]['weight'],.375,places=6)
        self.assertAlmostEqual(float(torch.tensor(result[1]['trajectory_log_weight']).exp()),.8*.2*.6,places=6)


class PublicAPITests(unittest.TestCase):
    def test_wrapper_masks_terminal_padding_and_exports_weights(self):
        from src.models.hf_llm import HFCausalLM
        from src.multiple_idx_llm import hf_to_l2v
        model = SimpleNamespace(model=FakeModel(), _subword_dfa=FakeAdapter())
        draws = iter([torch.tensor([0]), torch.tensor([1]), torch.tensor([0])])
        with patch('torch.multinomial', side_effect=lambda *a, **k: next(draws)):
            ids, masks, weights, _ = HFCausalLM.smc_sample(model,
                torch.tensor([[1]]), torch.tensor([[1]]), Tokenizer(), 10,
                {'[BOL]':1, '[EOL]':2, '[EOY]':3}, nb_particles=2, ess_threshold=.01)
        self.assertEqual(masks.sum(1).tolist(), [2,3])
        rows = hf_to_l2v(ids, Tokenizer(), {'[BOL]':1,'[EOL]':2,'[EOY]':3}, 4, masks)
        self.assertEqual(rows.tolist(), [[1,2,0,0], [1,3,2,0]])
        self.assertAlmostEqual(float(weights.sum()), 1)

if __name__ == '__main__':
    unittest.main()
