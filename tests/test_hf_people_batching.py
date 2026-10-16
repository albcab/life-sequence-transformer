import unittest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from src.llm.batched_dfa_smc import sample_people
from src.llm.dfa_smc import sample


class Tokens:
    pad_token_id=0
    def convert_tokens_to_ids(self, text): return 1

class Grammar:
    words={'[EOL]':2}; eoy=3; eol=2
    def initial(self, prefix):return 0
    def choices(self,state):
        if state >= 7:return {2:state+1,3:state+1}
        return {3:state+1,4:state+1} if state%2==0 else {3:state+1}

class BatchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=GPT2LMHeadModel(GPT2Config(vocab_size=8,n_positions=16,n_embd=16,n_layer=1,n_head=2,
            attn_pdrop=0,resid_pdrop=0,embd_pdrop=0)).eval()
        self.people=[dict(id=10,encoded=[1,4,3],prefix=[1],years=3),
                     dict(id=20,encoded=[1]+[4,3]*8,prefix=[1],years=8)]

    def test_mixed_length_cache_and_rolling_match_separate_runs(self):
        settings=dict(nb_particles=4,ess_threshold=1,max_new_tokens=40,seed=2025)
        separate=[sample_people(self.model,Tokens(),Grammar(),[p],**settings)[0] for p in self.people]
        together=sample_people(self.model,Tokens(),Grammar(),self.people,**settings)
        for expected,actual in zip(separate,together):
            for a,b in zip(expected,actual):
                for key in ['tokens','end','shifts','resamplings','boundary_fallbacks']:
                    self.assertEqual(a[key],b[key],key)
                self.assertAlmostEqual(a['weight'],b['weight'],places=5)
            self.assertAlmostEqual(sum(x['weight'] for x in actual),1)
        self.assertGreater(sum(p['resamplings'] for pop in together for p in pop),0)
        self.assertGreater(sum(p['shifts'] for pop in together for p in pop),0)

    def test_single_population_matches_original_sampler(self):
        person=self.people[0]
        torch.manual_seed(2025+person['id'])
        original=sample(self.model,Tokens(),Grammar(),person['encoded'],person['prefix'],person['years'],
                        nb_particles=4,ess_threshold=1,max_new_tokens=40)
        batched=sample_people(self.model,Tokens(),Grammar(),[person],nb_particles=4,ess_threshold=1,max_new_tokens=40)[0]
        for a,b in zip(original,batched):
            self.assertEqual(a['tokens'],b['tokens'])
            self.assertEqual(a['resamplings'],b['resamplings'])
            self.assertAlmostEqual(a['weight'],b['weight'],places=6)

if __name__=='__main__':unittest.main()
