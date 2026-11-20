import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.models.dfa import TOKEN_INDICES_DICT, evaluate_matcher, evaluate_matcher_batch
from src.constraints.subword import ConditionalSubwordDFA
from src.constraints.product import EventInYear
from src.llm.dfa_smc import sample
from test_conditional_smc import Tokens, Model
import test_conditional_smc as fixtures


class LocalTests(unittest.TestCase):
    def test_singletons_reject_duplicates_and_batch_matches(self):
        for name,token in [('duronetwelve',128),('monthone',19),('tipooneeleven',138),('eoy',3),('eol',2)]:
            matcher=TOKEN_INDICES_DICT[name]
            self.assertEqual(evaluate_matcher(matcher,[token]),(True,False,False))
            self.assertEqual(evaluate_matcher(matcher,[token,token]),(False,False,False))
            for prefix in [(),(token,)]:
                got=evaluate_matcher_batch(matcher,prefix,np.array([token,999]))
                for i,t in enumerate([token,999]):
                    self.assertEqual(tuple(bool(v[i]) for v in got),evaluate_matcher(matcher,list(prefix)+[t]))

    def test_exhausted_population_has_zero_weights_not_fake_success(self):
        adapter=ConditionalSubwordDFA(Tokens(),fixtures.ConditionalSMCTests.words,EventInYear({138},1),feasibility='local')
        # Every one-token prefix is incomplete, even though locally valid.
        rows=sample(Model(),Tokens(),adapter,[1],fixtures.ConditionalSMCTests.prefix,0,nb_particles=2,max_new_tokens=1)
        self.assertTrue(all(r['population_extinct'] and not r['accepted'] and r['weight']==0 for r in rows))

    def test_partial_population_final_weights_exclude_nonaccepted(self):
        class Adapter:
            conditional=True;allow_particle_failure=True;eoy=3;eol=2;words={'[EOL]':2}
            def initial(self,prefix):return 0
            def choices_with_budget(self,state,budget):return {2:1,3:0}
            def accepting(self,state):return state==1
        draws=iter([torch.tensor([0]),torch.tensor([1])])
        with patch('torch.multinomial',side_effect=lambda *a,**k:next(draws)):
            rows=sample(Model(),Tokens(),Adapter(),[1],[1],0,nb_particles=2,max_new_tokens=1)
        self.assertEqual([r['weight'] for r in rows],[1.,0.])
        self.assertEqual([r['accepted'] for r in rows],[True,False])


if __name__=='__main__':unittest.main()

class HorizonTests(unittest.TestCase):
    def test_fixed_window_stops_at_eoy_and_domain_audit_agrees(self):
        from src.constraints.product import ProductConstraint
        from src.constraints.life import LifeGrammar
        monitor=EventInYear({138},1,horizon_years=2)
        adapter=ConditionalSubwordDFA(Tokens(),fixtures.ConditionalSMCTests.words,monitor,feasibility='local')
        # Force two complete years; EOL must not be a proposal in this mode.
        expected=[4,5,6,3,4,5,6,3]
        state=adapter.initial(fixtures.ConditionalSMCTests.prefix)
        draws=[]
        for token in expected:
            choices=adapter.choices(state)
            self.assertNotIn(2,choices)
            draws.append(list(choices).index(token));state=choices[token]
        self.assertTrue(adapter.accepting(state))
        it=iter(draws)
        with patch('torch.multinomial',side_effect=lambda *a,**k:torch.tensor([next(it)])):
            row=sample(Model(),Tokens(),adapter,[1],fixtures.ConditionalSMCTests.prefix,0,
                       nb_particles=1,max_new_tokens=8)[0]
        self.assertEqual(row['end'],'property_horizon')
        self.assertTrue(row['accepted'])
        product=ProductConstraint(LifeGrammar(fixtures.ConditionalSMCTests.words.values()),monitor)
        state=product.initial(fixtures.ConditionalSMCTests.prefix)
        for token in [19,138,117,3,19,138,117,3]:state=product.step(state,token)
        self.assertTrue(product.accepting(state))
