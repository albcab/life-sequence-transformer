import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.llm.dfa_cache import TransitionCache
from src.llm.dfa_smc import SubwordDFA
from test_hf_dfa_smc import Tokenizer

WORDS = {'[BOL]':1,'[EOL]':2,'[EOY]':3,'MONTH_11':29,'MONTH_1':19,
         'TIPO_10':138,'DUR_12':128,'DUR_2':118,'INCOME_1':140}
PREFIX = [5,13,19,31,12,1]


class DiskCacheTests(unittest.TestCase):
    def test_warm_adapter_reuses_ordered_transitions_without_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            cold = SubwordDFA(Tokenizer(), WORDS, cache_dir=directory)
            state = cold.initial(PREFIX)
            expected = cold.choices(state)
            cold.disk_cache.close()
            warm = SubwordDFA(Tokenizer(), WORDS, cache_dir=directory)
            with patch.object(warm, 'char_next', side_effect=AssertionError('Must load from disk')):
                self.assertEqual(list(warm.choices(state).items()), list(expected.items()))
            self.assertEqual(warm.cache_stats()['disk_hits'], 1)
            self.assertEqual(warm.cache_stats()['computed_states'], 0)
            warm.disk_cache.close()

    def test_key_changes_with_vocabulary_tokenizer_and_grammar(self):
        base = SubwordDFA(Tokenizer(), WORDS, cache_dir=False).cache_fingerprint
        changed = dict(WORDS, INCOME_2=141)
        self.assertNotEqual(base, SubwordDFA(Tokenizer(), changed, cache_dir=False).cache_fingerprint)
        tok = Tokenizer()
        tok.pieces = tok.pieces + [' OTHER']
        self.assertNotEqual(base, SubwordDFA(tok, WORDS, cache_dir=False).cache_fingerprint)
        original = Path.read_bytes
        def changed_source(path):
            return original(path) + (b'\n# changed DFA' if path.name == 'dfa.py' else b'')
        with patch.object(Path, 'read_bytes', changed_source):
            self.assertNotEqual(base, SubwordDFA(Tokenizer(), WORDS, cache_dir=False).cache_fingerprint)

    def test_two_connections_preserve_entries_and_empty_results(self):
        state = (frozenset({('q1',(1,2))}), 'DUR_', False)
        other = (frozenset({('qfinal',())}), '', True)
        with tempfile.TemporaryDirectory() as directory:
            a = TransitionCache(directory, 'test')
            b = TransitionCache(directory, 'test')
            a.put(state, {7:other, 2:state})
            b.put(other, {})
            self.assertEqual(list(b.get(state)), [7,2])
            self.assertEqual(a.get(other), {})
            self.assertEqual(a.stats()['stored_states'], 2)
            a.close(); b.close()

if __name__ == '__main__':
    unittest.main()
