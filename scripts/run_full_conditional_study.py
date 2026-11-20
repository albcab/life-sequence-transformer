"""Full-vocabulary, multi-model temporal SMC study with durable job results.

Local proposals plus importance weights and a hard final acceptance potential.
This is not HMM lookahead and does not claim finite-particle exactness.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.constraints.product import EventInYear, ProductConstraint
from src.constraints.life import LifeGrammar
from src.constraints.subword import ConditionalSubwordDFA
from src.llm.dfa_smc import sample


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--vocabulary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--users', type=int, default=0)
    p.add_argument('--years', nargs='+', type=int, default=[1,3,5])
    p.add_argument('--seeds', nargs='+', type=int, default=[2025,2026])
    p.add_argument('--followup-years', type=int, default=2)
    p.add_argument('--particles', type=int, default=64)
    p.add_argument('--max-new-tokens', type=int, default=4096)
    args = p.parse_args()
    if args.followup_years < 0 or args.users < 0 or args.particles < 1 or args.max_new_tokens < 1 or min(args.years) < 1:
        p.error('Invalid user count, particles, years or token budget')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    people = manifest['people'][:args.users or None]
    with args.vocabulary.open(newline='') as f:
        words = {r['TOKEN']: int(r['ID']) for r in csv.DictReader(f, delimiter='\t')}
    inverse = {i:w for w,i in words.items()}
    event_ids = {i for w,i in words.items() if w.startswith('MATINT_') and w != 'MATINT_S0'}
    grammar = LifeGrammar(words.values())
    total = len(manifest['models'])*len(people)*len(args.years)*len(args.seeds)
    provenance = dict(manifest=manifest, args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        target='Rolling-context LM conditioned on corrected life grammar, positive maternity intensity in continuation year N, and a complete N+followup-years EOY window within BPE cap',
        proposal='Local product masks with SMC importance correction and hard terminal acceptance; no vocabulary restriction',
        source_hashes={str(path):sha(path) for path in [Path(__file__), Path('src/models/dfa.py'),
            Path('src/llm/dfa_smc.py'),*sorted(Path('src/constraints').glob('*.py'))]},
        vocabulary_sha256=sha(args.vocabulary),checkpoint_hashes={m['name']:sha(m['checkpoint']) for m in manifest['models']})
    write(args.output/'manifest.json', provenance)
    completed = []
    for spec in manifest['models']:
        print('LOAD', spec['name'], flush=True)
        tok = AutoTokenizer.from_pretrained(spec['model_name'], local_files_only=True)
        tok.add_special_tokens({'additional_special_tokens':['[BOL]','[EOL]','[EOY]','[PLCH0]','[UNK]']})
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(spec['model_name'], local_files_only=True)
        model.resize_token_embeddings(len(tok))
        checkpoint = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict({k.removeprefix('model.'):v for k,v in checkpoint['state_dict'].items()
                               if k.startswith('model.')}, strict=True)
        del checkpoint
        model.eval().to('cuda')
        for year in args.years:
            monitor = EventInYear(frozenset(event_ids),year,horizon_years=year+args.followup_years)
            product = ProductConstraint(grammar,monitor)
            adapter = ConditionalSubwordDFA(tok,words,monitor,feasibility='local')
            for person in people:
                prefix = person['prefix']
                # Calendar boundary selection is guaranteed in the input manifest.
                assert prefix[-1] == words['[EOY]']
                initial = product.initial(prefix)
                encoded = tok.encode(' '.join(inverse[i] for i in prefix))
                for seed in args.seeds:
                    name = f"{spec['name']}-{person['id']}-year{year}-seed{seed}"
                    write(args.output/'status.json',dict(total=total,active=name,completed=completed))
                    print('START',name,flush=True)
                    started=time.monotonic(); torch.manual_seed(seed)
                    torch.cuda.reset_peak_memory_stats()
                    dest=args.output/name;dest.mkdir()
                    try:
                        rows=sample(model,tok,adapter,encoded,prefix,0,nb_particles=args.particles,
                                    max_new_tokens=args.max_new_tokens,verbose=True)
                        for row in rows:
                            text=tok.decode(row['tokens'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                            row['text']=text
                            if row['accepted']:
                                tokens=[words[w] for w in text.split()]
                                state=initial
                                for token in tokens:
                                    if state is None or product.accepting(state):
                                        raise RuntimeError('Independent domain audit: post-terminal or invalid trajectory')
                                    state=product.step(state,token)
                                if state is None or not product.accepting(state):
                                    raise RuntimeError('Independent domain acceptance audit failed')
                                if any(a.startswith('DUR_') and b.startswith('DUR_') for a,b in zip(text.split(),text.split()[1:])):
                                    raise RuntimeError('Repeated duration survived the corrected grammar')
                            elif row['weight'] != 0:
                                raise RuntimeError('Rejected particle has nonzero final weight')
                            lp,lq=row['model_token_log_probs'],row['proposal_token_log_probs']
                            if len(lp)!=len(row['tokens']) or len(lq)!=len(lp):
                                raise RuntimeError('Misaligned probability ledger')
                            if not math.isclose(sum(lp)-sum(lq),row['trajectory_log_weight'],abs_tol=1e-5):
                                raise RuntimeError('Probability ledger mismatch')
                            row.update(model_log_probability=sum(lp),proposal_log_probability=sum(lq),
                                       model_nll_per_token=-sum(lp)/len(lp) if lp else None,
                                       property_satisfied=row['accepted'])
                        with (dest/'samples.jsonl').open('x') as f:
                            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
                        good=[r for r in rows if r['accepted']]
                        weights=[r['weight'] for r in rows]
                        report=dict(name=name,model=spec['name'],person=person['id'],year=year,seed=seed,
                            particles=args.particles,horizon_years=year+args.followup_years,accepted=len(good),extinct=rows[0]['population_extinct'],
                            unique_domain=len({tuple(r['text'].split()) for r in good}),
                            unique_bpe=len({tuple(r['tokens']) for r in good}),
                            weight_ess=1/sum(w*w for w in weights) if sum(weights)>0 else 0.,
                            resamplings=rows[0]['resamplings'],
                            weighted_token_nll=sum(r['weight']*r['model_nll_per_token'] for r in good) if good else None,
                            seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),exit_code=0)
                        write(dest/'summary.json',report)
                    except Exception:
                        (dest/'error.txt').write_text(traceback.format_exc())
                        report=dict(name=name,exit_code=1,seconds=time.monotonic()-started)
                        write(dest/'summary.json',report)
                        completed.append(report)
                        write(args.output/'status.json',dict(total=total,active=None,completed=completed,failed=name))
                        raise
                    completed.append(report)
                    write(args.output/'status.json',dict(total=total,active=None,completed=completed))
                    print('DONE',json.dumps(report),flush=True)
                for fn in [adapter.choices,adapter.words_after,adapter.domain_next,adapter.char_next,
                           adapter.choices_with_budget,adapter.distance]:fn.cache_clear()
            del adapter
        del model
        torch.cuda.empty_cache()
    write(args.output/'complete.json',dict(total=total,completed=completed))


if __name__=='__main__':main()
