"""Explicit single-person conditional pilot; never runs the legacy evaluator.

Prompt JSON is a list of domain token IDs. Year 1 starts at its boundary.
Example (CPU preflight only):
 python -m scripts.sample_conditional_lives --prompt prompt.json \
   --vocabulary data/processed/vocab/labor_100/result.tsv \
   --event-token MATINT_S1 --target-year 2 --max-new-tokens 256 --preflight-only
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.constraints.product import EventInYear, ProductConstraint
from src.constraints.life import LifeGrammar
from src.constraints.diagnostics import trajectory_metrics
from src.constraints.subword import ConditionalSubwordDFA
from src.llm.dfa_smc import sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--vocabulary', type=Path, required=True)
    parser.add_argument('--event-token', nargs='+', required=True)
    parser.add_argument('--generation-token', nargs='+', help='Explicit extra support restriction for bounded diagnostics')
    parser.add_argument('--target-year', type=int, required=True)
    parser.add_argument('--first-event', action='store_true')
    parser.add_argument('--model-name', default='openai-community/gpt2')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-new-tokens', type=int, required=True)
    parser.add_argument('--max-search-states', type=int, default=10000)
    parser.add_argument('--particles', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.target_year < 1 or args.particles < 1:
        parser.error('Token budget, target year and particles must be positive')
    if not args.preflight_only and (args.checkpoint is None or args.output is None):
        parser.error('Sampling requires --checkpoint and --output')
    if args.output is not None and args.output.exists():
        parser.error('Use a fresh output directory')
    with args.vocabulary.open(newline='') as f:
        words = {r['TOKEN']: int(r['ID']) for r in csv.DictReader(f, delimiter='\t')}
    unknown = set(args.event_token) - words.keys()
    if unknown:
        parser.error(f'Unknown event tokens: {sorted(unknown)}')
    prefix = json.loads(args.prompt.read_text())
    inverse = {v: k for k, v in words.items()}
    if not isinstance(prefix, list) or not prefix or any(type(i) is not int or i not in inverse for i in prefix):
        parser.error('Prompt must be a nonempty JSON list of known integer domain IDs')
    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.add_special_tokens({'additional_special_tokens': ['[BOL]', '[EOL]', '[EOY]', '[PLCH0]', '[UNK]']})
    tok.pad_token = tok.eos_token
    monitor = EventInYear(frozenset(words[w] for w in args.event_token), args.target_year,
                          eoy=words['[EOY]'], eol=words['[EOL]'], first_event=args.first_event)
    generation_words = words
    if args.generation_token:
        unknown = set(args.generation_token) - words.keys()
        if unknown:
            parser.error(f'Unknown generation tokens: {sorted(unknown)}')
        required = set(args.event_token) | {'[BOL]', '[EOY]', '[EOL]'}
        generation_words = {w: words[w] for w in set(args.generation_token) | required}
    adapter = ConditionalSubwordDFA(tok, generation_words, monitor, max_states=args.max_search_states)
    initial = adapter.initial(prefix)
    distance = adapter.distance(initial, args.max_new_tokens)
    if not math.isfinite(distance):
        raise ValueError('Property cannot be completed within this BPE budget')
    print(json.dumps({'minimum_bpe_completion': distance, 'target_year': args.target_year}), flush=True)
    if args.preflight_only:
        return
    # Preserve the same lexical prompt convention as the existing GPT-2 runner.
    encoded = tok.encode(' '.join(inverse[i] for i in prefix))
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tok))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict({k.removeprefix('model.'): v for k, v in checkpoint['state_dict'].items()
                           if k.startswith('model.')}, strict=True)
    del checkpoint
    model.eval().to(args.device)
    rows = sample(model, tok, adapter, encoded, prefix, 0, nb_particles=args.particles,
                  max_new_tokens=args.max_new_tokens, verbose=True)
    # Independent domain-level audit, not merely the sampler's acceptance flag.
    product = ProductConstraint(LifeGrammar(words.values()), monitor)
    domain_initial = product.initial(prefix)
    for row in rows:
        text = tok.decode(row['tokens'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        domain_tokens = [words[w] for w in text.split()]
        state = domain_initial
        for token in domain_tokens:
            if state is None or product.accepting(state):
                raise RuntimeError('Invalid or post-terminal domain trajectory')
            state = product.step(state, token)
        if state is None or not product.accepting(state):
            raise RuntimeError('Independent domain property audit failed')
        # Adapter provides the same product interface needed by this score audit.
        audit = trajectory_metrics(_BPEAudit(adapter), initial, row['tokens'],
                                   row['model_token_log_probs'], row['proposal_token_log_probs'])
        if not audit['accepted']:
            raise RuntimeError('BPE trajectory audit failed')
        row.update(audit, text=text, domain_tokens=domain_tokens, property_satisfied=True)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'samples.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata.update(checkpoint_sha256=_sha256(args.checkpoint), prompt_sha256=_sha256(args.prompt),
                    vocabulary_sha256=_sha256(args.vocabulary),
                    target='LM with rolling context conditioned on grammar, event, EOL within BPE budget, and any explicit generation-token restriction',
                    year_clock='continuation-relative EOY segments; first segment may be partial',
                    accepted=len(rows), source_sha256={str(p): _sha256(p) for p in
                        [Path(__file__), Path('src/llm/dfa_smc.py'), Path('src/models/dfa.py'),
                         *sorted(Path('src/constraints').glob('*.py'))]})
    (args.output/'summary.json').write_text(json.dumps(metadata, indent=2)+'\n')


class _BPEAudit:
    def __init__(self, adapter): self.adapter = adapter
    def step(self, state, token): return self.adapter.choices(state).get(token)
    def accepting(self, state): return self.adapter.accepting(state)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == '__main__': main()
