"""Paired maternity experiment; saved SMC prefixes are the source of truth."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil

import hydra
from hydra.utils import instantiate
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from scripts.check_hf_grammar import check_text
from scripts.rolling_window import rolling_sample
from src.models.dfa import LIFESEQUENCEDFA
import counter


class YearLimit(StoppingCriteria):
    def __init__(self, start, eoy, years):
        self.start, self.eoy, self.years = start, eoy, years

    def __call__(self, input_ids, scores, **kwargs):
        return (input_ids[:, self.start:] == self.eoy).sum(dim=1) >= self.years


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--users', type=int, default=8, help='0 selects the full cohort')
    parser.add_argument('--rolling_window_mode', action='store_true')
    args = parser.parse_args()
    if args.users < 0:
        parser.error('--users must be nonnegative')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2025)
    np.random.seed(2025)
    with hydra.initialize_config_dir(config_dir='/usr/src/conf', version_base=None):
        cfg = hydra.compose(config_name='config', overrides=['experiment=hf_llm'])
    tok = instantiate(cfg.tokenizer)
    data = instantiate(cfg.datamodule, tokenizer=tok, _convert_='all')
    vocab = data.vocabulary
    dfa = LIFESEQUENCEDFA()
    cohort = pd.read_csv('/usr/src/mothers_test_ids_clean.csv')
    if args.users:
        cohort = cohort.head(args.users)
    if cohort.USER_ID.duplicated().any():
        raise ValueError('Duplicate cohort IDs')
    baseline = Path(args.baseline)
    cohort.to_csv(out/'mothers_test_ids_clean.csv', index=False)
    shutil.copy('/usr/src/income_100_cache.json', out/'income_100_cache.json')
    selected = []
    for person in cohort.itertuples(index=False):
        uid = int(person.USER_ID)
        path = baseline/f'{uid}_index.csv'
        arr = np.loadtxt(path, delimiter=',', dtype=int)
        starts = [int(np.flatnonzero(row)[0]) for row in arr[1:]]
        assert len(arr) == 33 and len(set(starts)) == 1
        start = starts[0]
        prefix = arr[0, :start].tolist()
        assert all(prefix) and prefix[-1] == 3
        prompt = ' '.join(vocab.index2token[x] for x in prefix)
        selected.append(dict(id=uid, years=int(person.year_to_remove), start=start,
                             prompt=prompt, prefix=prefix, original=arr[0],
                             source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        # Re-evaluate exactly the same SMC subset using unchanged counter.py.
        dest = out/'generated'/'smc_100'/'decoder_only'/'mothers_test_ids_clean'/'0'
        dest.mkdir(parents=True, exist_ok=True)
        for kind in ('index', 'weights'):
            shutil.copy(baseline/f'{uid}_{kind}.csv', dest/f'{uid}_{kind}.csv')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, local_files_only=True)
    model.resize_token_embeddings(len(tok))
    model.load_state_dict({k.removeprefix('model.'): v for k, v in checkpoint['state_dict'].items()
                           if k.startswith('model.')}, strict=True)
    del checkpoint
    model.eval().to('cuda')
    eoy, eol = (tok.convert_tokens_to_ids(x) for x in ('[EOY]', '[EOL]'))
    records, people = [], []
    def save_progress():
        temp = out/'progress.tmp'
        temp.write_text(json.dumps(dict(total_users=len(selected), completed_users=len(people),
            attempted=sum(p['attempted'] for p in people), evaluable=sum(p['evaluable'] for p in people),
            users=people), indent=2)+'\n')
        temp.replace(out/'progress.json')
        print(f'Completed users: {len(people)}/{len(selected)}', flush=True)
    save_progress()
    dest = out/'generated'/'hf_100'/'decoder_only'/'mothers_test_ids_clean'/'0'
    dest.mkdir(parents=True, exist_ok=True)
    with (out/'samples.jsonl').open('w') as stream, torch.inference_mode():
        for person in selected:
            encoded = tok.encode(person['prompt'])
            remaining = model.config.n_positions-len(encoded)
            retained, retained_indices = [], []
            print(f"id={person['id']} prefix_bpe={len(encoded)} remaining={remaining} years={person['years']}", flush=True)
            if remaining <= 0 and not args.rolling_window_mode:
                people.append(dict(id=person['id'], status='prefix_exceeds_context', prefix_bpe=len(encoded), requested_years=person['years'], attempted=0, evaluable=0))
                save_progress()
                continue
            for first in range(0, 32, 8):
                if args.rolling_window_mode:
                    bol = tok.convert_tokens_to_ids('[BOL]')
                    background = encoded[:encoded.index(bol)+1]
                    sampled = rolling_sample(model, encoded, background, eoy, eol,
                        tok.eos_token_id, tok.pad_token_id, person['years'])
                else:
                    inputs = torch.tensor([encoded]*8, device='cuda')
                    generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                        do_sample=True, temperature=1., top_k=0, top_p=1., num_beams=1,
                        max_new_tokens=remaining, eos_token_id=[eol, tok.eos_token_id],
                        pad_token_id=tok.pad_token_id, use_cache=True,
                        stopping_criteria=StoppingCriteriaList([YearLimit(len(encoded), eoy, person['years'])]))
                    sampled = [dict(tokens=row[len(encoded):].tolist(), end='context_limit', shifts=0,
                                    boundary_fallbacks=0) for row in generated]
                for sample, result in enumerate(sampled):
                    new = result['tokens']
                    years, end, cut = 0, result['end'], len(new)
                    for i, token in enumerate(new):
                        years += token == eoy
                        if token in (eol, tok.eos_token_id) or years >= person['years']:
                            cut = i+1
                            end = 'EOL' if token == eol else ('native_EOS' if token == tok.eos_token_id else 'year_limit')
                            break
                    new = new[:cut]
                    text = person['prompt'] + tok.decode(new, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    assessment = check_text(text, vocab.token2index, dfa, capped=end in ('context_limit', 'token_budget'))
                    # The legacy evaluator right-aligns years, so only full-horizon valid rows enter it.
                    eligible = end == 'year_limit' and assessment['status'] == 'accepted'
                    record = dict(id=person['id'], sample=first+sample, end=end, generated_years=years,
                                  requested_years=person['years'], window_shifts=result['shifts'],
                                  boundary_fallbacks=result['boundary_fallbacks'], assessment=assessment, evaluator_eligible=eligible, text=text)
                    records.append(record)
                    stream.write(json.dumps(record)+'\n'); stream.flush()
                    if eligible:
                        ids = [vocab.token2index[x] for x in text.split()]
                        assert ids[:person['start']] == person['prefix']
                        saved = np.zeros(max(len(ids), len(person['original'])), dtype=int)
                        saved[person['start']:len(ids)] = ids[person['start']:]
                        retained.append(saved); retained_indices.append(first+sample)
            if retained:
                all_rows = [person['original']]+retained
                width = max(map(len, all_rows))
                padded = [np.pad(row, (0, width-len(row))) for row in all_rows]
                np.savetxt(dest/f"{person['id']}_index.csv", np.vstack(padded), fmt='%d', delimiter=',')
                np.savetxt(dest/f"{person['id']}_weights.csv", np.full(len(retained), 1/len(retained)))
            people.append(dict(id=person['id'], prefix_bpe=len(encoded), requested_years=person['years'],
                               attempted=32, evaluable=len(retained), retained_sample_indices=retained_indices))
            save_progress()
    os.chdir(out)
    for implementation in ('smc_100', 'hf_100'):
        if list((out/'generated'/implementation/'decoder_only'/'mothers_test_ids_clean'/'0').glob('*_index.csv')):
            counter.maternity(implementation, 'decoder_only', 'test_ids_clean', 0, logs=False)
    summary = dict(checkpoint=args.checkpoint, checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        seed=2025, rolling_window_mode=args.rolling_window_mode,
        context_policy='preserve background through BOL; evict whole oldest years to 768 BPE; reset positions and KV cache; token-tail fallback for oversized years; 8192 new-token safety cap',
        window_shifts=sum(r['window_shifts'] for r in records), boundary_fallbacks=sum(r['boundary_fallbacks'] for r in records),
        sampling=dict(temperature=1, top_k=0, top_p=1, dfa_mask=False),
        users=people, attempted=len(records), status=dict(Counter(r['assessment']['status'] for r in records)),
        termination=dict(Counter(r['end'] for r in records)), evaluable=sum(r['evaluator_eligible'] for r in records),
        evaluator_note='HF results conditional on grammar-valid full-horizon samples; uniform weights renormalized within retained samples. All attempts retained in samples.jsonl. SMC uses original weights.',
        source_hashes={p['id']:p['source_sha256'] for p in selected})
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
