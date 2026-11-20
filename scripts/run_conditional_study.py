"""Sequential bounded conditional study with durable per-job status and audits."""
import argparse
import collections
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--vocabulary', required=True)
    parser.add_argument('--duration-token', nargs='+', default=[f'DUR_{i}' for i in range(1,13)])
    parser.add_argument('--max-new-tokens', type=int, default=24)
    parser.add_argument('--timeout', type=int, default=240)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    plan = [(p, seed, n) for p, seed, n in itertools.product(
        sorted(args.prompts.glob('*.json')), [2025, 2026], [32, 128])]
    if not plan:
        raise ValueError('No prompt JSON files')
    def save(name, value):
        tmp = args.output/(name+'.tmp')
        tmp.write_text(json.dumps(value, indent=2)+'\n')
        tmp.replace(args.output/name)
    save('plan.json', [dict(prompt=str(p),seed=s,particles=n) for p,s,n in plan])
    for prompt, seed, particles in plan:
        name = f'{prompt.stem}-seed{seed}-p{particles}'
        command = [sys.executable, '-m', 'scripts.sample_conditional_lives',
                   '--prompt', str(prompt), '--vocabulary', args.vocabulary,
                   '--checkpoint', args.checkpoint, '--output', str(args.output/name),
                   '--event-token', 'MATINT_S1', 'MATINT_S2', 'MATINT_S3', 'MATINT_S4', 'MATINT_S4+',
                   '--generation-token', *args.duration_token,
                   '--target-year', '1', '--max-new-tokens', str(args.max_new_tokens), '--max-search-states', '16384',
                   '--particles', str(particles), '--seed', str(seed), '--device', 'cuda']
        save('status.json', dict(active=name,finished=jobs,total=len(plan)))
        print('START', name, flush=True)
        started = time.monotonic()
        with (args.output/(name+'.log')).open('x') as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code = 124
        result = dict(name=name, seed=seed, particles=particles, exit_code=code,
                      elapsed_seconds=time.monotonic()-started, command=command)
        if code == 0:
            rows = [json.loads(line) for line in (args.output/name/'samples.jsonl').read_text().splitlines()]
            weights = [r['weight'] for r in rows]
            result.update(accepted=sum(r['property_satisfied'] for r in rows),
                unique_texts=len({tuple(r['domain_tokens']) for r in rows}),
                unique_bpe=len({tuple(r['tokens']) for r in rows}),
                final_weight_ess=1/sum(w*w for w in weights),
                resamplings=rows[0]['resamplings'],
                mean_model_token_nll=sum(w*r['model_nll_per_token'] for w,r in zip(weights,rows)),
                repeated_duration_samples=sum(any(a.startswith('DUR_') and b.startswith('DUR_')
                    for a,b in zip(r['text'].split(),r['text'].split()[1:])) for r in rows),
                domain_counts=dict(collections.Counter(r['text'].strip() for r in rows)))
        jobs.append(result)
        save('status.json', dict(active=None,finished=jobs,total=len(plan)))
        print('DONE', name, 'exit', code, flush=True)
        if code != 0:
            save('failed.json', result)
            raise SystemExit(code)
    save('complete.json', dict(jobs=jobs,total=len(plan),
        scope='Restricted pre-maternity continuation diagnostic, not full-vocabulary or distant-event conditioning'))


if __name__ == '__main__': main()
