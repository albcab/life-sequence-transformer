"""Run the six saved LST experiment configurations sequentially."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

EXPERIMENTS = [('mothers', 0), ('mothers', 1), ('non_mothers', 1),
               ('unemployed', 0), ('pensioner', 1), ('pensioner', 3)]

def plan(repo, baseline):
    jobs = []
    for cohort, offset in EXPERIMENTS:
        cohort_path = repo/f'{cohort}_test_ids_clean.csv'
        with cohort_path.open() as stream:
            people = list(csv.DictReader(stream))
        ids = [row['USER_ID'] for row in people]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate IDs: {cohort_path}')
        folder = baseline/f'{cohort}_test_ids_clean'/str(offset)
        for row in people:
            if int(row['year_to_remove']) + offset <= 0:
                raise ValueError(f'Nonpositive year limit: {row}')
            for kind in ('index', 'weights'):
                path = folder/f"{row['USER_ID']}_{kind}.csv"
                if not path.is_file():
                    raise FileNotFoundError(path)
        jobs.append(dict(cohort=cohort, offset=offset, users=len(people), baseline=str(folder)))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path('/usr/src'))
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--dfa-smc', action='store_true')
    parser.add_argument('--experiments', nargs='+', help='Subset in execution order, e.g. mothers:0 pensioner:3')
    args = parser.parse_args()
    jobs = plan(args.repo, args.baseline)
    if args.experiments:
        available = {f"{job['cohort']}:{job['offset']}": job for job in jobs}
        if len(set(args.experiments)) != len(args.experiments):
            parser.error('Duplicate experiment selection')
        unknown = set(args.experiments) - available.keys()
        if unknown:
            parser.error(f'Unknown experiments: {sorted(unknown)}')
        jobs = [available[key] for key in args.experiments]
    if args.plan:
        print(json.dumps(jobs, indent=2))
        return
    if not args.output or not args.checkpoint:
        parser.error('--output and --checkpoint are required to run')
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    manifest = dict(checkpoint_sha256=checkpoint_hash, model='hf_llm_exo',
                    rolling_window_mode=True, dfa_smc=args.dfa_smc, samples_per_user=32, jobs=jobs)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for job in jobs:
        name = f"{job['cohort']}_offset{job['offset']}_time2vec_rolling_window_mode"
        if args.dfa_smc:
            name += '_dfa_smc'
        dest = args.output/name
        if dest.exists():
            raise FileExistsError(f'Refusing to overwrite {dest}; use a fresh suite folder')
        dest.mkdir()
        command = [sys.executable, '-m', 'scripts.run_hf_experiment', '--cohort', job['cohort'],
                   '--offset', str(job['offset']), '--baseline', job['baseline'],
                   '--checkpoint', str(args.checkpoint), '--output', str(dest), '--users', '0', '--rolling_window_mode']
        if args.dfa_smc:
            command.append('--dfa-smc')
        (dest/'command.json').write_text(json.dumps(command, indent=2)+'\n')
        print(f'START {name}: {job["users"]} users', flush=True)
        with (dest/'console.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=args.repo)
        (dest/'exit_code.txt').write_text(str(result.returncode)+'\n')
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {dest}/console.log')
        print(f'DONE {name}', flush=True)
    (args.output/'complete.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
