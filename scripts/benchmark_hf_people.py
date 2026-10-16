"""Small paired throughput study; no production result files are modified."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import hydra
from hydra.utils import instantiate
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM
from src.llm.dfa_smc import SubwordDFA
from src.llm.batched_dfa_smc import sample_people
from scripts.check_hf_grammar import check_text


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint',required=True)
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(2025)
    with hydra.initialize_config_dir(config_dir='/usr/src/conf',version_base=None):
        cfg=hydra.compose(config_name='config',overrides=['experiment=hf_llm'])
    tok=instantiate(cfg.tokenizer)
    data=instantiate(cfg.datamodule,tokenizer=tok,_convert_='all')
    vocab=data.vocabulary
    adapter=SubwordDFA(tok,vocab.token2index,cache_dir=out/'grammar_cache')
    cohort=pd.read_csv('/usr/src/mothers_test_ids_clean.csv').head(4)
    people=[]
    base=Path('/usr/src/generated_smc-batched-20260922T083211Z/vno12_norm_pos_100sp/decoder_only/mothers_test_ids_clean/0')
    for row in cohort.itertuples(index=False):
        saved=np.loadtxt(base/f'{row.USER_ID}_index.csv',delimiter=',',dtype=int)
        start=int(np.flatnonzero(saved[1])[0]);prefix=saved[0,:start].tolist()
        prompt=' '.join(vocab.index2token[t] for t in prefix)
        people.append(dict(id=int(row.USER_ID),prefix=prefix,encoded=tok.encode(prompt),prompt=prompt,years=2))
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=AutoModelForCausalLM.from_pretrained(cfg.model_name,local_files_only=True)
    model.resize_token_embeddings(len(tok))
    model.load_state_dict({k.removeprefix('model.'):v for k,v in checkpoint['state_dict'].items() if k.startswith('model.')},strict=True)
    del checkpoint
    model.eval().to('cuda')
    metadata=dict(checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        users=[dict(id=p['id'],prefix_bpe=len(p['encoded'])) for p in people],particles=32,
        generated_year_upper_bound=3,legacy_num_years=2,seed=2025,orders=[[1,2,4],[4,2,1]],
        device=torch.cuda.get_device_name(),note='Fixed three-year horizon, private warm grammar cache, per-person RNG seeds, no change to GPU 2/3 production jobs.')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Warm-up: four real users, three generated years, 32 particles/person',flush=True)
    for person in people:
        sample_people(model,tok,adapter,[person],max_new_tokens=2048)
    measurements=[];reference=None
    for repeat,order in enumerate(metadata['orders'],start=1):
        for size in order:
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            before=adapter.cache_stats();start=time.perf_counter();results=[]
            print(f'START repeat={repeat} people_per_batch={size}',flush=True)
            try:
                for first in range(0,len(people),size):
                    results.extend(sample_people(model,tok,adapter,people[first:first+size],max_new_tokens=2048))
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                measurement=dict(repeat=repeat,people_per_batch=size,seconds=elapsed,
                    people_per_minute=len(people)*60/elapsed,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    new_cache_states=adapter.cache_stats()['computed_states']-before['computed_states'])
                statuses=Counter();ends=Counter();shifts=0
                for person,rows in zip(people,results):
                    assert abs(sum(r['weight'] for r in rows)-1)<1e-6
                    for r in rows:
                        text=person['prompt']+tok.decode(r['tokens'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                        statuses[check_text(text,vocab.token2index,adapter.dfa,capped=r['end']=='token_budget')['status']]+=1
                        ends[r['end']]+=1;shifts+=r['shifts']
                measurement.update(status=dict(statuses),termination=dict(ends),window_shifts=shifts)
                if reference is None:reference=results
                measurement['different_trajectories_from_batch1']=sum(a['tokens']!=b['tokens'] for arows,brows in zip(reference,results) for a,b in zip(arows,brows))
                measurement['max_weight_difference_from_batch1']=max(abs(a['weight']-b['weight']) for arows,brows in zip(reference,results) for a,b in zip(arows,brows))
                (out/f'samples_repeat{repeat}_batch{size}.json').write_text(json.dumps(results)+'\n')
            except torch.cuda.OutOfMemoryError as error:
                measurement=dict(repeat=repeat,people_per_batch=size,error='CUDA out of memory',detail=str(error))
                gc.collect();torch.cuda.empty_cache()
            measurements.append(measurement)
            (out/'measurements.json').write_text(json.dumps(measurements,indent=2)+'\n')
            print(json.dumps(measurement),flush=True)
    summary=[]
    for size in [1,2,4]:
        rows=[m for m in measurements if m['people_per_batch']==size and 'error' not in m]
        summary.append(dict(people_per_batch=size,successful_repeats=len(rows),
            median_people_per_minute=statistics.median(m['people_per_minute'] for m in rows) if rows else None,
            peak_allocated_gib=max((m['peak_allocated_gib'] for m in rows),default=None)))
    (out/'summary.json').write_text(json.dumps(dict(metadata=metadata,measurements=measurements,summary=summary),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
