"""Bounded, unmasked HF generation diagnostic on real validation prefixes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import hydra
from hydra.utils import instantiate
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from src.models.dfa import LIFESEQUENCEDFA, evaluate_matcher


def check_ids(ids, dfa):
    configs = {(dfa.initial_state, ())}
    for position, token in enumerate(ids):
        following = set()
        for state, prefix in configs:
            for matcher, target in dfa.transitions_by_state.get(state, ()):
                satisfied, extendable, fixable = evaluate_matcher(matcher, list(prefix) + [token])
                if satisfied:
                    following.add((target, ()))
                    if extendable:
                        following.add((state, prefix + (token,)))
                elif fixable:
                    following.add((state, prefix + (token,)))
        configs = following
        if not configs:
            return {"status": "invalid", "failure_position": position, "failure_id": int(token)}
    return {"status": "accepted" if any(s in dfa.accepting_states and not p for s, p in configs)
            else "valid_prefix"}


def check_text(text, token2index, dfa, capped=False):
    words = text.strip().split()
    partial_tail = None
    # A BPE budget can cut a domain token in half: do not call that a grammar error.
    if (capped and words and words[-1] not in token2index and not text[-1:].isspace()
            and any(token.startswith(words[-1]) for token in token2index)):
        partial_tail = words.pop()
    unknown = [(i, word) for i, word in enumerate(words) if word not in token2index]
    if unknown:
        return {"status": "unknown_token", "unknown": unknown[:3], "tokens": len(words)}
    result = check_ids([token2index[word] for word in words], dfa)
    result.update(tokens=len(words), partial_tail=partial_tail)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--users", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    with hydra.initialize_config_dir(config_dir=str(Path('/usr/src/conf')), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=["experiment=hf_llm"])
    tokenizer = instantiate(cfg.tokenizer)
    data = instantiate(cfg.datamodule, tokenizer=tokenizer, _convert_="all")
    tokenizer.padding_side = "left"
    dataset = data.get_dataset("val")
    vocabulary = data.vocabulary
    dfa = LIFESEQUENCEDFA()

    # Matcher sanity checks, independent of any model output.
    for ids in ([5, 13, 19, 31, 11, 1, 19, 128, 3], [5, 13, 19, 31, 11, 1, 19, 128, 2]):
        assert check_ids(ids, dfa)["status"] == "accepted"
        assert bool(dfa.get_initial_conf(ids))
    assert check_ids([5, 13, 19, 31, 11, 1, 19, 128, 133], dfa)["status"] == "invalid"

    selected, exclusions = [], Counter()
    for index in random.sample(range(len(dataset)), min(len(dataset), 1000)):
        sample = dataset[index]
        full = [int(v) for v in sample.original_sequence if v != 0]
        boundaries = [i + 1 for i, v in enumerate(full) if v == 3 and i >= 20]
        if not boundaries:
            exclusions['no_year_boundary_after_20_tokens'] += 1
            continue
        cut = boundaries[0]
        prompt = " ".join(vocabulary.index2token[v] for v in full[:cut])
        if len(tokenizer.encode(prompt)) > 256:
            exclusions['prefix_over_256_bpe_tokens'] += 1
            continue
        if check_ids(full[:cut], dfa)["status"] == "invalid":
            exclusions['invalid_real_prefix'] += 1
            continue
        selected.append({"id": int(sample.sequence_id), "dataset_index": index,
                         "prompt": prompt, "prefix_ids": full[:cut], "full_ids": full,
                         "gold_full": check_ids(full, dfa)})
        if len(selected) == args.users:
            break
    if len(selected) != args.users:
        raise RuntimeError(f"Only found {len(selected)} usable prefixes; exclusions={exclusions}")

    digest = hashlib.sha256()
    with open(args.checkpoint, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, local_files_only=True)
    model.resize_token_embeddings(len(tokenizer))
    state = {key.removeprefix('model.'): value for key, value in checkpoint['state_dict'].items()
             if key.startswith('model.')}
    model.load_state_dict(state, strict=True)
    meta = {"checkpoint": args.checkpoint, "sha256": digest.hexdigest(),
            "epoch_zero_based": checkpoint['epoch'], "global_step": checkpoint['global_step'],
            "users": len(selected), "split": "val", "seed": 2025,
            "prefix_rule": "first EOY after at least 20 domain tokens; at most 256 BPE tokens",
            "max_new_bpe_tokens": args.max_new_tokens, "prefix_exclusions": dict(exclusions),
            "gold_full_status": dict(Counter(x['gold_full']['status'] for x in selected))}
    del checkpoint, state
    model.eval().to('cuda')
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('[EOL]')]
    results = []
    with (out/'samples.jsonl').open('w') as stream, torch.inference_mode():
        for mode in ('greedy', 'sample'):
            for start in range(0, len(selected), 8):
                users = selected[start:start+8]
                inputs = tokenizer([x['prompt'] for x in users], padding=True, return_tensors='pt').to('cuda')
                options = {"do_sample": mode == 'sample', "max_new_tokens": args.max_new_tokens,
                           "eos_token_id": stop_ids, "pad_token_id": tokenizer.pad_token_id,
                           "num_beams": 1, "use_cache": True}
                if mode == 'sample':
                    options.update(temperature=1., top_k=0, top_p=1.)
                sequences = model.generate(**inputs, **options)
                for user, row in zip(users, sequences):
                    new = row[inputs['input_ids'].shape[1]:].tolist()
                    stop = next((i for i, token in enumerate(new) if token in stop_ids), None)
                    end = 'budget' if stop is None else ('EOL' if new[stop] == stop_ids[1] else 'native_EOS')
                    if stop is not None:
                        new = new[:stop+1] if end == 'EOL' else new[:stop]
                    text = user['prompt'] + tokenizer.decode(new, skip_special_tokens=False,
                                                            clean_up_tokenization_spaces=False)
                    assessment = check_text(text, vocabulary.token2index, dfa, capped=end == 'budget')
                    # Check actual data over a comparable domain-token horizon as a control.
                    gold_cut = min(len(user['full_ids']), assessment.get('tokens', len(text.split())))
                    gold = check_ids(user['full_ids'][:gold_cut], dfa)
                    result = {"id": user['id'], "mode": mode, "end": end, "bpe_tokens": len(new),
                              "prefix_tokens": len(user['prefix_ids']), "text": text,
                              "assessment": assessment, "gold_matched_horizon": gold,
                              "gold_full": user['gold_full']}
                    stream.write(json.dumps(result)+'\n')
                    stream.flush()
                    results.append(result)
                print(f"{mode}: {min(start+8,len(selected))}/{len(selected)} users", flush=True)
    meta['modes'] = {}
    for mode in ('greedy', 'sample'):
        rows = [r for r in results if r['mode'] == mode]
        meta['modes'][mode] = {
            "status": dict(Counter(r['assessment']['status'] for r in rows)),
            "termination": dict(Counter(r['end'] for r in rows)),
            "partial_bpe_tail": sum(bool(r['assessment'].get('partial_tail')) for r in rows),
            "gold_matched_horizon": dict(Counter(r['gold_matched_horizon']['status'] for r in rows)),
            "mean_generated_bpe_tokens": sum(r['bpe_tokens'] for r in rows)/len(rows),
        }
    (out/'summary.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == '__main__':
    main()
