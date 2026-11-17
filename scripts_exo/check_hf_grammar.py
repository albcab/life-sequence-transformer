
"""Bounded, unmasked HF generation diagnostic on real validation prefixes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import hydra
from hydra.utils import get_class, instantiate
import numpy as np
import torch

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

@torch.inference_mode()
def generate_temporal(model, input_ids, attention_mask, years, ages, temporal_mask,
                      eoy, stop_ids, max_new_tokens, do_sample):
    device = input_ids.device
    B = input_ids.shape[0]
    past = None
    generated = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=device)
    next_tokens = None
    next_years = years[:, -1:].clone()
    next_ages = ages[:, -1:].clone()
    next_temporal_mask = temporal_mask[:, -1:].clone()

    for step in range(max_new_tokens):
        if past is None:
            embeds = model.embed_tokens(input_ids, years, ages, temporal_mask)
            output = model.model(inputs_embeds=embeds, attention_mask=attention_mask,
                                 position_ids=(attention_mask.cumsum(-1)-1).clamp(min=0), use_cache=True)
        else:
            attention_mask = torch.cat([attention_mask, (~done).long()[:, None]], dim=1)
            embeds = model.embed_tokens(next_tokens[:, None], next_years, next_ages, next_temporal_mask)
            output = model.model(inputs_embeds=embeds, attention_mask=attention_mask,
                                 position_ids=(attention_mask.sum(-1)-1).clamp(min=0)[:, None],
                                 past_key_values=past, use_cache=True)
        past = output.past_key_values
        logits = output.logits[:, -1, :]
        next_tokens = (torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)
                       if do_sample else logits.argmax(dim=-1))

        for i, token in enumerate(next_tokens.tolist()):
            if done[i]:
                next_tokens[i] = model.model.config.pad_token_id
                continue
            generated[i].append(token)
            if token in stop_ids:
                done[i] = True

        # [EOY] retains the previous year; the following token advances it.
        next_years = next_years + (next_tokens[:, None] == eoy).long()
        next_ages = next_ages + (next_tokens[:, None] == eoy).long()

        if done.all():
            break

    return generated

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
        cfg = hydra.compose(config_name="config", overrides=["experiment=hf_llm_exo"])
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
        prefix = full[:cut]
        tokens = [vocabulary.index2token[v] for v in prefix]
        encoded = tokenizer(tokens, is_split_into_words=True, return_tensors='pt')
        if encoded['input_ids'].shape[1] > 256:
            exclusions['prefix_over_256_bpe_tokens'] += 1
            continue
        if check_ids(prefix, dfa)["status"] == "invalid":
            exclusions['invalid_real_prefix'] += 1
            continue

        length = int(sample.padding_mask.sum())
        yearseq = np.append(sample.input_ids[1, :length], sample.input_ids[1, length-1])[:cut]
        ageseq = np.append(sample.input_ids[2, :length], sample.input_ids[2, length-1])[:cut]

        years = torch.zeros_like(encoded['input_ids'])
        ages = torch.zeros_like(encoded['input_ids'])
        temporal_mask = torch.zeros_like(encoded['input_ids'])
        background_length = prefix.index(1) + 1

        for j, word_id in enumerate(encoded.word_ids(batch_index=0)):
            if word_id is not None:
                years[0, j] = int(yearseq[word_id])
                ages[0, j] = int(ageseq[word_id])
                temporal_mask[0, j] = int(word_id >= background_length)

        prompt = " ".join(tokens)
        selected.append({"id": int(sample.sequence_id), "dataset_index": index,
                         "prompt": prompt, "prefix_ids": prefix, "full_ids": full,
                         "encoded": encoded['input_ids'][0].tolist(),
                         "years": years[0].tolist(), "ages": ages[0].tolist(),
                         "temporal_mask": temporal_mask[0].tolist(),
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
    ModelClass = get_class(cfg.model._target_)
    model = ModelClass.load_from_checkpoint(
        args.checkpoint, strict=True, hparams=cfg.model.hparams, tokenizer=tokenizer
    )
    meta = {"checkpoint": args.checkpoint, "sha256": digest.hexdigest(),
            "epoch_zero_based": checkpoint['epoch'], "global_step": checkpoint['global_step'],
            "users": len(selected), "split": "val", "seed": 2025,
            "prefix_rule": "first EOY after at least 20 domain tokens; at most 256 BPE tokens",
            "max_new_bpe_tokens": args.max_new_tokens, "prefix_exclusions": dict(exclusions),
            "gold_full_status": dict(Counter(x['gold_full']['status'] for x in selected))}
    del checkpoint
    model.eval().to('cuda')
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('[EOL]')]
    eoy = tokenizer.convert_tokens_to_ids('[EOY]')
    results = []
    with (out/'samples.jsonl').open('w') as stream, torch.inference_mode():
        for mode in ('greedy', 'sample'):
            for start in range(0, len(selected), 8):
                users = selected[start:start+8]
                width = max(len(x['encoded']) for x in users)
                inputs = torch.full((len(users), width), tokenizer.pad_token_id, dtype=torch.long, device='cuda')
                attention = torch.zeros_like(inputs)
                years = torch.zeros_like(inputs)
                ages = torch.zeros_like(inputs)
                temporal_mask = torch.zeros_like(inputs)

                for i, user in enumerate(users):
                    n = len(user['encoded'])
                    inputs[i, -n:] = torch.tensor(user['encoded'], device='cuda')
                    attention[i, -n:] = 1
                    years[i, -n:] = torch.tensor(user['years'], device='cuda')
                    ages[i, -n:] = torch.tensor(user['ages'], device='cuda')
                    temporal_mask[i, -n:] = torch.tensor(user['temporal_mask'], device='cuda')

                sequences = generate_temporal(
                    model, inputs, attention, years, ages, temporal_mask,
                    eoy, stop_ids, args.max_new_tokens, mode == 'sample'
                )
                for user, new in zip(users, sequences):
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
