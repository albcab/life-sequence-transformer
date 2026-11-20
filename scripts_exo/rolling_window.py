"""Unconstrained GPT-2 sampling with year-aligned context eviction."""
import torch


def crop_context(tokens, background, eoy, target):
    """Retain background and newest complete-year suffix; fallback for oversized years."""
    if len(tokens) <= target:
        return list(tokens), False, False, list(range(len(tokens)))
    assert tokens[:len(background)] == background
    capacity = target - len(background)
    if capacity < 1:
        raise ValueError('Background exceeds rolling context capacity')
    tail = tokens[len(background):]
    for i, token in enumerate(tail):
        if token == eoy and len(tail) - i - 1 <= capacity:
            keep = list(range(len(background))) + list(range(len(background)+i+1, len(tokens)))
            return background + tail[i+1:], True, False, keep
    # A single unfinished year can itself exceed the context. Record this exception.
    keep = list(range(len(background))) + list(range(len(tokens)-capacity, len(tokens)))
    return background + tail[-capacity:], True, True, keep

@torch.inference_mode()
def rolling_sample(model, encoded, background, eoy, eol, eos, pad, years,
                   prefix_years, prefix_ages, prefix_temporal_mask,
                   batch_size=8, max_new_tokens=8192):
    device = next(model.parameters()).device
    limit = model.model.config.n_positions
    target = min(768, limit - 1)
    contexts = [list(encoded) for _ in range(batch_size)]
    context_years = [list(prefix_years) for _ in contexts]
    context_ages = [list(prefix_ages) for _ in contexts]
    context_temporal_masks = [list(prefix_temporal_mask) for _ in contexts]
    generated = [[] for _ in contexts]
    counts = [0] * batch_size
    ends = [None] * batch_size
    shifts = [0] * batch_size
    fallbacks = [0] * batch_size
    past = None
    attention = None
    next_tokens = None
    for step in range(max_new_tokens):
        if past is None or attention.shape[1] >= limit:
            for i in range(batch_size):
                if ends[i] is None:
                    # Initial short contexts are retained intact; later rebuilds free room.
                    if step > 0 or len(contexts[i]) >= limit:
                        cropped, moved, fallback, keep = crop_context(contexts[i], background, eoy, target)
                        contexts[i] = cropped
                        context_years[i] = [context_years[i][j] for j in keep]
                        context_ages[i] = [context_ages[i][j] for j in keep]
                        context_temporal_masks[i] = [context_temporal_masks[i][j] for j in keep]
                        shifts[i] += int(moved)
                        fallbacks[i] += int(fallback)
                else:
                    contexts[i] = list(background)
                    context_years[i] = list(prefix_years[:len(background)])
                    context_ages[i] = list(prefix_ages[:len(background)])
                    context_temporal_masks[i] = list(prefix_temporal_mask[:len(background)])
            width = max(map(len, contexts))
            assert width < limit
            inputs = torch.full((batch_size, width), pad, dtype=torch.long, device=device)
            attention = torch.zeros_like(inputs)
            year_inputs = torch.zeros_like(inputs)
            age_inputs = torch.zeros_like(inputs)
            temporal_mask = torch.zeros_like(inputs)
            for i, context in enumerate(contexts):
                n = len(context)
                inputs[i, -n:] = torch.tensor(context, device=device)
                attention[i, -n:] = 1
                year_inputs[i, -n:] = torch.tensor(context_years[i], device=device)
                age_inputs[i, -n:] = torch.tensor(context_ages[i], device=device)
                temporal_mask[i, -n:] = torch.tensor(context_temporal_masks[i], device=device)
            positions = (attention.cumsum(-1)-1).clamp(min=0)
            embeds = model.embed_tokens(inputs, year_inputs, age_inputs, temporal_mask)
            output = model.model(inputs_embeds=embeds, attention_mask=attention,
                                 position_ids=positions, use_cache=True)
        else:
            active = torch.tensor([end is None for end in ends], device=device, dtype=torch.long)
            attention = torch.cat([attention, active[:, None]], dim=1)
            positions = (attention.sum(-1)-1).clamp(min=0)[:, None]
            year_inputs = torch.tensor([context_years[i][-1] for i in range(batch_size)], device=device)[:, None]
            age_inputs = torch.tensor([context_ages[i][-1] for i in range(batch_size)], device=device)[:, None]
            temporal_mask = torch.tensor([context_temporal_masks[i][-1] for i in range(batch_size)], device=device)[:, None]
            embeds = model.embed_tokens(next_tokens[:, None], year_inputs, age_inputs, temporal_mask)
            output = model.model(inputs_embeds=embeds, attention_mask=attention,
                                 position_ids=positions, past_key_values=past, use_cache=True)
        past = output.past_key_values
        next_tokens = torch.multinomial(torch.softmax(output.logits[:, -1, :], dim=-1), 1).squeeze(1)
        for i, token in enumerate(next_tokens.tolist()):
            if ends[i] is not None:
                next_tokens[i] = pad
                continue
            generated[i].append(token)
            contexts[i].append(token)
            increment = int(token == eoy)
            context_years[i].append(context_years[i][-1] + increment)
            context_ages[i].append(context_ages[i][-1] + increment)
            context_temporal_masks[i].append(1)
            counts[i] += increment
            if token == eol:
                ends[i] = 'EOL'
            elif token == eos:
                ends[i] = 'native_EOS'
            elif counts[i] >= years:
                ends[i] = 'year_limit'
        if all(end is not None for end in ends):
            break
    return [dict(tokens=tokens, end=end or 'token_budget', shifts=shift,
                 boundary_fallbacks=fallback) for tokens, end, shift, fallback
            in zip(generated, ends, shifts, fallbacks)]
