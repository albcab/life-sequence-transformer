"""Unconstrained GPT-2 sampling with year-aligned context eviction."""
import torch


def crop_context(tokens, background, eoy, target):
    """Retain background and newest complete-year suffix; fallback for oversized years."""
    if len(tokens) <= target:
        return list(tokens), False, False
    assert tokens[:len(background)] == background
    capacity = target - len(background)
    if capacity < 1:
        raise ValueError('Background exceeds rolling context capacity')
    tail = tokens[len(background):]
    for i, token in enumerate(tail):
        if token == eoy and len(tail) - i - 1 <= capacity:
            return background + tail[i+1:], True, False
    # A single unfinished year can itself exceed the context. Record this exception.
    return background + tail[-capacity:], True, True


@torch.inference_mode()
def rolling_sample(model, encoded, background, eoy, eol, eos, pad, years,
                   batch_size=8, max_new_tokens=8192):
    device = next(model.parameters()).device
    limit = model.config.n_positions
    target = min(768, limit - 1)
    contexts = [list(encoded) for _ in range(batch_size)]
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
                        contexts[i], moved, fallback = crop_context(contexts[i], background, eoy, target)
                        shifts[i] += int(moved)
                        fallbacks[i] += int(fallback)
                else:
                    contexts[i] = list(background)
            width = max(map(len, contexts))
            assert width < limit
            inputs = torch.full((batch_size, width), pad, dtype=torch.long, device=device)
            attention = torch.zeros_like(inputs)
            for i, context in enumerate(contexts):
                inputs[i, -len(context):] = torch.tensor(context, device=device)
                attention[i, -len(context):] = 1
            positions = (attention.cumsum(-1)-1).clamp(min=0)
            output = model(input_ids=inputs, attention_mask=attention,
                           position_ids=positions, use_cache=True)
        else:
            active = torch.tensor([end is None for end in ends], device=device, dtype=torch.long)
            attention = torch.cat([attention, active[:, None]], dim=1)
            positions = (attention.sum(-1)-1).clamp(min=0)[:, None]
            output = model(input_ids=next_tokens[:, None], attention_mask=attention,
                           position_ids=positions, past_key_values=past, use_cache=True)
        past = output.past_key_values
        next_tokens = torch.multinomial(torch.softmax(output.logits[:, -1, :], dim=-1), 1).squeeze(1)
        for i, token in enumerate(next_tokens.tolist()):
            if ends[i] is not None:
                next_tokens[i] = pad
                continue
            generated[i].append(token)
            contexts[i].append(token)
            counts[i] += int(token == eoy)
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
