"""GPT-2 subword proposals constrained by the same DFA as the LST sampler.

SMC steps are BPE tokens, not life tokens. Grammar state follows the full
trajectory even when model context rolls. EOL is scored once then frozen.
"""
import torch
from scripts_exo.rolling_window import crop_context


def gather_cache(cache, ancestors):
    if hasattr(cache, 'reorder_cache'):
        cache.reorder_cache(ancestors)
        return cache
    return tuple(tuple(value.index_select(0, ancestors) for value in layer) for layer in cache)


@torch.inference_mode()
def sample(model, tokenizer, adapter, encoded, domain_prefix, num_years,
           prefix_years, prefix_ages, prefix_temporal_mask,
           nb_particles=32, ess_threshold=.5, max_new_tokens=8192,
           verbose=False):
    if nb_particles < 1 or not 0 < ess_threshold <= 1 or num_years < 0:
        raise ValueError('Invalid particle count, ESS threshold, or year limit')
    initial = adapter.initial(domain_prefix)
    device = next(model.parameters()).device
    bol = tokenizer.convert_tokens_to_ids('[BOL]')
    background = encoded[:encoded.index(bol)+1]
    limit = model.model.config.n_positions
    contexts = [list(encoded) for _ in range(nb_particles)]
    context_years = [list(prefix_years) for _ in contexts]
    context_ages = [list(prefix_ages) for _ in contexts]
    context_temporal_masks = [list(prefix_temporal_mask) for _ in contexts]
    generated = [[] for _ in contexts]
    states = [initial for _ in contexts]
    counts = [0] * nb_particles
    ends = ['EOL' if domain_prefix[-1] == adapter.words['[EOL]'] else None for _ in contexts]
    shifts, fallbacks = [0]*nb_particles, [0]*nb_particles
    weights = torch.zeros(nb_particles, dtype=torch.float64, device=device)
    trajectory = weights.clone()
    past = attention = next_tokens = None
    resamplings = 0
    for step in range(max_new_tokens):
        if all(end is not None for end in ends):
            break
        if past is None or attention.shape[1] >= limit:
            for i in range(nb_particles):
                if ends[i] is not None:
                    contexts[i] = list(background)
                    context_years[i] = list(prefix_years[:len(background)])
                    context_ages[i] = list(prefix_ages[:len(background)])
                    context_temporal_masks[i] = list(prefix_temporal_mask[:len(background)])
                elif step > 0 or len(contexts[i]) >= limit:
                    contexts[i], moved, fallback, keep = crop_context(contexts[i], background, adapter.eoy, min(768, limit-1))
                    context_years[i] = [context_years[i][j] for j in keep]
                    context_ages[i] = [context_ages[i][j] for j in keep]
                    context_temporal_masks[i] = [context_temporal_masks[i][j] for j in keep]
                    shifts[i] += int(moved); fallbacks[i] += int(fallback)
            width = max(map(len, contexts))
            inputs = torch.full((nb_particles,width),tokenizer.pad_token_id,dtype=torch.long,device=device)
            attention = torch.zeros_like(inputs)
            year_inputs = torch.zeros_like(inputs)
            age_inputs = torch.zeros_like(inputs)
            temporal_mask = torch.zeros_like(inputs)
            for i, context in enumerate(contexts):
                n = len(context)
                inputs[i,-n:] = torch.tensor(context,device=device)
                attention[i,-n:] = 1
                year_inputs[i,-n:] = torch.tensor(context_years[i],device=device)
                age_inputs[i,-n:] = torch.tensor(context_ages[i],device=device)
                temporal_mask[i,-n:] = torch.tensor(context_temporal_masks[i],device=device)
            embeds = model.embed_tokens(inputs, year_inputs, age_inputs, temporal_mask)
            output = model.model(inputs_embeds=embeds, attention_mask=attention,
                           position_ids=(attention.cumsum(-1)-1).clamp(min=0), use_cache=True)
        else:
            active = torch.tensor([end is None for end in ends],dtype=torch.long,device=device)
            attention = torch.cat([attention,active[:,None]],dim=1)
            year_inputs = torch.tensor([context_years[i][-1] for i in range(nb_particles)],device=device)[:,None]
            age_inputs = torch.tensor([context_ages[i][-1] for i in range(nb_particles)],device=device)[:,None]
            temporal_mask = torch.tensor([context_temporal_masks[i][-1] for i in range(nb_particles)],device=device)[:,None]
            embeds = model.embed_tokens(next_tokens[:,None], year_inputs, age_inputs, temporal_mask)
            output = model.model(inputs_embeds=embeds,attention_mask=attention,
                position_ids=(attention.sum(-1)-1).clamp(min=0)[:,None],past_key_values=past,use_cache=True)
        past = output.past_key_values
        log_probs = torch.log_softmax(output.logits[:,-1,:],dim=-1)
        next_tokens = torch.full((nb_particles,),tokenizer.pad_token_id,dtype=torch.long,device=device)
        for i in range(nb_particles):
            if ends[i] is not None:
                continue
            valid = adapter.choices(states[i])
            if not valid:
                raise RuntimeError(f'No DFA-valid subword for particle {i}, step {step}')
            ids = torch.tensor(list(valid),device=device)
            mass = torch.logsumexp(log_probs[i,ids],dim=0).double()
            chosen = torch.multinomial((log_probs[i,ids]-mass).exp(),1)
            token = int(ids[chosen].item())
            weights[i] += mass; trajectory[i] += mass
            next_tokens[i] = token
            states[i] = valid[token]
            increment = int(token == adapter.eoy)
            contexts[i].append(token); generated[i].append(token)
            context_years[i].append(context_years[i][-1] + increment)
            context_ages[i].append(context_ages[i][-1] + increment)
            context_temporal_masks[i].append(1)
            counts[i] += increment
            if token == adapter.eol:
                ends[i] = 'EOL'
            # Match the existing custom-model sampler's > comparison, not >=.
            elif counts[i] > num_years:
                ends[i] = 'year_limit'
        normalized = weights.softmax(0)
        ess = 1/normalized.square().sum()
        if ess < ess_threshold*nb_particles and not all(end is not None for end in ends):
            ancestors = torch.multinomial(normalized,nb_particles,replacement=True)
            indices = ancestors.tolist()
            contexts = [contexts[j].copy() for j in indices]
            context_years = [context_years[j].copy() for j in indices]
            context_ages = [context_ages[j].copy() for j in indices]
            context_temporal_masks = [context_temporal_masks[j].copy() for j in indices]
            generated = [generated[j].copy() for j in indices]
            states = [states[j] for j in indices]
            counts = [counts[j] for j in indices]
            ends = [ends[j] for j in indices]
            shifts = [shifts[j] for j in indices]
            fallbacks = [fallbacks[j] for j in indices]
            trajectory = trajectory[ancestors].clone()
            attention = attention[ancestors].clone()
            next_tokens = next_tokens[ancestors].clone()
            past = gather_cache(past, ancestors)
            weights.zero_(); resamplings += 1
        if verbose and step % 100 == 0:
            print(f'SMC step={step}, finished={sum(e is not None for e in ends)}/{nb_particles}, ESS={float(ess):.2f}, resamplings={resamplings}',flush=True)
    final = weights.softmax(0).tolist()
    return [dict(tokens=tokens,end=end or 'token_budget',shifts=shift,boundary_fallbacks=fallback,
                 weight=weight,trajectory_log_weight=float(logw),resamplings=resamplings)
            for tokens,end,shift,fallback,weight,logw in zip(generated,ends,shifts,fallbacks,final,trajectory.tolist())]
