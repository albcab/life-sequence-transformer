"""GPT-2 subword proposals constrained by the same DFA as the LST sampler.

SMC steps are BPE tokens, not life tokens. Grammar state follows the full
trajectory even when model context rolls. EOL is scored once then frozen.
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from src.llm.dfa_cache import TransitionCache
import torch
from src.models.dfa import LIFESEQUENCEDFA, evaluate_matcher
from scripts.rolling_window import crop_context


class SubwordDFA:
    def __init__(self, tokenizer, token2index, cache_dir=None):
        self.dfa = LIFESEQUENCEDFA()
        self.words = {str(k): int(v) for k, v in token2index.items()}
        self.eoy = tokenizer.convert_tokens_to_ids('[EOY]')
        self.eol = tokenizer.convert_tokens_to_ids('[EOL]')
        self.trie = {}
        pieces = []
        alphabet = set(''.join(self.words)) | {' '}
        for idx in range(len(tokenizer)):
            text = tokenizer.decode([idx], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            pieces.append(text)
            if not text or not set(text) <= alphabet:
                continue
            # Structural specials must be atomic so counting and termination agree
            # with the tokenizer and serialized evaluator outputs.
            if ('[' in text or ']' in text) and idx not in (self.eoy, self.eol):
                continue
            node = self.trie
            for char in text:
                node = node.setdefault(char, {})
            node.setdefault(None, []).append(idx)

        # Invalidate on grammar/matcher code, lexical adapter, storage format,
        # domain IDs, tokenizer decoder/configuration, or special-token changes.
        code = [Path(__file__), Path(__file__).with_name('dfa_cache.py'),
                Path(__file__).parents[1]/'models/dfa.py']
        backend = getattr(tokenizer, 'backend_tokenizer', None)
        identity = dict(schema=1, sources=[hashlib.sha256(p.read_bytes()).hexdigest() for p in code],
                        words=sorted(self.words.items()), pieces=pieces,
                        backend=backend.to_str() if backend is not None else None,
                        eoy=self.eoy, eol=self.eol)
        self.cache_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.disk_cache = None
        if cache_dir is not False:
            directory = cache_dir if cache_dir is not None else os.environ.get(
                'HF_DFA_CACHE', str(Path(__file__).resolve().parents[2]/'.cache/hf_dfa_smc'))
            self.disk_cache = TransitionCache(directory, self.cache_fingerprint)

    def cache_stats(self):
        return dict(fingerprint=self.cache_fingerprint,
                    **(self.disk_cache.stats() if self.disk_cache is not None else {'enabled': False}))

    @lru_cache(maxsize=8192)
    def domain_next(self, configs, token):
        result = set()
        for state, prefix in configs:
            for matcher, target in self.dfa.transitions_by_state.get(state, ()):
                satisfied, extendable, fixable = evaluate_matcher(matcher, list(prefix) + [token])
                if satisfied:
                    result.add((target, ()))
                    if extendable:
                        result.add((state, prefix + (token,)))
                elif fixable:
                    result.add((state, prefix + (token,)))
        return frozenset(result)

    @lru_cache(maxsize=4096)
    def words_after(self, configs):
        allowed = {w for w, idx in self.words.items() if self.domain_next(configs, idx)}
        prefixes = {w[:n] for w in allowed for n in range(1, len(w)+1)}
        return allowed, prefixes

    def initial(self, domain_prefix):
        configs = frozenset((s, tuple(p)) for s, p in self.dfa.get_initial_conf(domain_prefix, verbose=False))
        if not configs:
            raise ValueError('Observed prefix is invalid under LIFESEQUENCEDFA')
        return configs, '', True

    @lru_cache(maxsize=65536)
    def char_next(self, state, char):
        configs, word, needs_space = state
        if char == ' ':
            if word:
                if word not in self.words:
                    return None
                configs = self.domain_next(configs, self.words[word])
                if not configs:
                    return None
            return configs, '', False
        if needs_space:
            return None
        word += char
        if word not in self.words_after(configs)[1]:
            return None
        return configs, word, False

    @lru_cache(maxsize=8192)
    def choices(self, state):
        if self.disk_cache is not None:
            cached = self.disk_cache.get(state)
            if cached is not None:
                return cached
        result = {}
        stack = [(self.trie, state)]
        while stack:
            node, current = stack.pop()
            for idx in node.get(None, ()):
                following = current
                if idx in (self.eoy, self.eol):
                    configs, word, _ = current
                    if word not in self.words:
                        continue
                    configs = self.domain_next(configs, self.words[word])
                    if not configs:
                        continue
                    following = (configs, '', True)
                result[idx] = following
            for char, child in node.items():
                if char is None:
                    continue
                following = self.char_next(current, char)
                if following is not None:
                    stack.append((child, following))
        if self.disk_cache is not None:
            self.disk_cache.put(state, result)
        return result


def gather_cache(cache, ancestors):
    if hasattr(cache, 'reorder_cache'):
        cache.reorder_cache(ancestors)
        return cache
    return tuple(tuple(value.index_select(0, ancestors) for value in layer) for layer in cache)


@torch.inference_mode()
def sample(model, tokenizer, adapter, encoded, domain_prefix, num_years,
           nb_particles=32, ess_threshold=.5, max_new_tokens=8192,
           verbose=False):
    if nb_particles < 1 or not 0 < ess_threshold <= 1 or num_years < 0:
        raise ValueError('Invalid particle count, ESS threshold, or year limit')
    if max_new_tokens < 1:
        raise ValueError('max_new_tokens must be positive')
    initial = adapter.initial(domain_prefix)
    conditional = getattr(adapter, 'conditional', False)
    allow_failure = getattr(adapter, 'allow_particle_failure', False)
    if conditional and not adapter.choices_with_budget(initial, max_new_tokens):
        raise ValueError('No jointly accepting continuation within the BPE token budget')
    device = next(model.parameters()).device
    bol = tokenizer.convert_tokens_to_ids('[BOL]')
    background = encoded[:encoded.index(bol)+1]
    limit = model.config.n_positions
    contexts = [list(encoded) for _ in range(nb_particles)]
    generated = [[] for _ in contexts]
    model_scores = [[] for _ in contexts]
    proposal_scores = [[] for _ in contexts]
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
                elif step > 0 or len(contexts[i]) >= limit:
                    contexts[i], moved, fallback = crop_context(contexts[i], background, adapter.eoy, min(768, limit-1))
                    shifts[i] += int(moved); fallbacks[i] += int(fallback)
            width = max(map(len, contexts))
            inputs = torch.full((nb_particles,width),tokenizer.pad_token_id,dtype=torch.long,device=device)
            attention = torch.zeros_like(inputs)
            for i, context in enumerate(contexts):
                inputs[i,-len(context):] = torch.tensor(context,device=device)
                attention[i,-len(context):] = 1
            output = model(input_ids=inputs, attention_mask=attention,
                           position_ids=(attention.cumsum(-1)-1).clamp(min=0), use_cache=True)
        else:
            active = torch.tensor([end is None for end in ends],dtype=torch.long,device=device)
            attention = torch.cat([attention,active[:,None]],dim=1)
            output = model(input_ids=next_tokens[:,None],attention_mask=attention,
                position_ids=(attention.sum(-1)-1).clamp(min=0)[:,None],past_key_values=past,use_cache=True)
        past = output.past_key_values
        log_probs = torch.log_softmax(output.logits[:,-1,:],dim=-1)
        next_tokens = torch.full((nb_particles,),tokenizer.pad_token_id,dtype=torch.long,device=device)
        for i in range(nb_particles):
            if ends[i] is not None:
                continue
            valid = (adapter.choices_with_budget(states[i], max_new_tokens - step)
                     if conditional else adapter.choices(states[i]))
            if not valid and allow_failure:
                ends[i] = 'dead_end'
                weights[i] = -torch.inf
                continue
            if not valid:
                raise RuntimeError(f'No DFA-valid subword for particle {i}, step {step}')
            ids = torch.tensor(list(valid),device=device)
            mass = torch.logsumexp(log_probs[i,ids],dim=0).double()
            chosen = torch.multinomial((log_probs[i,ids]-mass).exp(),1)
            token = int(ids[chosen].item())
            model_scores[i].append(float(log_probs[i, token]))
            proposal_scores[i].append(float(log_probs[i, token].double() - mass))
            weights[i] += mass; trajectory[i] += mass
            next_tokens[i] = token
            states[i] = valid[token]
            contexts[i].append(token); generated[i].append(token)
            counts[i] += int(token == adapter.eoy)
            if conditional and adapter.accepting(states[i]):
                ends[i] = 'EOL' if token == adapter.eol else 'property_horizon'
            elif token == adapter.eol:
                if conditional and not adapter.accepting(states[i]):
                    raise RuntimeError('Conditional sampler reached a nonaccepting EOL')
                ends[i] = 'EOL'
            # Match the existing custom-model sampler's > comparison, not >=.
            elif not conditional and counts[i] > num_years:
                ends[i] = 'year_limit'
        if not torch.isfinite(weights).any():
            break
        normalized = weights.softmax(0)
        ess = 1/normalized.square().sum()
        if ess < ess_threshold*nb_particles and not all(end is not None for end in ends):
            ancestors = torch.multinomial(normalized,nb_particles,replacement=True)
            indices = ancestors.tolist()
            contexts = [contexts[j].copy() for j in indices]
            generated = [generated[j].copy() for j in indices]
            model_scores = [model_scores[j].copy() for j in indices]
            proposal_scores = [proposal_scores[j].copy() for j in indices]
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
    accepted = [not conditional or adapter.accepting(state) for state in states]
    if allow_failure:
        for i, ok in enumerate(accepted):
            if not ok:
                weights[i] = -torch.inf
    if conditional and not allow_failure and not all(accepted):
        raise RuntimeError('Conditional sampler failed to reach joint acceptance')
    extinct = not bool(torch.isfinite(weights).any())
    final = [0.] * nb_particles if extinct else weights.softmax(0).tolist()
    return [dict(tokens=tokens,end=end or 'token_budget',shifts=shift,boundary_fallbacks=fallback,
                 weight=weight,trajectory_log_weight=float(logw),resamplings=resamplings,
                 model_token_log_probs=lp, proposal_token_log_probs=lq, accepted=ok, population_extinct=extinct)
            for tokens,end,shift,fallback,weight,logw,lp,lq,ok in zip(
                generated,ends,shifts,fallbacks,final,trajectory.tolist(),model_scores,proposal_scores,accepted)]
