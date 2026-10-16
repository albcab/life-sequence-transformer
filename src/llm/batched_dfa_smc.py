"""Experimental cross-person batching; only transformer inference is shared.

Each person retains its own random generator, SMC population, rolling schedule,
attention cache and normalization. Ragged caches are left-padded only for a
forward call and immediately split back into their original populations.
"""
import torch
import torch.nn.functional as F
from src.llm.dfa_smc import gather_cache
from scripts.rolling_window import crop_context


class Population:
    def __init__(self, model, tokenizer, adapter, person, particles, seed):
        self.n = particles
        self.adapter = adapter
        self.pad = tokenizer.pad_token_id
        self.device = next(model.parameters()).device
        self.limit = model.config.n_positions
        encoded = person['encoded']
        self.background = encoded[:encoded.index(tokenizer.convert_tokens_to_ids('[BOL]'))+1]
        self.contexts = [list(encoded) for _ in range(particles)]
        self.generated = [[] for _ in range(particles)]
        self.states = [adapter.initial(person['prefix'])]*particles
        self.years = person['years']
        self.counts = [0]*particles
        terminal = person['prefix'][-1] == adapter.words['[EOL]']
        self.ends = ['EOL' if terminal else None for _ in range(particles)]
        self.shifts = [0]*particles
        self.fallbacks = [0]*particles
        self.weights = torch.zeros(particles, dtype=torch.float64, device=self.device)
        self.trajectory = self.weights.clone()
        self.past = self.attention = self.next_tokens = None
        self.steps = self.resamplings = 0
        self.rng = torch.Generator(device=self.device).manual_seed(seed)

    @property
    def done(self):
        return all(end is not None for end in self.ends)

    def prepare(self):
        prefill = self.past is None or self.attention.shape[1] >= self.limit
        if prefill:
            for i in range(self.n):
                if self.ends[i] is not None:
                    self.contexts[i] = list(self.background)
                elif self.steps > 0 or len(self.contexts[i]) >= self.limit:
                    self.contexts[i], moved, fallback = crop_context(
                        self.contexts[i], self.background, self.adapter.eoy, min(768,self.limit-1))
                    self.shifts[i] += int(moved); self.fallbacks[i] += int(fallback)
            width = max(map(len,self.contexts))
            inputs = torch.full((self.n,width),self.pad,dtype=torch.long,device=self.device)
            attention = torch.zeros_like(inputs)
            for i, context in enumerate(self.contexts):
                inputs[i,-len(context):] = torch.tensor(context,device=self.device)
                attention[i,-len(context):] = 1
            positions = (attention.cumsum(-1)-1).clamp(min=0)
            return dict(pop=self,prefill=True,inputs=inputs,attention=attention,positions=positions)
        active = torch.tensor([e is None for e in self.ends],device=self.device,dtype=torch.long)
        attention = torch.cat([self.attention,active[:,None]],dim=1)
        return dict(pop=self,prefill=False,inputs=self.next_tokens[:,None],attention=attention,
                    positions=(attention.sum(-1)-1).clamp(min=0)[:,None])

    def advance(self, logits, threshold):
        log_probs = logits.log_softmax(-1)
        self.next_tokens = torch.full((self.n,),self.pad,device=self.device,dtype=torch.long)
        for i in range(self.n):
            if self.ends[i] is not None:
                continue
            valid = self.adapter.choices(self.states[i])
            if not valid:
                raise RuntimeError('No DFA-valid subword')
            ids = torch.tensor(list(valid),device=self.device)
            mass = torch.logsumexp(log_probs[i,ids],0).double()
            choice = torch.multinomial((log_probs[i,ids]-mass).exp(),1,generator=self.rng)
            token = int(ids[choice].item())
            self.weights[i] += mass; self.trajectory[i] += mass
            self.next_tokens[i] = token
            self.states[i] = valid[token]
            self.contexts[i].append(token); self.generated[i].append(token)
            self.counts[i] += int(token == self.adapter.eoy)
            if token == self.adapter.eol:
                self.ends[i] = 'EOL'
            elif self.counts[i] > self.years:
                self.ends[i] = 'year_limit'
        normalized = self.weights.softmax(0)
        if 1/normalized.square().sum() < threshold*self.n and not self.done:
            ancestors = torch.multinomial(normalized,self.n,replacement=True,generator=self.rng)
            indices = ancestors.tolist()
            for name in ['contexts','generated']:
                old = getattr(self,name)
                setattr(self,name,[old[j].copy() for j in indices])
            for name in ['states','counts','ends','shifts','fallbacks']:
                old = getattr(self,name)
                setattr(self,name,[old[j] for j in indices])
            self.trajectory = self.trajectory[ancestors].clone()
            self.attention = self.attention[ancestors].clone()
            self.next_tokens = self.next_tokens[ancestors].clone()
            self.past = gather_cache(self.past,ancestors)
            self.weights.zero_(); self.resamplings += 1
        self.steps += 1

    def results(self):
        return [dict(tokens=tokens,end=end or 'token_budget',shifts=shift,boundary_fallbacks=fallback,
                     weight=weight,trajectory_log_weight=logw,resamplings=self.resamplings)
                for tokens,end,shift,fallback,weight,logw in zip(self.generated,self.ends,self.shifts,
                    self.fallbacks,self.weights.softmax(0).tolist(),self.trajectory.tolist())]


def forward_group(model, requests):
    """Batch like-kind requests and unpad caches before returning to their owners."""
    prefill = requests[0]['prefill']
    width = max(r['attention'].shape[1] for r in requests)
    attentions = [F.pad(r['attention'],(width-r['attention'].shape[1],0)) for r in requests]
    attention = torch.cat(attentions)
    if prefill:
        inputs = torch.cat([F.pad(r['inputs'],(width-r['inputs'].shape[1],0),value=r['pop'].pad) for r in requests])
        positions = torch.cat([F.pad(r['positions'],(width-r['positions'].shape[1],0)) for r in requests])
        output = model(input_ids=inputs,attention_mask=attention,position_ids=positions,use_cache=True)
    else:
        inputs = torch.cat([r['inputs'] for r in requests])
        positions = torch.cat([r['positions'] for r in requests])
        past = tuple(tuple(torch.cat([F.pad(r['pop'].past[layer][kind],
                    (0,0,width-r['attention'].shape[1],0)) for r in requests])
                    for kind in range(2)) for layer in range(len(requests[0]['pop'].past)))
        output = model(input_ids=inputs,attention_mask=attention,position_ids=positions,past_key_values=past,use_cache=True)
    cache = output.past_key_values
    if hasattr(cache,'to_legacy_cache'):
        cache = cache.to_legacy_cache()
    row = 0
    results = []
    for request in requests:
        pop = request['pop']; stop = row+pop.n
        pad = width-request['attention'].shape[1]
        pop.past = tuple(tuple(t[row:stop,:,pad:,:].contiguous() for t in layer) for layer in cache)
        pop.attention = request['attention']
        results.append((pop,output.logits[row:stop,-1,:].clone()))
        row = stop
    return results


@torch.inference_mode()
def sample_people(model,tokenizer,adapter,people,nb_particles=32,ess_threshold=.5,max_new_tokens=8192,seed=2025):
    if not people or nb_particles<1 or not 0<ess_threshold<=1:
        raise ValueError('Invalid population settings')
    if len({p['id'] for p in people}) != len(people):
        raise ValueError('Duplicate people')
    populations = [Population(model,tokenizer,adapter,p,nb_particles,seed+int(p['id'])) for p in people]
    for _ in range(max_new_tokens):
        active = [pop for pop in populations if not pop.done]
        if not active:
            break
        requests = [pop.prepare() for pop in active]
        for prefill in [True,False]:
            group = [r for r in requests if r['prefill']==prefill]
            if group:
                for pop, logits in forward_group(model,group):
                    pop.advance(logits,ess_threshold)
    return [pop.results() for pop in populations]
