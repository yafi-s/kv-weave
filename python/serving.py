"""CPU microbatch prefill/decode with trained Llama weights and native paged KV.

The MIT llama2.c model implementation supplies embeddings/projections/MLPs and
the independent full-context oracle. KV ownership and attention use KV-Weave.
"""
from dataclasses import dataclass, field
import math
from pathlib import Path
import time
import numpy as np
import torch

from native import NativeCache, PrefixConflict
from reference.model import ModelArgs, Transformer


def load_model(path):
    torch.set_num_threads(2)
    checkpoint = torch.load(path,map_location='cpu',weights_only=True)
    model = Transformer(ModelArgs(**checkpoint['model_args']))
    state = {key.removeprefix('_orig_mod.'):value for key,value in checkpoint['model'].items()}
    model.load_state_dict(state)
    model.eval()
    return model


@dataclass
class Request:
    key: str
    prompt: list
    max_new_tokens: int
    tenant: int
    submitted: float
    sequences: list = field(default_factory=list)
    cursor: int = 0
    output: list = field(default_factory=list)
    state: str = 'queued'
    first_token: float | None = None
    finished: float | None = None
    token_latencies_ms: list = field(default_factory=list)
    reused: int = 0
    error: str | None = None


class Engine:
    def __init__(self, model, *, pages=64, page_tokens=16, max_active=4, max_queued=16, sharing=True, cache_class=NativeCache):
        self.model = model
        self.pages, self.page_tokens = pages,page_tokens
        self.max_active,self.max_queued,self.sharing = max_active,max_queued,sharing
        self.requests = {}
        self.reserved = 0
        self.caches = [cache_class(layer.attention.n_local_kv_heads,layer.attention.head_dim,
                                  pages=pages,page_tokens=page_tokens,entries=8 if sharing else 0,
                                  sequences=max_active) for layer in model.layers]
        self.peak_payload_bytes = 0
        self.last_logits = {}

    def submit(self,key,prompt,max_new_tokens=8,tenant=0):
        if key in self.requests:
            raise ValueError('duplicate request key')
        if len(self.requests)>=4096:
            raise RuntimeError('request history bound; create a new engine')
        if (not isinstance(prompt,list) or not prompt or any(type(x) is not int or x<0 or x>=self.model.vocab_size for x in prompt)
                or type(max_new_tokens) is not int or max_new_tokens<1
                or len(prompt)+max_new_tokens>self.model.params.max_seq_len
                or type(tenant) is not int or not 0<=tenant<2**64):
            raise ValueError('invalid request')
        pending = sum(r.state in ('queued','active') for r in self.requests.values())
        needed = math.ceil((len(prompt)+max_new_tokens)/self.page_tokens)
        if pending >= self.max_queued or self.reserved+needed>self.pages:
            raise RuntimeError('bounded admission rejected')
        request = Request(key,prompt.copy(),max_new_tokens,tenant,time.perf_counter())
        self.requests[key]=request
        self.reserved += needed
        return request

    def finish(self,request,state,error=None):
        for cache,seq in zip(self.caches,request.sequences):
            cache.release(seq)
        request.sequences=[]
        request.state,request.error,request.finished=state,error,time.perf_counter()
        self.reserved -= math.ceil((len(request.prompt)+request.max_new_tokens)/self.page_tokens)

    def cancel(self,key):
        request = self.requests[key]
        if request.state not in ('queued','active'):
            return False
        self.finish(request,'cancelled')
        return True

    def activate(self):
        slots = self.max_active-sum(r.state=='active' for r in self.requests.values())
        for request in self.requests.values():
            if request.state!='queued' or slots==0:
                continue
            try:
                reused=[]
                # Always compute final prompt token to recover output logits.
                for cache in self.caches:
                    seq,n=cache.checkout(request.prompt[:-1],request.tenant)
                    request.sequences.append(seq)
                    reused.append(n)
                if len(set(reused))!=1:
                    raise RuntimeError('layer prefix mismatch')
                request.cursor=request.reused=reused[0]
                request.state='active'
                slots-=1
            except Exception as exc:
                self.finish(request,'failed',str(exc))

    @torch.inference_mode()
    def step(self):
        self.activate()
        batch=[r for r in self.requests.values() if r.state=='active']
        if not batch:
            return False
        tokens=[r.prompt[r.cursor] if r.cursor<len(r.prompt) else r.output[-1] for r in batch]
        positions=[r.cursor for r in batch]
        h=self.model.tok_embeddings(torch.tensor(tokens)).unsqueeze(1)
        cos=self.model.freqs_cos[positions][:,None,None,:]
        sin=self.model.freqs_sin[positions][:,None,None,:]
        def rope(x):
            real,imag=x.float().reshape(*x.shape[:-1],-1,2).unbind(-1)
            return torch.stack((real*cos-imag*sin,real*sin+imag*cos),dim=-1).flatten(3)
        started=time.perf_counter()
        try:
            for layer_index,layer in enumerate(self.model.layers):
                att=layer.attention
                normalized=layer.attention_norm(h)
                q=rope(att.wq(normalized).reshape(len(batch),1,att.n_local_heads,att.head_dim))
                k=rope(att.wk(normalized).reshape(len(batch),1,att.n_local_kv_heads,att.head_dim))
                v=att.wv(normalized).reshape(len(batch),1,att.n_local_kv_heads,att.head_dim)
                cache=self.caches[layer_index]
                attended=[]
                for i,r in enumerate(batch):
                    seq=r.sequences[layer_index]
                    cache.append(seq,tokens[i],k[i].numpy(),v[i].numpy())
                    attended.append(cache.attention(seq,q[i].numpy()))
                attention=torch.from_numpy(np.stack(attended)).reshape(len(batch),1,-1)
                h=h+att.wo(attention)
                h=h+layer.feed_forward(layer.ffn_norm(h))
            logits=self.model.output(self.model.norm(h))[:,0,:]
            for i,r in enumerate(batch):
                self.last_logits[r.key] = logits[i].clone()
            self.peak_payload_bytes=max(self.peak_payload_bytes,sum(c.stats()['payload_bytes'] for c in self.caches))
            for i,r in enumerate(batch):
                r.cursor+=1
                if self.sharing and r.cursor<len(r.prompt) and r.cursor%self.page_tokens==0:
                    for cache,seq in zip(self.caches,r.sequences):
                        try:
                            cache.publish(seq)
                        except PrefixConflict:
                            # Batched rows may differ by a few FP32 ulps. Retain
                            # the first published prefix and this request's own KV.
                            # Native conflict rejection remains strict and never
                            # overwrites previously cached tensors.
                            pass
                if r.cursor>=len(r.prompt):
                    r.output.append(int(logits[i].argmax()))
                    r.token_latencies_ms.append((time.perf_counter()-started)*1000)
                    if r.first_token is None:
                        r.first_token=time.perf_counter()
                    if len(r.output)==r.max_new_tokens:
                        self.finish(r,'completed')
        except Exception as exc:
            # A failed multi-layer step can leave layers at different positions.
            # Abort the affected microbatch rather than reuse inconsistent KV.
            for r in batch:
                if r.state=='active':
                    self.finish(r,'failed',str(exc))
        return True

    def run(self):
        while self.step():
            pass
        return self.requests

    def close(self):
        for r in self.requests.values():
            if r.state in ('active','queued'):
                self.finish(r,'cancelled')
        for cache in self.caches:
            cache.clear()
            assert cache.stats()['pages']==0
            cache.close()


@torch.inference_mode()
def dense_generate(model,prompt,tokens):
    sequence=prompt.copy()
    result=[]
    for _ in range(tokens):
        token=int(model(torch.tensor([sequence]))[0,0].argmax())
        result.append(token)
        sequence.append(token)
    return result
