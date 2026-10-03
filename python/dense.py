"""PyTorch growing dense KV baseline; same projections and serving scheduler."""
import math
import numpy as np
import torch


class DenseCache:
    def __init__(self, heads,dim,**kwargs):
        self.heads,self.dim = heads,dim
        self.sequences={}
        self.next=1

    def checkout(self,prompt=(),tenant=0):
        seq=self.next
        self.next+=1
        self.sequences[seq]=None
        return seq,0

    def append(self,seq,token,key,value):
        k,v=(torch.from_numpy(np.asarray(x).reshape(1,self.heads,self.dim)) for x in (key,value))
        old=self.sequences[seq]
        self.sequences[seq]=(k.clone(),v.clone()) if old is None else (torch.cat((old[0],k)),torch.cat((old[1],v)))

    def attention(self,seq,query):
        k,v=self.sequences[seq]
        q=torch.from_numpy(query.reshape(-1,self.dim))
        rep=len(q)//self.heads
        k=k.repeat_interleave(rep,dim=1).transpose(0,1)
        v=v.repeat_interleave(rep,dim=1).transpose(0,1)
        scores=(q[:,None,:]*k).sum(dim=-1)/math.sqrt(self.dim)
        out=(scores.softmax(-1)[:,:,None]*v).sum(dim=1)
        return out.numpy().reshape(-1)

    def release(self,seq):
        del self.sequences[seq]

    def stats(self):
        size=sum(sum(x.numel()*x.element_size() for x in pair) for pair in self.sequences.values() if pair)
        return {'payload_bytes':size,'pages':0,'sequences':len(self.sequences),'prefixes':0,'hits':0,'reused':0,'copies':0,'evictions':0}

    def clear(self):
        pass

    def close(self):
        self.sequences.clear()
