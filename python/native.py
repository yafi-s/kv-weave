"""ctypes binding for KV-Weave. Tensors remain owned during every native call."""
import ctypes as C
from pathlib import Path
import sys
import numpy as np


class PrefixConflict(RuntimeError):
    """An existing exact-key prefix is retained; publication did not mutate it."""



class NativeCache:
    def __init__(self, heads, dim, *, pages=64, page_tokens=16, entries=8, sequences=32):
        root = Path(__file__).resolve().parents[1]
        filename = 'kv_weave.dll' if sys.platform == 'win32' else 'libkv_weave.dylib' if sys.platform=='darwin' else 'libkv_weave.so'
        self.lib = C.CDLL(str(root/'target'/'release'/filename))
        u, z, p = C.c_uint64, C.c_size_t, C.c_void_p
        specs = {
            'new': ([z,z,z,z,z,z],u), 'drop':([u],C.c_int),
            'checkout':([u,u,p,z,C.POINTER(z)],u), 'append':([u,u,C.c_uint32,p,p,z],C.c_int),
            'attention':([u,u,p,z,p],C.c_int),'release':([u,u],C.c_int),'fork':([u,u],u),
            'publish':([u,u],C.c_int64),'clear':([u],C.c_int),'stats':([u,p,z],C.c_int),
        }
        for name,(args,result) in specs.items():
            fn = getattr(self.lib, 'kw_'+name)
            fn.argtypes, fn.restype = args,result
        self.handle = self.lib.kw_new(pages,page_tokens,heads,dim,sequences,entries)
        if not self.handle:
            raise ValueError('invalid cache configuration')

    @staticmethod
    def check(code):
        if code != 0:
            raise RuntimeError(f'native cache error {code}')

    def checkout(self, prompt=(), tenant=0):
        array = np.ascontiguousarray(prompt,dtype=np.uint32)
        reused = C.c_size_t()
        seq = self.lib.kw_checkout(self.handle,tenant,array.ctypes.data,len(array),C.byref(reused))
        if not seq:
            raise RuntimeError('checkout rejected')
        return seq,reused.value

    def append(self, seq, token, key, value):
        k,v = (np.ascontiguousarray(x,dtype=np.float32).reshape(-1) for x in (key,value))
        if len(k) != len(v):
            raise ValueError('KV width mismatch')
        self.check(self.lib.kw_append(self.handle,seq,token,k.ctypes.data,v.ctypes.data,len(k)))

    def attention(self,seq,query):
        q = np.ascontiguousarray(query,dtype=np.float32).reshape(-1)
        output = np.empty_like(q)
        self.check(self.lib.kw_attention(self.handle,seq,q.ctypes.data,len(q),output.ctypes.data))
        return output

    def release(self,seq):
        self.check(self.lib.kw_release(self.handle,seq))

    def publish(self,seq):
        length = self.lib.kw_publish(self.handle,seq)
        if length == -4:
            raise PrefixConflict('existing prefix retained')
        if length < 0:
            raise RuntimeError('publish rejected')
        return length

    def fork(self,seq):
        child = self.lib.kw_fork(self.handle,seq)
        if not child:
            raise RuntimeError('fork rejected')
        return child

    def stats(self):
        fields = np.zeros(8,dtype=np.uint64)
        self.check(self.lib.kw_stats(self.handle,fields.ctypes.data,8))
        return dict(zip(('payload_bytes','pages','sequences','prefixes','hits','reused','copies','evictions'),map(int,fields)))

    def clear(self):
        self.check(self.lib.kw_clear(self.handle))

    def close(self):
        if self.handle:
            self.check(self.lib.kw_drop(self.handle))
            self.handle = 0
