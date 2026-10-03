from pathlib import Path
import unittest
import numpy as np
import torch
from native import NativeCache
from serving import Engine,load_model,dense_generate
from dense import DenseCache

ROOT=Path(__file__).resolve().parents[1]


class ServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=load_model(ROOT/'models'/'stories260K.pt')

    def test_real_model_logits_and_greedy_outputs_match_full_context_oracle(self):
        prompt=[1]+[(i*11)%510+2 for i in range(35)]
        for backend in (NativeCache,DenseCache):
            engine=Engine(self.model,cache_class=backend,sharing=False)
            try:
                request=engine.submit('r',prompt,5)
                engine.run()
                self.assertEqual(request.state,'completed',request.error)
                self.assertEqual(request.output,dense_generate(self.model,prompt,5))
                expected=self.model(torch.tensor([prompt+request.output[:-1]]))[0,0].detach()
                torch.testing.assert_close(engine.last_logits['r'],expected,rtol=1e-4,atol=1e-4)
            finally:
                engine.close()

    def test_interleaved_mixed_lengths_prefix_reuse_and_cancellation(self):
        prefix=[1]+list(range(2,49))
        engine=Engine(self.model,max_active=2)
        try:
            seed=engine.submit('seed',prefix+[60,61],2)
            engine.run()
            a=engine.submit('a',prefix+[80,81],4)
            b=engine.submit('b',prefix[:21],3)
            c=engine.submit('c',prefix+[90],4)
            self.assertTrue(engine.cancel('c'))
            self.assertFalse(engine.cancel('c'))
            engine.step()
            # Cancellation after KV allocation must release all layer ownership.
            self.assertTrue(engine.cancel('b'))
            engine.run()
            self.assertGreater(a.reused,0)
            self.assertEqual(a.output,dense_generate(self.model,a.prompt,4))
            self.assertEqual(b.state,'cancelled')
            self.assertEqual(c.output,[])
            self.assertTrue(all(cache.stats()['sequences']==0 for cache in engine.caches))
        finally:
            engine.close()

    def test_memory_admission_and_queue_bounds(self):
        engine=Engine(self.model,pages=2,max_queued=1)
        try:
            engine.submit('one',[1]*20,4)
            with self.assertRaisesRegex(RuntimeError,'admission'):
                engine.submit('two',[1],2)
            with self.assertRaises(ValueError):
                engine.submit('bad',[600],2)
            engine.cancel('one')
            engine.submit('two',[1],2)
            engine.run()
            self.assertEqual(engine.requests['two'].state,'completed')
        finally:
            engine.close()

    def test_native_cow_forks_eviction_and_exhaustion(self):
        cache=NativeCache(1,2,pages=2,page_tokens=2,entries=1,sequences=4)
        try:
            seq,_=cache.checkout()
            cache.append(seq,1,[1.,2.],[3.,4.])
            child=cache.fork(seq)
            before=cache.attention(seq,[1.,1.])
            cache.append(child,2,[8.,9.],[10.,11.])
            np.testing.assert_array_equal(before,cache.attention(seq,[1.,1.]))
            with self.assertRaises(RuntimeError):
                cache.append(child,3,[8.,9.],[10.,11.])
            cache.release(child)
            cache.append(seq,2,[1.,2.],[3.,4.])
            cache.publish(seq)
            cache.release(seq)
            other,_=cache.checkout([9],tenant=2)
            for token in (9,10,11,12):
                cache.append(other,token,[1.,2.],[3.,4.])
            self.assertGreater(cache.stats()['evictions'],0)
            cache.release(other)
            cache.clear()
            self.assertEqual(cache.stats()['pages'],0)
        finally:
            cache.close()

    def test_tenant_isolation(self):
        engine=Engine(self.model)
        try:
            prompt=[1]+list(range(2,36))
            engine.submit('seed',prompt,1,tenant=1)
            engine.run()
            request=engine.submit('other',prompt,1,tenant=2)
            engine.run()
            self.assertEqual(request.reused,0)
        finally:
            engine.close()

    def test_concurrent_cold_duplicate_and_common_prefix_prompts(self):
        prompt=[1]+[(i*11)%510+2 for i in range(35)]
        for prompts in ([prompt,prompt], [prompt,prompt[:-1]+[51],prompt]):
            with self.subTest(requests=len(prompts)):
                engine=Engine(self.model,max_active=len(prompts))
                try:
                    requests=[engine.submit(str(i),p,3) for i,p in enumerate(prompts)]
                    engine.run()
                    self.assertTrue(all(r.state=='completed' for r in requests),[(r.state,r.error) for r in requests])
                    self.assertEqual([r.output for r in requests],[dense_generate(self.model,p,3) for p in prompts])
                    # A future request must still consume the retained prefix correctly.
                    follow=engine.submit('follow',prompt,3)
                    engine.run()
                    self.assertGreater(follow.reused,0)
                    self.assertEqual(follow.output,dense_generate(self.model,prompt,3))
                finally:
                    engine.close()

    def test_partial_layer_failure_aborts_batch_and_releases_ownership(self):
        engine=Engine(self.model,max_active=2)
        try:
            a=engine.submit('a',[1,20,30],2)
            b=engine.submit('b',[1,20,31],2)
            original=engine.caches[2].attention
            def injected_failure(*args):
                raise RuntimeError('injected layer failure')
            engine.caches[2].attention=injected_failure
            engine.step()
            engine.caches[2].attention=original
            self.assertEqual((a.state,b.state),('failed','failed'))
            self.assertEqual(engine.reserved,0)
            self.assertTrue(all(c.stats()['sequences']==0 for c in engine.caches))
            later=engine.submit('later',[1,20,30],2)
            engine.run()
            self.assertEqual(later.state,'completed')
            self.assertEqual(later.output,dense_generate(self.model,later.prompt,2))
        finally:
            engine.close()

    def test_duplicate_prefix_recency_stays_aligned_across_layers(self):
        engine=Engine(self.model,max_active=3)
        a=[1]+[(i*11)%510+2 for i in range(16)]
        b=[99]+[(i*7)%510+2 for i in range(16)]
        try:
            for key,prompt in [('a0',a),('b',b),('a1',a)]:
                engine.submit(key,prompt,1)
            engine.run()
            for j in range(7):
                request=engine.submit(f'fill{j}',[120+j]+[(i*13+j)%380+2 for i in range(16)],1)
                engine.run()
                self.assertEqual(request.state,'completed',request.error)
            follow=engine.submit('follow',a,1)
            engine.run()
            self.assertEqual(follow.state,'completed',follow.error)
            self.assertEqual(follow.output,dense_generate(self.model,a,1))
            self.assertEqual(follow.reused,16)
        finally:
            engine.close()


if __name__=='__main__':
    unittest.main()
