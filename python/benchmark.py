"""Three-repeat CPU real-model workload matrix with dense and native baselines."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import numpy as np
import psutil
import sentencepiece as spm
import torch

from dense import DenseCache
from native import NativeCache
from serving import Engine,load_model,dense_generate

ROOT=Path(__file__).resolve().parents[1]


def percentile(values,q):
    return float(np.quantile(values,q))


def prompts(tokenizer,context,concurrency,share):
    base=tokenizer.encode('Once upon a time, there was a little girl who loved books. She walked into a forest and found a bright red ball. ')
    common=(base*20)[:int(context*share)//16*16]
    result=[]
    for i in range(concurrency):
        suffix=tokenizer.encode(f'The child named {"Lily" if i%2==0 else "Tim"} saw {i+1} birds in the sky. ')
        result.append(common+((suffix*30)[:context-len(common)]))
    return result,common


def trial(model,mode,prompts,common,expected):
    sharing=mode=='native-prefix'
    engine=Engine(model,max_active=len(prompts),sharing=sharing,
                  cache_class=DenseCache if mode=='torch-dense' else NativeCache)
    process=psutil.Process()
    rss_before=process.memory_info().rss
    rss_peak=rss_before
    started=time.perf_counter()
    warmup_ms=0.
    try:
        if sharing and common:
            seed=engine.submit('seed',common+[1],1)
            engine.run()
            assert seed.state=='completed',seed.error
            warmup_ms=(time.perf_counter()-started)*1000
        measured=time.perf_counter()
        requests=[engine.submit(str(i),p,8) for i,p in enumerate(prompts)]
        while engine.step():
            rss_peak=max(rss_peak,process.memory_info().rss)
        finished=time.perf_counter()
        assert all(r.state=='completed' for r in requests),[(r.state,r.error)for r in requests]
        assert [r.output for r in requests]==expected
        ttft=[(r.first_token-r.submitted)*1000 for r in requests]
        tokens=[v for r in requests for v in r.token_latencies_ms[1:]]
        stats=[c.stats() for c in engine.caches]
        return {'mode':mode,'warmup_ms':warmup_ms,'request_elapsed_s':finished-measured,
                'elapsed_including_warmup_s':finished-started,
                'tokens_per_s_including_warmup':len(requests)*8/(finished-started),
                'tokens_per_s_after_warmup':len(requests)*8/(finished-measured),
                'ttft_p50_ms':percentile(ttft,.5),'ttft_p99_ms':percentile(ttft,.99),
                'decode_step_p50_ms':percentile(tokens,.5),'decode_step_p99_ms':percentile(tokens,.99),
                'peak_retained_kv_payload_bytes':engine.peak_payload_bytes,
                'sampled_process_rss_before_bytes':rss_before,'sampled_process_peak_rss_bytes':rss_peak,
                'reused_tokens_per_request':[r.reused for r in requests],
                'layer_prefix_hits':sum(s['hits'] for s in stats),
                'output_sha256':hashlib.sha256(json.dumps(expected).encode()).hexdigest(),
                'outputs':expected}
    finally:
        engine.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='docs/benchmarks/real-model')
    args=parser.parse_args()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    model=load_model(ROOT/'models'/'stories260K.pt')
    tokenizer=spm.SentencePieceProcessor(model_file=str(ROOT/'models'/'tok512.model'))
    # Warm model kernels once outside all trials; model remains shared and read-only.
    dense_generate(model,[1,20,30],1)
    rows=[]
    for context in (32,96):
        for concurrency in (1,4):
            for share in (0.,.75):
                inputs,common=prompts(tokenizer,context,concurrency,share)
                expected=[dense_generate(model,p,8) for p in inputs]
                for repeat in range(3):
                    # Rotate mode order to reduce order/cache bias.
                    modes=['torch-dense','native-cold','native-prefix']
                    modes=modes[repeat:]+modes[:repeat]
                    for mode in modes:
                        row=trial(model,mode,inputs,common,expected)
                        rows.append({**row,'context':context,'concurrency':concurrency,
                                     'share_fraction':share,'repeat':repeat,'prompt_tokens':inputs})
    (output/'trials.jsonl').write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows),encoding='utf8')
    summary=[]
    for context in (32,96):
        for concurrency in (1,4):
            for share in (0.,.75):
                for mode in ('torch-dense','native-cold','native-prefix'):
                    subset=[r for r in rows if (r['context'],r['concurrency'],r['share_fraction'],r['mode'])==(context,concurrency,share,mode)]
                    summary.append({'context':context,'concurrency':concurrency,'share_fraction':share,'mode':mode,'repetitions':len(subset),
                                    **{key:statistics.median(r[key]for r in subset)for key in ('tokens_per_s_including_warmup','ttft_p50_ms','decode_step_p50_ms','peak_retained_kv_payload_bytes','sampled_process_peak_rss_bytes')}})
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    library_name='kv_weave.dll' if sys.platform=='win32' else 'libkv_weave.dylib' if sys.platform=='darwin' else 'libkv_weave.so'
    manifest={'platform':platform.platform(),'processor':platform.processor(),'python':sys.version,'torch':torch.__version__,
              'threads':torch.get_num_threads(),'model':'karpathy/tinyllamas stories260K; pretrained TinyStories model',
              'model_revision':(ROOT/'models'/'REVISION').read_text().strip(),
              'reference_revision':(ROOT/'python'/'reference'/'REVISION').read_text().strip(),
              'model_sha256':hashlib.sha256((ROOT/'models'/'stories260K.pt').read_bytes()).hexdigest(),
              'command':'python python/benchmark.py','trials':len(rows),'generated_tokens_per_request':8,
              'source_hash_format':'sha256-lf-normalized-text',
              'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in list((ROOT/'python').glob('*.py'))+list((ROOT/'src').glob('*.rs'))+[ROOT/'python'/'reference'/name for name in ('model.py','LICENSE','REVISION','ATTRIBUTION.md','UPSTREAM_VERIFICATION.json')]},
              'binary_sha256':hashlib.sha256((ROOT/'target'/'release'/library_name).read_bytes()).hexdigest(),
              'artifact_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in (output/'trials.jsonl',output/'summary.json')},
              'limitations':['CPU only; tiny pretrained model, no quality/adoption measurement.',
                             'Microbatch projections, one prefill/decode token per active request per step.',
                             'RSS sampled at step boundaries; includes interpreter/model/allocator and previous trials.',
                             'Payload bytes are retained KV arrays at step boundaries, excluding transient dense copies; not process-memory savings.',
                             'Dense baseline uses torch.cat growth and expanded heads; not an optimized dense serving system or paging-only comparison.',
                             'Warm prefix setup is included in total throughput; TTFT begins at request submission after setup.',
                             'Tiny batch latency percentiles have few samples; repetitions are not production SLO evidence.']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps({'trials':len(rows),'greedy_output_parity':'all trials','model':'stories260K','cpu_threads':2}))


if __name__=='__main__':
    main()
