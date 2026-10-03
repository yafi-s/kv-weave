# Real-model CPU prefill and decode

The new `python/` path runs Karpathy's trained stories260K transformer, using its
five-layer Llama architecture, rotary embeddings, eight query heads and four KV
heads. Embeddings, projection and MLP weights run in PyTorch on CPU. KV-Weave's
Rust library owns each layer's KV pages and computes grouped-query attention.
The reference model implementation and checkpoint are third-party MIT work;
the cache binding, scheduler, reliability checks and experiments are this upgrade.
There is no training job, downloaded corpus, GPU, hosted API, or account setup.

```sh
cargo build --locked --release
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.26.4 psutil==6.1.1 sentencepiece==0.2.0
python python/download_model.py
python -m unittest discover -s python -p test_serving.py -v
python python/demo.py
python python/benchmark.py
```

The checkpoint/tokenizer download is explicit, pinned to a revision and SHA-256,
and bounded to 2 MB per file. Subsequent tests, demo and benchmark are offline.
The vendored reference source is pinned in `python/reference/REVISION`, with its
original license. `UPSTREAM_VERIFICATION.json` records a byte comparison against
the immutable upstream source/license and the checkpoint's MIT declaration.
Native library names support Windows, Linux, and macOS; the recorded benchmark
used Windows/MSVC and Rust 1.85.1. CI also executes the CPU serving tests and demo.

```mermaid
flowchart LR
  Requests[Bounded request admission] --> Queue[Microbatch scheduler]
  Queue --> Proj[PyTorch embeddings and QKV projections]
  Proj --> Cache[Rust per-layer paged KV ownership]
  Cache --> GQA[Rust grouped-query attention]
  GQA --> MLP[PyTorch output projections and MLP]
  MLP --> Tokens[Greedy token and request lifecycle]
  Tokens --> Queue
```

Each step processes one token per active request and batches projections and
MLPs. This provides interleaved prefill/decode and permits cancellation between
steps. It does not implement a GPU fused batch-attention kernel. Queued/active
requests, each request's total tokens, active sequences, physical pages, and
request-history metadata are bounded. Admission conservatively reserves full
request pages even when reuse might save space. Cancellation releases all layer
sequence ownership. If a native/multi-layer step fails, its microbatch is aborted
so partially advanced layers cannot silently produce later tokens. The C ABI
trusts in-process tensor pointers; it is not a safe boundary for arbitrary native
callers. Integer cache handles reject stale/double releases.

Prefix sharing preserves tenant isolation and copies a partial page on a fork.
Only complete pages are published. The last prompt token is always computed to
recover output logits; the cache does not store logits. Eight serving tests cover
real-model logits and greedy parity, mixed-length interleaving, warmed prefix
reuse, cancellation before/after allocation, quotas, namespace isolation,
copy-on-write, eviction, exhaustion, concurrent cold duplicate prefixes, and
partial-layer failure cleanup. Floating-point differences in concurrent duplicate
prefixes preserve the first cached tensors and skip only conflicting optional
publication; strict core conflict rejection remains intact. Conflicts refresh
recency consistently so every layer evicts the same prefix. The independent full-context reference
has no native cache. Logit tolerance is `rtol=1e-4, atol=1e-4`; greedy outputs must
match exactly. The eight original Rust tests and their ownership/attention oracle
remain intact, plus a new FFI-handle/buffer validation test.

## Measured matrix

`docs/benchmarks/real-model` contains 72 raw trials: contexts 32/96, concurrency
1/4, requested shared fractions 0/0.75, three storage modes, and three repetitions.
Shared prefix lengths round down to complete pages: 16 of 32 or 64 of 96 tokens,
so actual reused fractions are 50%/66.7% when a warm prefix is present. Actual
prompts, reused lengths and output hashes are in every raw trial. Mode order
rotates between repetitions. Prefix warmup is included in total throughput;
TTFT begins when the workload is submitted after that setup. The PyTorch dense
baseline uses `torch.cat` to grow dense KV arrays and expands grouped heads,
with the same projection/MLP/scheduler. It is not an optimized dense serving
system, and timing differences do not isolate paging alone;
the full-context reference is used for correctness, not the performance baseline.

For four 96-token requests with a 64-token reusable prefix, median throughput
including warmup was 38.82 generated tokens/s for PyTorch dense KV, 69.19 for
native cold pages, and 89.87 with native prefix reuse. All trial greedy outputs
matched. Peak retained KV payload was 527,360, 573,440, and 327,680 bytes,
respectively: 37.9% less than dense or 42.9% less than native cold pages. Retained
payload excludes transient dense copies and attention buffers. Page
rounding makes native cold storage larger than dense storage. Sampled process RSS
was about 199 MB in each mode; this is not a 38% process-memory saving.

For a single short request, prefix setup did not materially improve throughput:
native cold/prefix medians were 67.46/67.77 tokens/s, and pages used more KV
payload than dense arrays. Regressions and all configurations remain in the raw
results. RSS is sampled at step boundaries and includes the interpreter, model,
allocator, and prior trials; it is not a precise native allocation peak or an
independent-process comparison. Eight output tokens/request and these tiny
concurrency levels do not establish production SLOs. This is a real trained
tiny-model CPU execution experiment, not large-model/GPU-serving performance,
model quality, adoption, or production capacity evidence.
