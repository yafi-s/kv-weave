# Shared-prefix experiment

Raw output: [local.jsonl](benchmarks/local.jsonl).

Recorded on 2026-09-04 America/Chicago (2026-09-05 UTC), Apple M2, macOS 26.5.2,
Rust 1.98.1, release profile. One uncontrolled desktop run; no CPU pinning,
statistical repetitions, or cross-machine comparisons. Timings are illustrative.

```sh
cargo run --release --example experiment
```

Both modes create 32 sequences, each with 64 shared-prefix tokens and 8 tail
tokens. Each page has 16 token slots, 2 KV heads, dimension 32, and f32 K/V tensors.
The cache has 192 pages and 8 prefix entries. The shared case first creates,
publishes, and releases a seed prefix; that work is included in preparation time
and appended-token count. Inputs are deterministic synthetic tensors, not model
projections. Each sequence performs one single-layer CPU attention call.

| Mode | Appended tokens | Live pages | KV payload bytes | Prepare µs | Attention µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold | 2,304 | 160 | 1,310,720 | 618 | 192 |
| Prefix sharing | 320 | 36 | 294,912 | 100 | 178 |

The shared mode reuses 2,048 tokens. Full attention output vectors are asserted
exactly equal across modes; both checksums are -10.545705479569733. Pages fall from
160 to 36: four shared prefix pages plus one partial tail page per request.
Payload savings are **77.5%**. Both modes release all sequences and cached entries
and assert complete reclamation after the measurement.

Preparation includes synthetic tensor generation, validation, metadata, and page
insertion. Attention time includes output allocation. Payload counts reserve full
live-page capacity, including unused tail slots, but exclude metadata, the free
page descriptors, allocator overhead, and process RSS. Prefix reuse does not
remove attention reads or arithmetic. The small attention timing difference is
not evidence of a reliable speedup. There is no real model, GPU, multilayer
prefill, tokenizer, request queue, or end-to-end serving throughput in this test.
