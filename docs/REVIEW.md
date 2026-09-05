# Review and extension guide

Start at `Cache::append`: trace the shared-partial-page case and explain why the
old page cannot be modified before allocation succeeds. Then trace eviction when
all cached pages are also held by active requests. A cache eviction is not the
same thing as a physical page reclamation.

The tests compare attention against an independent two-pass dense softmax oracle
across page boundaries and grouped-query head ratios. A deterministic 10,000-step
model alternates checkout, append, fork, publish, release, and eviction, validates
ownership invariants, and compares live attention outputs. Focused cases cover
exhaustion during copy on write, namespace separation, conflicting tensors,
invalid shapes, extreme finite logits, and complete reclamation.

Useful experiments before claiming broader applicability:

1. Sweep prefix lengths, divergence points, page sizes, and cache pressure. Plot
   memory waste from partial pages alongside copy counts and eviction rates.
2. Add multi-layer admission that reserves every layer before changing any active
   sequence. Define rollback behavior under partial allocation failure.
3. Replace the bounded prefix scan with a radix index. Compare metadata, lookup
   latency, and exact-key correctness under adversarial shared prefixes.
4. Integrate a real model projection path and compare full logits against a trusted
   dense implementation before measuring token latency or throughput.
5. Build and verify a GPU kernel with identical page-table semantics. Treat kernel
   correctness, host/device ownership, and transfer accounting as separate tasks.

Be prepared to explain why tenant IDs do not implement authentication, why cached
prefixes need their own references, why online softmax must rescale both numerator
and denominator, and why payload savings are not an end-to-end speedup claim.
