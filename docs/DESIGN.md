# Ownership and attention design

## Page ownership

A page stores `page_tokens × 2 × kv_heads × head_dim` f32 elements. Within each
token, keys precede values. Sequences carry ordered page IDs, exact token IDs,
and a namespace. A cached prefix carries its own ordered page IDs and an LRU
timestamp. Each sequence and each prefix contributes one reference to every page
it owns. A page is reusable exactly when that count reaches zero.

A fork copies sequence metadata and adds references, leaving tensor payload shared.
Appending into an unshared partial page writes in place. Appending into a shared
partial page first allocates another page, copies the initialized region, changes
only the writer's page-table entry, and drops its old reference. Appending beyond
a full page allocates a fresh page. Published prefixes end on full-page boundaries,
so later appends cannot modify their visible tensors.

On page exhaustion, allocation evicts cached prefixes in LRU order. An evicted
prefix may release no physical pages when active sequences still reference them;
eviction continues until a page becomes free or no prefixes remain. The latter
returns `OutOfPages`. Active sequence contents remain unchanged after failed
admission, although attempted allocation may have evicted idle prefix entries.
Ordinary allocation failure inside Rust containers is outside that guarantee.

Sequence IDs increase monotonically and are never recycled. Releasing an ID makes
subsequent use an error. Ownership counters are audited by reconstructing expected
references from all live page tables and cache entries, and comparing the free
list and payload lengths against those expectations.

## Prefix identity

A key is the complete `(model, revision, tenant, token vector)` tuple. HashMap
lookup resolves collisions by equality. Checkout scans bounded cache entries to
find the longest exact prefix of the requested prompt; it does not allocate a
candidate key for every possible prompt length. Publication checks existing KV
values before accepting an identical key. Namespaces must be nonempty and have
at most 256 bytes per field. The application must use revisions that capture all
state affecting KV tensors. Namespace strings alone do not enforce access control.

## Attention

For query head `h`, the KV head is `h / (query_heads / kv_heads)`. Query head count
must be a positive multiple of KV head count. For every stored token, compute
`z = dot(q, k) / sqrt(head_dim)`. Given running maximum `m`, denominator `d`, and
unnormalized output vector `a`, update with `m2 = max(m, z)`:

```
d2 = exp(m - m2) * d + exp(z - m2)
a2 = exp(m - m2) * a + exp(z - m2) * v
```

The initial state uses the first token to avoid undefined empty sums; empty
sequences are rejected. Final output is `a / d`. Dot products and accumulators use
f64; stored tensors and returned outputs use f32. Shape and finite-value checks
reject invalid input. This attends over the stored causal history for one query
position. The caller supplies any positional transformation before insertion.

Time is O(tokens × query_heads × head_dim), with O(query_heads × head_dim) output
and accumulation work space. No dense score matrix is materialized. Reference
counting saves storage and prefix insertion work, not attention arithmetic.

## Bounds

Configuration limits pages to 4,096, tokens per page to 1,024, KV heads to 128,
head dimension to 512, active sequences to 128, and cached prefixes to 128.
Additional combined limits cap physical payload at 64 Mi f32 elements and the
worst-case token-metadata estimate at 16 Mi entries. Query heads are capped at 128.
These are admission bounds for this reference library, not a process RSS limit.
Config validation uses checked multiplication for the payload product.

Prefix lookup is O(entries × prefix length), fork is O(sequence metadata), and
copy on write copies at most one page. Mutation needs `&mut Cache`; read-only
attention needs `&Cache`. There is no internal scheduling or synchronization.
