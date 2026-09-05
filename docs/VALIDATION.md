# Validation record

Implementation commit: `b862cae3824f09a974956acee4085d3b06b9a2db`.

[GitHub CI run](https://github.com/yafi-s/kv-weave/actions/runs/33942589589)
completed successfully on 2026-09-05 UTC. The matrix covered Ubuntu and macOS with
Rust 1.85.0 and stable. Every job checked formatting, ran the locked test suite,
ran Clippy with warnings denied, and executed the release-mode experiment.

Local checks also passed on Apple M2 with Rust 1.98.1. Eight integration tests
cover dense attention equivalence across page boundaries and grouped-query head
ratios, partial-page copy on write, namespace separation, longest-prefix reuse,
LRU pressure, failed admission, conflicting tensors, invalid inputs, and extreme
finite logits. One test executes 10,000 deterministic lifecycle operations with
ownership and dense-output checks for every live sequence.

The shared-prefix experiment asserts equality of complete attention output
vectors between cold and shared storage, then verifies complete page reclamation.
Its memory and timing results describe a synthetic single-layer CPU workload;
they do not establish performance of a complete language-model serving system.
