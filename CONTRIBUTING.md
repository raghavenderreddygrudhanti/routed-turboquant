# Contributing

Contributions welcome. Here's what would move the project forward.

## High-Impact Areas

### 1. TQ Subset Scoring (biggest win)
turbovec's Rust API doesn't support scoring a subset of vectors (allowlist). If we could score only the ~2500 unique candidates from routed partitions instead of all vectors in those partitions, latency would drop 5-10x.

Options:
- Add allowlist/mask support to turbovec's `search()` (upstream PR)
- Build a custom SIMD scoring kernel that supports bitmask filtering
- Use turbovec's Python `allowlist=` parameter via pyo3 (v0.4.3+)

### 2. Reduce Per-Partition Overhead
Each `TurboQuantIndex::search()` call has ~0.03ms fixed overhead (rotation matrix, blocked cache lookup). With 12 partitions probed, that's 0.36ms of overhead alone.

Ideas:
- Share rotation matrix and centroids across partition indices
- Pre-compute blocked layout at build time (already done via `prepare()`)
- Batch multiple partition searches into one kernel call

### 3. Scale Benchmarks on Real Embeddings
Current benchmarks use random vectors. Real sentence-transformer embeddings cluster much better, which means:
- Higher partition hit rate at lower R
- Better routing quality
- Potentially competitive latency at 100K+

Need: benchmark with `all-MiniLM-L6-v2` embeddings at 100K-1M scale.

### 4. Adaptive Rerank Budget
Currently rerank=25 is fixed. Could be adaptive:
- If TQ top-1 score is much higher than top-2, rerank fewer (confident)
- If scores are clustered, rerank more (uncertain)

### 5. Build Time Optimization
The O(n × P) multi-assignment loop is slow at 100K+. Options:
- Approximate nearest centroids (random projection, locality-sensitive hashing)
- Parallel assignment with rayon
- Batch centroid scoring with BLAS sgemm

## How to Run

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone (needs turbovec as sibling directory)
git clone https://github.com/RyanCodrai/turbovec.git
git clone https://github.com/raghavenderreddygrudhanti/routed-turboquant.git

# Build and test
cd routed-turboquant
cargo test --release

# Run main benchmark
cargo run --release --example bench_v1_tuned
```

## Code Style

- Terse comments, no fluff
- `cargo clippy` clean
- Tests for any new feature
- Benchmark before/after for performance changes

## Pull Requests

- One feature per PR
- Include benchmark results in PR description
- Tests must pass (`cargo test --release`)
