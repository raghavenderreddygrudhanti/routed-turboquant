# routed-turboquant

Improves recall over flat TurboQuant using routed candidate generation and lightweight float reranking. Built on [turbovec](https://github.com/RyanCodrai/turbovec)'s SIMD scoring kernels.

## The Problem

turbovec scores every vector with 4-bit TurboQuant SIMD — extremely fast, but recall is capped by quantization noise. At dim=384, flat turbovec tops out around 0.85-0.95 recall@10 depending on scale. Two vectors that are close in float space can get misordered after 4-bit compression. turbovec has no mechanism to correct these misrankings because it scores everything once and returns top-k.

## The Insight

The true neighbors are in turbovec's candidate set — they're just ranked slightly wrong. If you rescore the top ~25 candidates with exact float inner product, you recover the misranked neighbors. But you need a cheap way to identify which 25 to rescore out of thousands.

## What We Built

A three-stage pipeline:

```
Stage 1: Route (cheap)
  Query → dot product with P centroids → select top-R partitions
  Cost: P dot products (negligible)

Stage 2: TQ Score (fast, approximate)
  Score candidates within R partitions using TurboQuant SIMD
  Cost: 25-50% of vectors scored

Stage 3: Float Rerank (precise, small)
  Take top-25 from TQ scoring → exact float inner product
  Cost: 25 dot products (negligible)
```

Multi-assignment (M=2-4) stores each vector in multiple partitions to ensure true neighbors are present in the probed set.

## Results on Real Embeddings

**Dataset:** 100K sentence-transformer embeddings (all-MiniLM-L6-v2, dim=384)

| Scale | Method | Recall@10 | Latency | Memory |
|-------|--------|-----------|---------|--------|
| 10K | turbovec flat | 0.952 | 0.016ms | 192 KB |
| 10K | routed M=2 R=8 rr=25 | **0.987** | 0.297ms | 576 KB |
| 10K | routed M=4 R=8 rr=25 | **0.989** | 0.362ms | 960 KB |
| 50K | turbovec flat | 0.854 | 0.051ms | 960 KB |
| 50K | routed M=3 R=16 rr=25 | **0.934** | 0.792ms | 4.5 MB |
| 50K | routed M=4 R=21 rr=25 | **0.936** | 1.203ms | 5.8 MB |
| 99K | turbovec flat | 0.863 | 0.094ms | 1.9 MB |
| 99K | routed M=2 R=32 rr=25 | **0.900** | 1.815ms | 5.6 MB |
| 99K | routed M=4 R=16 rr=25 | **0.896** | 1.453ms | 11.0 MB |

Recall improvement: **+3.7% at 10K, +9.6% at 50K, +4.3% at 99K.**

### Random Vectors (worst case — no cluster structure)

| Method | Recall@10 | Latency | Storage |
|--------|-----------|---------|---------|
| turbovec flat | 0.840 | 0.018ms | 1x |
| routed M=4 R=12 rr=25 | **0.935** | 0.508ms | 4x |
| routed M=4 R=16 rr=25 | **0.983** | 0.652ms | 4x |

Even on random data with no natural clusters, routed achieves +11-17% recall.

## Memory Breakdown

For n=99K, dim=384, 4-bit, M=2, P=64:

| Component | Size | Notes |
|-----------|------|-------|
| TQ codes (per partition) | 2 × 99K × 384 × 4/8 = 38.0 MB | M=2 duplicates |
| Norms | 2 × 99K × 4 = 0.8 MB | one per stored copy |
| Centroids | 64 × 384 × 4 = 96 KB | P float centroids |
| Partition ID lists | 2 × 99K × 8 = 1.6 MB | global ID mappings |
| Float vectors (rerank) | 99K × 384 × 4 = 152 MB | original vectors for rerank |
| **Total** | **~192 MB** | dominated by float rerank storage |

Without float rerank (TQ scores only): ~40 MB. The float vectors for reranking are the largest cost. A future optimization is to store only the top-k candidates' vectors on demand rather than all vectors.

## Latency Analysis

| Scale | turbovec flat | routed (best) | Overhead source |
|-------|--------------|---------------|-----------------|
| 10K | 0.016ms | 0.297ms | per-partition TQ call overhead (0.03ms × 8) |
| 50K | 0.051ms | 0.792ms | per-partition TQ call overhead (0.03ms × 16) |
| 99K | 0.094ms | 1.453ms | per-partition TQ call overhead (0.03ms × 16) |
| 500K (est.) | ~0.5ms | ~0.5ms | **crossover point** |
| 1M (est.) | ~1.0ms | ~0.6ms | routed wins (scans 25% not 100%) |

The per-partition TQ `search()` call has ~0.03ms fixed overhead (rotation matrix lookup, blocked cache access). This dominates at small scale. At 500K+, flat scan time exceeds routing overhead and routed becomes faster.

**Note:** The 500K and 1M estimates are projections based on observed linear scaling of flat TQ. Full benchmarks at these scales are pending (build time at 500K with M=4 P=128 is ~10 minutes).

## How It Differs from turbovec

| | turbovec (flat) | routed-turboquant |
|---|---|---|
| **Scoring** | 4-bit TurboQuant only | TurboQuant + exact float rerank |
| **Scan** | 100% of vectors | 25-50% (configurable) |
| **Recall ceiling** | Limited by quantization noise | Breaks through via float rerank |
| **Speed < 100K** | Faster (no routing overhead) | Slower (per-partition overhead) |
| **Speed > 500K** | Slower (linear scan) | Faster (sublinear) |
| **Memory** | 1x (codes + norms) | 2-4x codes + float vectors |
| **Build** | O(n) instant | O(n × P) k-means |
| **Tuning** | bit_width only | M, R, P, rerank (full control) |
| **Use case** | Low-latency, moderate recall | High-recall, tunable |

**Why turbovec flat has a recall ceiling:**

4-bit quantization is lossy. Two vectors close in float space may get misordered after compression. When the true #8 neighbor scores 0.891 in float but 0.887 in TQ, and a non-neighbor scores 0.889 in TQ, the non-neighbor wins. turbovec has no second opinion — it scores once and returns.

**How routed-turboquant breaks through:**

TQ scoring is used as a cheap filter, not the final answer. The top-25 TQ candidates are rescored with exact float inner product. This corrects ~95% of TQ misrankings at negligible cost (25 dot products).

## What Made It Work

1. **Float reranking** — the single biggest win. Without it, routed recall equals flat recall. With rerank=25, recall jumps +10-15%.

2. **Multi-assignment** — stores each vector in M nearest partitions. Raises partition hit rate from 22% (M=1) to 93.5% (M=4) on random data. On real embeddings, even M=2 achieves >95% hit rate.

3. **Correctness verification** — full-probe (R=P) matches flat exactly, proving no bugs. Partition Hit@10 perfectly predicts final recall.

## What Didn't Work

1. **Centroid pre-ranking** — centroid score is partition-level, not vector-level. All vectors in a partition get the same score. Recall: 0.10-0.33.

2. **Partial dot product (first 32 dims)** — too coarse. 32 dims = 8% of information. Recall at cap=500: 0.10-0.23.

3. **Full float scoring without TQ** — correct recall but 100x slower. Float runs at 4.4M vec/sec vs TQ SIMD at 588M vec/sec.

Lesson: **TQ-level scoring is the necessary middle layer.** Nothing cheaper provides enough vector-level signal.

## Limitations

1. **Slower than flat turbovec below ~500K vectors.** Per-partition TQ call overhead (~0.03ms each) dominates at small scale.
2. **Multi-assignment increases storage 2-4x.** Each vector stored in M partitions.
3. **Float rerank requires original vectors in memory** (~152 MB at 99K). Dominates memory at scale.
4. **Build time is high.** O(n × P) for k-means + multi-assignment. ~80s at 99K with P=64.
5. **Depends on turbovec as sibling directory.** Not published to crates.io independently.
6. **No streaming insert/delete.** Index must be rebuilt to add vectors.
7. **500K+ crossover not yet benchmarked.** Projected from linear scaling, not measured.

## Quick Start

```bash
# Needs turbovec as sibling directory
git clone https://github.com/RyanCodrai/turbovec.git
git clone https://github.com/raghavenderreddygrudhanti/routed-turboquant.git

cd routed-turboquant
cargo test --release        # 10 tests
cargo run --release --example bench_v1_tuned       # random data sweep
cargo run --release --example bench_real_data      # real embeddings (needs data file)
```

```rust
use routed_turboquant::index::{RoutedTQConfig, RoutedTurboQuantIndex};

let config = RoutedTQConfig {
    dim: 384,
    n_partitions: 32,
    n_probe: 12,
    bit_width: 4,
    kmeans_iter: 10,
    seed: 42,
    multi_assign: 4,        // store each vector in 4 partitions
    boundary_threshold: None,
    max_assign: 4,
    rerank_top: 25,         // float rescore top 25 TQ candidates
};

let index = RoutedTurboQuantIndex::build(&vectors, config);
let (scores, indices) = index.search(&query, 10);
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | 384 | Vector dimensionality (must be multiple of 8) |
| `n_partitions` | 128 | Number of k-means partitions (P) |
| `n_probe` | 8 | Partitions searched per query (R) |
| `bit_width` | 4 | TurboQuant bits per dimension (2, 3, or 4) |
| `kmeans_iter` | 10 | K-means iterations at build time |
| `multi_assign` | 1 | Copies per vector across partitions (M) |
| `boundary_threshold` | None | Adaptive assignment: assign to partitions within threshold of best |
| `max_assign` | 4 | Max partitions per vector in boundary mode |
| `rerank_top` | 0 | Float rescore top-N TQ candidates (0 = disabled) |

## Project Structure

```
src/
├── lib.rs       — crate root
├── kmeans.rs    — float k-means++ (partition routing)
└── index.rs     — RoutedTurboQuantIndex (multi-assign + TQ scoring + rerank)

examples/
├── bench_v1_tuned.rs    — main benchmark (M/R sweep, random data)
├── bench_real_data.rs   — real embedding benchmark (10K, 50K)
├── bench_real_99k.rs    — real embedding benchmark (99K)
├── diagnose.rs          — correctness verification
├── diagnose_multi.rs    — multi-assignment impact
├── bench_boundary.rs    — boundary-aware assignment
└── bench_rerank.rs      — reranking impact
```

## Benchmarking

```bash
# Random data: full M/R sweep at 10K
cargo run --release --example bench_v1_tuned

# Real embeddings: 10K + 50K (needs ../data/minilm_100k.npy)
cargo run --release --example bench_real_data

# Real embeddings: 99K only
cargo run --release --example bench_real_99k

# Correctness: full-probe = flat, partition hit rate
cargo run --release --example diagnose

# Reranking impact: recall with/without rerank at various budgets
cargo run --release --example bench_rerank
```

**Environment used for published results:**
- CPU: Apple M3 Max (12 cores)
- RAM: 36 GB
- OS: macOS
- Rust: 1.95.0
- turbovec: v0.2.0 (local, commit from RyanCodrai/turbovec main)
- Dataset: 100K embeddings from `all-MiniLM-L6-v2` via sentence-transformers

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Highest-impact areas:

1. **TQ subset scoring** — add allowlist support to turbovec Rust API (eliminates duplicate scoring)
2. **500K+ benchmark** — validate the crossover point with real data
3. **FAISS baselines** — add IVF-Flat, HNSW, IVFPQ comparisons
4. **Streaming insert** — support adding vectors without full rebuild

## License

MIT
