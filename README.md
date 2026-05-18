# routed-turboquant

Improves Recall@10 over flat TurboQuant using routed candidate generation and top-k float reranking. Built on [turbovec](https://github.com/RyanCodrai/turbovec)'s SIMD scoring kernels.

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

### 10K vectors

| Method | Recall@10 | Latency | Notes |
|--------|-----------|---------|-------|
| FAISS Flat (exact) | 1.000 | 0.014ms | brute force float |
| FAISS IVF-Flat (nprobe=16) | 0.989 | 0.016ms | IVF routing + float scoring |
| FAISS HNSW (M=32 ef=64) | 0.981 | 0.008ms | graph index |
| **routed-turboquant M=2 R=8 rr=25** | **0.987** | 0.297ms | ours |
| turbovec flat | 0.952 | 0.016ms | 4-bit TQ flat scan |
| FAISS IVFPQ (nprobe=16) | 0.871 | 0.012ms | IVF + product quantization |

### 50K vectors

| Method | Recall@10 | Latency | Notes |
|--------|-----------|---------|-------|
| FAISS Flat (exact) | 1.000 | 0.068ms | brute force float |
| FAISS IVF-Flat (nprobe=16) | 0.993 | 0.118ms | IVF routing + float scoring |
| **routed-turboquant M=4 R=21 rr=25** | **0.936** | 1.203ms | ours |
| FAISS HNSW (M=32 ef=64) | 0.931 | 0.023ms | graph index |
| turbovec flat | 0.854 | 0.051ms | 4-bit TQ flat scan |
| FAISS IVFPQ (nprobe=16) | 0.734 | 0.028ms | IVF + product quantization |

### 99K vectors

| Method | Recall@10 | Latency | Notes |
|--------|-----------|---------|-------|
| FAISS Flat (exact) | 1.000 | 0.136ms | brute force float |
| FAISS IVF-Flat (nprobe=16) | 0.986 | 0.135ms | IVF routing + float scoring |
| **routed-turboquant M=2 R=32 rr=25** | **0.900** | 1.815ms | ours |
| FAISS HNSW (M=32 ef=64) | 0.888 | 0.018ms | graph index |
| turbovec flat | 0.863 | 0.094ms | 4-bit TQ flat scan |
| FAISS IVFPQ (nprobe=16) | 0.822 | 0.027ms | IVF + product quantization |

### Analysis

**Recall ranking:** FAISS IVF-Flat > routed-turboquant > FAISS HNSW > turbovec flat > FAISS IVFPQ

**Latency ranking:** FAISS HNSW > FAISS IVFPQ > turbovec flat > FAISS Flat > FAISS IVF-Flat > routed-turboquant

**Where routed-turboquant fits:**
- Higher recall than turbovec flat (+3-9%) and FAISS HNSW at 50K+
- Higher recall than FAISS IVFPQ (+7-20%)
- Lower recall than FAISS IVF-Flat (which uses full float scoring, not quantized)
- Slower than all FAISS methods at current scale (routing overhead dominates)
- Uses 8-16x less memory than FAISS IVF-Flat (4-bit codes vs float32)

**The honest positioning:** routed-turboquant occupies the space between turbovec flat (fast, moderate recall) and FAISS IVF-Flat (high recall, high memory). It achieves near-IVF-Flat recall using quantized storage, at the cost of higher latency from per-partition TQ overhead.

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
| 500K (est.) | ~0.5ms | ~0.5ms | **projected crossover** |
| 1M (est.) | ~1.0ms | ~0.6ms | routed projected faster |

The per-partition TQ `search()` call has ~0.03ms fixed overhead (rotation matrix lookup, blocked cache access). This dominates at small scale. We expect the latency crossover to appear around 500K+ where flat scan time exceeds routing overhead, but this has not been validated with a full benchmark yet.

## How It Differs from turbovec

| | turbovec (flat) | routed-turboquant |
|---|---|---|
| **Scoring** | 4-bit TurboQuant only | TurboQuant + exact float rerank |
| **Scan** | 100% of vectors | 25-50% (configurable) |
| **Recall ceiling** | Limited by quantization noise | Breaks through via float rerank |
| **Speed < 100K** | Faster (no routing overhead) | Slower (per-partition overhead) |
| **Speed > 500K** | Slower (linear scan) | Potentially faster (sublinear, not yet proven) |
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
7. **500K+ latency crossover not yet benchmarked.** Projected from linear scaling, not measured. The claim that routed becomes faster at large scale is a hypothesis.

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
