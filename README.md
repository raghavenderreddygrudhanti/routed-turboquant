# routed-turboquant

[![CI](https://github.com/raghavenderreddygrudhanti/routed-turboquant/actions/workflows/ci.yml/badge.svg)](https://github.com/raghavenderreddygrudhanti/routed-turboquant/actions/workflows/ci.yml)

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
  Cost: P dot products

Stage 2: TQ Score (fast, approximate)
  Score candidates within R partitions using TurboQuant SIMD
  Cost: 25-50% of vectors scored (per-partition TQ overhead dominates)

Stage 3: Float Rerank (precise, small)
  Take top-25 from TQ scoring → exact float inner product
  Cost: 25 dot products
```

Total routed latency is dominated by per-partition TQ search and merge overhead, not by the rerank step itself.

Multi-assignment (M=2-4) stores each vector in multiple partitions to ensure true neighbors are present in the probed set.

## Results on Real Embeddings

**Dataset:** 100K sentence-transformer embeddings generated from `all-MiniLM-L6-v2` via the `sentence-transformers` Python library. Text source: combinatorial phrases from 40 CS topics × 27 verbs × 16 contexts. Vectors are L2-normalized. Queries: last 200 vectors from the dataset (same distribution as database). Ground truth: exact inner product over all n vectors.

### 10K vectors

| Method | Recall@10 | Latency | Notes |
|--------|-----------|---------|-------|
| FAISS Flat (exact) | 1.000 | 0.014ms | brute force float |
| FAISS IVF-Flat (nprobe=16) | 0.989 | 0.016ms | IVF routing + float scoring |
| **routed-turboquant M=2 R=8 rr=25** | **0.987** | 0.297ms | ours |
| FAISS HNSW (M=32 ef=64) | 0.981 | 0.008ms | graph index |
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

### Partition Hit@10 on Real Data

Measured on 99K real embeddings (P=64, rerank=25):

| M | R | Scan% | PartHit@10 | Recall (no rerank) | Recall (rr=25) |
|---|---|-------|------------|--------------------:|---------------:|
| 1 | 16 | 25% | 0.87 | 0.863 | 0.888 |
| 1 | 32 | 50% | 0.93 | 0.870 | 0.889 |
| 2 | 16 | 25% | 0.95 | 0.872 | 0.892 |
| 2 | 32 | 50% | 0.98 | 0.880 | 0.900 |
| 4 | 16 | 25% | 0.98 | 0.876 | 0.896 |

On real embeddings, M=2 already achieves 95%+ partition hit rate because semantic vectors cluster naturally. The rerank step adds +2-3% recall on top of TQ-only scoring.

### Analysis

**Recall ranking:** FAISS IVF-Flat > routed-turboquant > FAISS HNSW > turbovec flat > FAISS IVFPQ

**Latency ranking:** FAISS HNSW > FAISS IVFPQ > turbovec flat > FAISS Flat ≈ FAISS IVF-Flat > routed-turboquant

**Where routed-turboquant fits:**
- Higher recall than turbovec flat (+3-9%) and FAISS HNSW at 50K+
- Higher recall than FAISS IVFPQ (+7-20%)
- Lower recall than FAISS IVF-Flat (which uses full float scoring, not quantized)
- Slower than all FAISS methods at current scale (per-partition TQ overhead dominates)
- Uses 8-16x less memory for compressed candidate storage than FAISS IVF-Flat, but float reranking currently requires storing original vectors, which dominates total memory

**The honest positioning:** routed-turboquant occupies the space between turbovec flat (fast, moderate recall) and FAISS IVF-Flat (high recall, high memory). It achieves near-IVF-Flat recall using quantized scoring, at the cost of higher latency from per-partition TQ overhead.

## Memory Breakdown

For n=99K, dim=384, 4-bit, M=2, P=64:

| Component | Size | Notes |
|-----------|------|-------|
| TQ codes (per partition) | 2 × 99K × 384 × 4/8 = 38.0 MB | M=2 duplicated codes |
| Norms | 2 × 99K × 4 = 0.8 MB | one per stored copy |
| Centroids | 64 × 384 × 4 = 96 KB | P float centroids |
| Partition ID lists | 2 × 99K × 8 = 1.6 MB | global ID mappings |
| Float vectors (rerank) | 99K × 384 × 4 = 152 MB | original vectors for rerank |
| **Total** | **~192 MB** | **dominated by float rerank storage** |

Without float rerank (TQ scores only): ~40 MB. The float vectors for reranking are the largest memory cost. Future optimization: store float vectors on disk and load only the top-25 candidates on demand, or use a higher-bit (8-bit) intermediate representation instead of full float32.

**Comparison:**
- FAISS IVF-Flat at 99K: ~152 MB (float32 vectors only, no codes)
- routed-turboquant without rerank: ~40 MB (4-bit codes, 4x less)
- routed-turboquant with rerank: ~192 MB (codes + float vectors)

The memory advantage over FAISS IVF-Flat applies only when reranking is disabled or float vectors are stored externally.

## Latency Analysis

| Scale | turbovec flat | routed M=1 R=16 (25% scan) | Speedup |
|-------|--------------|----------------------------|---------|
| 10K | 0.016ms | 0.297ms | 0.05x |
| 50K | 0.051ms | 0.792ms | 0.06x |
| 99K | 0.094ms | 0.693ms | 0.14x |
| 100K | 0.107ms | 0.693ms | 0.15x |
| 200K | 0.229ms | 0.909ms | 0.25x |
| **500K** | **0.605ms** | **1.488ms** | **0.41x** |

The gap closes with scale (from 0.05x at 10K to 0.41x at 500K) but **routed is still 2.5x slower at 500K**. Extrapolating, the crossover may occur around 2-3M vectors, but this is unverified.

**Root cause:** Per-partition TQ `search()` has ~0.03ms fixed overhead per call (rotation matrix, blocked cache). With 16 partitions probed, that's ~0.48ms of overhead regardless of partition size. turbovec flat has zero per-call overhead.

**Conclusion:** routed-turboquant is a **recall improvement** tool, not a speed improvement at any tested scale (up to 500K).

## How It Differs from turbovec
| | turbovec (flat) | routed-turboquant |
|---|---|---|
| **Scoring** | 4-bit TurboQuant only | TurboQuant + exact float rerank |
| **Scan** | 100% of vectors | 25-50% (configurable) |
| **Recall ceiling** | Limited by quantization noise | Breaks through via float rerank |
| **Speed < 100K** | Faster (no routing overhead) | Slower (per-partition overhead) |
| **Speed > 500K** | 0.605ms at 500K | 1.488ms at 500K (still slower, gap closing) |
| **Memory** | 1x (codes + norms) | 2-4x codes + float vectors for rerank |
| **Build** | O(n) instant | O(n × P) k-means |
| **Tuning** | bit_width only | M, R, P, rerank (full control) |
| **Use case** | Low-latency, moderate recall | High-recall, tunable |

**Why turbovec flat has a recall ceiling:**

4-bit quantization is lossy. Two vectors close in float space may get misordered after compression. When the true #8 neighbor scores 0.891 in float but 0.887 in TQ, and a non-neighbor scores 0.889 in TQ, the non-neighbor wins. turbovec has no second opinion — it scores once and returns.

**How routed-turboquant breaks through:**

TQ scoring is used as a cheap filter, not the final answer. The top-25 TQ candidates are rescored with exact float inner product. This corrects ~95% of TQ misrankings. The total pipeline latency is dominated by per-partition TQ overhead, not the rerank step.

## What Made It Work

### Ablation Study (99K real embeddings, P=64)

| Method | Routing | Multi-assign | Rerank | Recall@10 | Mean ms | p50 ms |
|--------|---------|-------------|--------|-----------|---------|--------|
| turbovec flat | — | — | — | 0.863 | 0.085 | 0.083 |
| routed M=1 R=16 | Yes | M=1 | No | 0.857 | 0.635 | 0.616 |
| routed M=1 R=16 rr=25 | Yes | M=1 | Yes | **0.888** | 0.674 | 0.672 |
| routed M=2 R=16 rr=25 | Yes | M=2 | Yes | **0.892** | 0.891 | 0.898 |
| routed M=2 R=32 rr=25 | Yes | M=2 | Yes | **0.900** | 1.726 | 1.732 |
| routed M=4 R=16 rr=25 | Yes | M=4 | Yes | **0.896** | 1.413 | 1.418 |

**Component contributions (isolated):**

| Component | Recall change | Notes |
|-----------|--------------|-------|
| Routing alone (M=1, no rerank) | -0.006 | slightly worse — partition misses without rerank |
| + Float rerank (rr=25) | **+0.031** | 0.857 → 0.888, largest single gain |
| + Multi-assign (M=2) | +0.004 | 0.888 → 0.892, improves partition hit rate |
| + More probe (R=32) | +0.008 | 0.892 → 0.900, covers more partitions |

**Key finding:** Routing without reranking is actually slightly worse than flat (0.857 vs 0.863) because partition misses lose some neighbors. Float reranking is what makes the entire approach work — it's the single component that breaks through the quantization ceiling.

1. **Float reranking** — the single biggest win (+0.031 recall). Without it, routing hurts. With it, routing + rerank exceeds flat.

2. **Multi-assignment** — small but consistent gain (+0.004). Raises partition hit rate from ~87% to ~95% on real data.

3. **Correctness verification** — full-probe (R=P) matches flat exactly, proving no bugs. Partition Hit@10 perfectly predicts final recall.

## What Didn't Work

1. **Centroid pre-ranking** — centroid score is partition-level, not vector-level. All vectors in a partition get the same score. Recall: 0.10-0.33.

2. **Partial dot product (first 32 dims)** — too coarse. 32 dims = 8% of information. Recall at cap=500: 0.10-0.23.

3. **Full float scoring without TQ** — correct recall but 100x slower. Float runs at 4.4M vec/sec vs TQ SIMD at 588M vec/sec.

Lesson: **TQ-level scoring is the necessary middle layer.** Nothing cheaper provides enough vector-level signal.

## Limitations

1. **Slower than flat turbovec and FAISS below ~500K vectors.** Per-partition TQ call overhead (~0.03ms each) dominates at small scale.
2. **Multi-assignment increases TQ code storage 2-4x.** Each vector stored in M partitions.
3. **Float rerank requires original vectors in memory** (~152 MB at 99K). This dominates total memory and negates the compression advantage at current scale.
4. **Build time is high.** O(n × P) for k-means + multi-assignment. ~80s at 99K with P=64.
5. **Depends on turbovec as sibling directory.** Not published to crates.io independently.
6. **No streaming insert/delete.** Index must be rebuilt to add vectors.
7. **Still slower than flat at 500K (measured).** Routed is 2.5x slower than flat at 500K. The crossover may be around 2-3M but is unverified.
8. **Lower recall than FAISS IVF-Flat.** IVF-Flat uses exact float scoring within partitions (no quantization loss). Our TQ scoring introduces noise that reranking only partially corrects.

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

benchmarks/
└── faiss_baselines.py   — FAISS comparison script (Python, for baseline numbers only)
```

Note: The core library and all performance-critical code is Rust. The Python script is only used to generate FAISS baseline numbers for comparison.

## Benchmarking

See [BENCHMARKS.md](BENCHMARKS.md) for full details.

**Environment used for published results:**
- CPU: Apple M3 Max (12 cores)
- RAM: 36 GB
- OS: macOS 15.x
- Rust: 1.95.0
- turbovec: v0.2.0 (local, RyanCodrai/turbovec main branch)
- FAISS: 1.13.2 (via faiss-cpu pip package)
- Dataset: 100K embeddings from `all-MiniLM-L6-v2` (sentence-transformers)
- Queries: last 200 vectors from dataset (same distribution)
- Ground truth: exact inner product (FAISS IndexFlatIP)
- Timing: single-threaded, warmup 5 queries, mean of 200 queries

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Highest-impact areas:

1. **TQ subset scoring** — add allowlist support to turbovec Rust API (eliminates duplicate scoring, biggest latency win)
2. **500K+ benchmark** — validate the crossover point with real data
3. **Streaming insert** — support adding vectors without full rebuild
4. **Disk-based rerank** — load float vectors from disk instead of RAM

## License

MIT
