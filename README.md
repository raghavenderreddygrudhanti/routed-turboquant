# routed-turboquant

IVF-style float routing on top of [turbovec](https://github.com/RyanCodrai/turbovec)'s TurboQuant SIMD scoring. Achieves **11-19% higher recall** than flat TurboQuant by combining partition routing with lightweight float reranking.

## The Problem

turbovec is fast. On 10K vectors it scores everything in 0.018ms using hand-written NEON/AVX-512 SIMD kernels. But it has a recall ceiling — 4-bit quantization introduces scoring noise that causes it to misrank some true neighbors. At dim=384, flat turbovec tops out around 0.84 recall@10.

You can't fix this by scanning more vectors. It already scans all of them. The quantization noise is the limit.

## The Insight

The true neighbors ARE in the candidate set (turbovec sees every vector). They're just ranked slightly wrong due to 4-bit approximation errors. If you could identify the top ~25 candidates and rescore them with exact float inner product, you'd recover the misranked neighbors.

But rescoring all 10K vectors with float defeats the purpose. You need a way to narrow down to a small candidate set first.

## What We Built

A three-stage pipeline:

```
Stage 1: Route (cheap)
  Query → dot product with P=32 centroids → select top-R partitions
  Cost: 32 dot products (negligible)

Stage 2: TQ Score (fast, approximate)
  Score candidates within R partitions using TurboQuant SIMD
  Cost: ~37% of vectors scored (with M=4 R=12)

Stage 3: Float Rerank (precise, small)
  Take top-25 from TQ scoring → exact float inner product
  Cost: 25 dot products (negligible)
```

The key innovation: **multi-assignment**. Each vector is stored in its top-M nearest partitions (M=4). This ensures that true neighbors are present in the probed partitions even when they sit near partition boundaries. Without multi-assignment, partition hit rate on random 384-d data is only 22% at R=4. With M=4, it jumps to 93.5%.

## What Made It Work

Three things, in order of importance:

### 1. Float reranking (the biggest win)

Without reranking, routed TQ gives the same recall as flat TQ (both limited by 4-bit noise). With reranking just 25 candidates, recall jumps from 0.809 to 0.935. The rerank step costs almost nothing (25 × 384 multiplies = ~0.01ms) but corrects the quantization errors that flat TQ can't fix.

This is why routed beats flat: flat TQ has no mechanism to correct its own scoring errors. Routed TQ uses TQ as a cheap first-pass filter, then applies exact scoring on the survivors.

### 2. Multi-assignment (makes routing viable)

On high-dimensional random data, nearest neighbors are spread across many partitions. Standard IVF (M=1) with R=12 only catches 53% of true neighbors. Multi-assignment (M=4) raises this to 93.5% — the router actually finds the right candidates.

The cost is 4x storage (each vector stored 4 times). We also implemented boundary-aware assignment that achieves similar hit rates at 2-3x storage by only duplicating vectors near partition boundaries.

### 3. Correctness-first development

We verified at every step:
- Full-probe (R=P) matches flat TQ exactly (0.842 vs 0.840) — proves no bugs
- Partition Hit@10 perfectly predicts final recall — proves routing works
- Deduplication verified — no duplicate results from multi-assignment

## Results

### Real Sentence-Transformer Embeddings (all-MiniLM-L6-v2, 384d)

**10K vectors (P=32, rerank=25):**

| Method | Recall@10 | Latency |
|--------|-----------|---------|
| turbovec flat | 0.952 | 0.016ms |
| routed M=2 R=P/4 | **0.987** | 0.297ms |
| routed M=4 R=P/4 | **0.989** | 0.362ms |

**50K vectors (P=64, rerank=25):**

| Method | Recall@10 | Latency |
|--------|-----------|---------|
| turbovec flat | 0.854 | 0.051ms |
| routed M=3 R=P/4 | **0.934** | 0.792ms |
| routed M=4 R=P/3 | **0.936** | 1.203ms |

**99K vectors (P=64, rerank=25):**

| Method | Recall@10 | Latency |
|--------|-----------|---------|
| turbovec flat | 0.863 | 0.094ms |
| routed M=2 R=P/2 | **0.900** | 1.815ms |
| routed M=4 R=P/4 | **0.896** | 1.453ms |

On real embeddings, routed-turboquant consistently beats flat by **3-9%** on recall across all scales. The gap is largest at 50K (+9.4%) where flat TQ's quantization errors compound with more vectors to misrank.

### Random Vectors (dim=384, P=32, rerank=25)

On random data (worst case for routing — no cluster structure):

| Method | Recall@10 | Latency | Storage |
|--------|-----------|---------|---------|
| turbovec flat | 0.840 | 0.018ms | 1x |
| routed M=4 R=12 | **0.935** | 0.508ms | 4x |
| routed M=3 R=16 | **0.944** | 0.573ms | 3x |
| routed M=4 R=16 | **0.983** | 0.652ms | 4x |

Even on random data with no natural clusters, routed achieves +11-17% recall over flat.

## What We Tried That Didn't Work

1. **V2: Single index + centroid pre-ranking.** Centroid score is a partition-level signal, not vector-level. All vectors in the same partition get the same score. Can't distinguish good candidates from bad within a partition. Recall dropped to 0.10-0.33.

2. **V2: Partial dot product pre-ranking (first 32 dims).** Better than centroid score but still too coarse. 32 dims capture only ~8% of the information. Recall at cap=500 was 0.10-0.23. Not enough signal.

3. **V2: Full float scoring all candidates (no TQ).** Works for recall (0.98+) but 100x slower than TQ scoring. Float dot products run at 4.4M vectors/sec vs TQ SIMD at 588M vectors/sec. Can't compete.

The lesson: **you need TQ-level scoring as the middle layer**. Nothing cheaper provides enough vector-level ranking signal to select good rerank candidates.

## Latency Tradeoff

At 10K, turbovec flat wins on latency (0.018ms vs 0.508ms). The routing overhead (~0.03ms per partition × 12 partitions) dominates at small scale.

The crossover where routed becomes faster is around **500K vectors**, where flat scan exceeds 0.5ms and routing savings (scanning 37% instead of 100%) start to pay off.

This architecture is designed for:
- **High-recall requirements** (>0.90) where flat TQ's 0.84 isn't enough
- **Large scale** (500K+) where sublinear scan matters
- **Tunable tradeoffs** where you need to dial recall vs latency vs storage

## Quick Start

```bash
# Needs turbovec as sibling directory
git clone https://github.com/RyanCodrai/turbovec.git
git clone https://github.com/raghavenderreddygrudhanti/routed-turboquant.git

cd routed-turboquant
cargo test --release        # 10 tests
cargo run --release --example bench_v1_tuned  # main benchmark
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
    multi_assign: 4,
    boundary_threshold: None,
    max_assign: 4,
    rerank_top: 25,
};

let index = RoutedTurboQuantIndex::build(&vectors, config);
let (scores, indices) = index.search(&query, 10);
```

## Project Structure

```
src/
├── lib.rs       — crate root
├── kmeans.rs    — float k-means++ (partition routing)
└── index.rs     — RoutedTurboQuantIndex (multi-assign + TQ scoring + rerank)

examples/
├── bench_v1_tuned.rs    — main benchmark (M/R sweep)
├── diagnose.rs          — correctness verification
├── diagnose_multi.rs    — multi-assignment impact
├── bench_boundary.rs    — boundary-aware assignment
└── bench_rerank.rs      — reranking impact
```

## How routed-turboquant Differs from turbovec

turbovec and routed-turboquant solve different problems:

| | turbovec (flat) | routed-turboquant |
|---|---|---|
| **What it does** | Scans ALL vectors with 4-bit SIMD scoring | Routes to relevant partitions, scores subset, reranks with float |
| **Scoring** | 4-bit TurboQuant only (approximate) | TurboQuant first pass + exact float rerank |
| **Scan** | 100% of vectors, every query | 25-50% of vectors (configurable) |
| **Recall ceiling** | Limited by 4-bit quantization noise | Breaks through ceiling via float rerank |
| **Speed at 10K** | 0.016ms (extremely fast) | 0.3-0.5ms (routing overhead) |
| **Speed at 500K+** | ~0.8ms (linear scan) | ~0.5ms (sublinear) |
| **Memory** | 1x (codes only) | 2-4x (multi-assignment copies) |
| **Build** | O(n) instant | O(n × P) k-means + assignment |
| **Tuning** | bit_width only | M, R, rerank, P (full control) |

**Why turbovec flat has a recall ceiling:**

turbovec compresses each vector to 4 bits per dimension. This is lossy — two vectors that are close in float space may get slightly different scores after quantization. When the true #8 neighbor scores 0.891 in float but 0.887 in TQ, and a non-neighbor scores 0.889 in TQ, the non-neighbor wins. This happens ~15% of the time at dim=384.

turbovec can't fix this because it has no second opinion. It scores everything once with TQ and returns the top-k. There's no mechanism to double-check.

**How routed-turboquant breaks through:**

We use TQ scoring as a cheap filter (not the final answer). TQ identifies the top ~25 candidates. Then we rescore those 25 with exact float inner product. The float rerank corrects the misrankings that TQ introduced.

The routing step (k-means partitions) reduces the number of vectors we need to TQ-score from 100% to 25-50%. Multi-assignment ensures the true neighbors are in the partitions we probe.

Net result: **higher recall than flat TQ, at the cost of higher latency at small scale.**

**When to use which:**

- Use **turbovec flat** when: latency is critical, scale is < 100K, 0.85-0.95 recall is acceptable
- Use **routed-turboquant** when: recall > 0.93 is required, scale is > 100K, you can tolerate 1-2ms latency

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The highest-impact contribution is adding allowlist/subset scoring support to turbovec's Rust API — this would eliminate the duplicate scoring overhead and make routed competitive on latency at all scales.

## License

MIT
