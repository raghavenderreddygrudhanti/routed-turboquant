# Improving TurboQuant Recall with Routed Candidate Generation and Float Reranking

**Raghavender Reddy Grudhanti**

## Abstract

TurboQuant-style 4-bit compressed vector search provides very low latency through SIMD-optimized scoring kernels, but its recall is limited by quantization-induced ranking errors. We present routed-turboquant, a three-stage retrieval pipeline that combines centroid-based partition routing, TurboQuant SIMD candidate scoring, and exact float reranking of a small candidate set. On 99K real sentence-transformer embeddings (all-MiniLM-L6-v2, dim=384), routed-turboquant improves Recall@10 from 0.863 to 0.900 over flat TurboQuant. Ablation results show that float reranking is the primary contributor (+0.031 recall), while multi-assignment and additional partition probes provide smaller gains. However, routed search remains 2.5x slower than flat TurboQuant at 500K vectors due to fixed per-partition search overhead. These results position routed-turboquant as a high-recall candidate generation approach and motivate future fused subset-scoring support in compressed ANN systems.

## 1. Introduction

Approximate nearest neighbor (ANN) search in high-dimensional spaces is fundamental to retrieval-augmented generation, recommendation systems, and semantic search. TurboQuant (ICLR 2026) achieves near-optimal distortion by compressing vectors to 2-4 bits per coordinate using data-oblivious scalar quantization after random rotation. The turbovec implementation provides extremely fast SIMD scoring — processing 10K vectors in 0.016ms on ARM NEON.

However, 4-bit quantization introduces scoring noise that limits recall. At dim=384 on real sentence-transformer embeddings, flat TurboQuant achieves 0.863 Recall@10 at 99K scale — meaning ~14% of true nearest neighbors are misranked due to quantization errors. This ceiling cannot be overcome by scanning more vectors, since flat TQ already scores all of them.

We observe that the true neighbors are often present in the TQ candidate set but misranked. A lightweight float reranking of the top-25 TQ candidates can correct these misrankings. The challenge is identifying which 25 candidates to rescore without computing exact distances for all vectors.

Our approach uses IVF-style partition routing with multi-assignment to generate a candidate set, TurboQuant SIMD scoring within partitions for vector-level ranking, and exact float inner product reranking of the top-25 candidates. This three-stage pipeline achieves recall that exceeds flat TQ by 3-9% across scales from 10K to 99K vectors.

## 2. Method

### 2.1 Architecture

The routed-turboquant pipeline consists of three stages:

**Stage 1: Partition Routing.** Given P partitions built by float k-means on unit-normalized vectors, route the query to the R nearest centroids by inner product. Cost: P dot products (negligible at P=32-128).

**Stage 2: TurboQuant Scoring.** Score candidates within the R selected partitions using per-partition TurboQuantIndex instances with SIMD-accelerated 4-bit scoring. Each vector is scored exactly once within its partition. With multi-assignment (M>1), vectors appear in multiple partitions, increasing the probability that true neighbors are in the probed set.

**Stage 3: Float Reranking.** Collect the top-25 candidates by TQ score across all probed partitions (after deduplication), and rescore them with exact float32 inner product against the original vectors. Return the top-k by exact score.

### 2.2 Multi-Assignment

Standard IVF assigns each vector to its single nearest centroid. On high-dimensional random data (dim=384), this results in only 22% partition hit rate at R=4 probes — most true neighbors are in unprobed partitions.

Multi-assignment stores each vector in its M nearest partitions. With M=4, partition hit rate rises to 93.5% on random data and >95% on real embeddings. The cost is M× storage for TQ codes.

We also implement boundary-aware adaptive assignment, where vectors near partition boundaries are stored in 2-4 partitions while vectors deep inside clusters are stored once. This achieves similar hit rates at 2-3× storage instead of fixed 4×.

### 2.3 Why Reranking Works

TurboQuant's 4-bit Lloyd-Max quantization introduces per-coordinate errors bounded by the quantization cell width. When two vectors have similar true inner products, the quantized scores may swap their ordering. Float reranking corrects these swaps for the top candidates.

Critically, reranking only works if the true neighbors are already in the candidate set. Our ablation shows that routing without reranking actually hurts recall (-0.006) because partition misses lose some neighbors. Reranking is what makes the entire approach viable — it's the mechanism that breaks through the quantization ceiling.

## 3. Experimental Setup

**Dataset:** 100K sentence-transformer embeddings generated from all-MiniLM-L6-v2. Text source: combinatorial phrases from 40 CS topics × 27 verbs × 16 contexts. Vectors are L2-normalized, dim=384.

**Queries:** Last 200 vectors from the dataset (same distribution). Ground truth: exact inner product over all n vectors.

**Baselines:**
- turbovec flat: TurboQuantIndex with 4-bit, full scan (RyanCodrai/turbovec v0.2.0)
- FAISS Flat: exact brute-force float32 (faiss-cpu 1.13.2)
- FAISS HNSW: M=32, efConstruction=200, efSearch=64
- FAISS IVF-Flat: nlist=64/128, nprobe=16
- FAISS IVFPQ: nlist=64/128, m=48, nbits=8, nprobe=16

**Environment:** Apple M3 Max (12 cores), 36 GB RAM, macOS, Rust 1.95.0. Single-threaded, warmup 5 queries, mean of 200 queries.

## 4. Results

### 4.1 Main Results on Real Embeddings

| Scale | Method | Recall@10 | Latency |
|-------|--------|-----------|---------|
| 10K | FAISS Flat (exact) | 1.000 | 0.014ms |
| 10K | FAISS IVF-Flat | 0.989 | 0.016ms |
| 10K | **routed-turboquant** | **0.987** | 0.297ms |
| 10K | FAISS HNSW | 0.981 | 0.008ms |
| 10K | turbovec flat | 0.952 | 0.016ms |
| 10K | FAISS IVFPQ | 0.871 | 0.012ms |
| 50K | FAISS IVF-Flat | 0.993 | 0.118ms |
| 50K | **routed-turboquant** | **0.936** | 1.203ms |
| 50K | FAISS HNSW | 0.931 | 0.023ms |
| 50K | turbovec flat | 0.854 | 0.051ms |
| 50K | FAISS IVFPQ | 0.734 | 0.028ms |
| 99K | FAISS IVF-Flat | 0.986 | 0.135ms |
| 99K | **routed-turboquant** | **0.900** | 1.815ms |
| 99K | FAISS HNSW | 0.888 | 0.018ms |
| 99K | turbovec flat | 0.863 | 0.094ms |
| 99K | FAISS IVFPQ | 0.822 | 0.027ms |

Routed-turboquant achieves higher recall than turbovec flat (+3.7% at 10K, +9.6% at 50K, +4.3% at 99K) and FAISS HNSW at 50K+. It uses quantized 4-bit storage (unlike FAISS IVF-Flat which stores full float32) but achieves near-IVF-Flat recall.

### 4.2 Ablation Study

Measured on 99K real embeddings, P=64:

| Method | Routing | Multi-assign | Rerank | Recall@10 | Mean ms |
|--------|---------|-------------|--------|-----------|---------|
| turbovec flat | — | — | — | 0.863 | 0.085 |
| routed M=1 R=16 | Yes | M=1 | No | 0.857 | 0.635 |
| routed M=1 R=16 rr=25 | Yes | M=1 | Yes | **0.888** | 0.674 |
| routed M=2 R=16 rr=25 | Yes | M=2 | Yes | **0.892** | 0.891 |
| routed M=2 R=32 rr=25 | Yes | M=2 | Yes | **0.900** | 1.726 |
| routed M=4 R=16 rr=25 | Yes | M=4 | Yes | **0.896** | 1.413 |

**Component contributions (isolated):**

| Component | Recall change |
|-----------|--------------|
| Routing alone (M=1, no rerank) | -0.006 (hurts) |
| + Float rerank (rr=25) | **+0.031** (largest gain) |
| + Multi-assign (M=2) | +0.004 |
| + More probe (R=32) | +0.008 |

Float reranking is the dominant contributor. Routing without reranking is slightly worse than flat because partition misses lose some neighbors that flat TQ would have scored.

### 4.3 Latency Analysis

| Scale | turbovec flat | routed (25% scan) | Speedup |
|-------|--------------|-------------------|---------|
| 10K | 0.016ms | 0.297ms | 0.05x |
| 50K | 0.051ms | 0.792ms | 0.06x |
| 100K | 0.107ms | 0.693ms | 0.15x |
| 200K | 0.229ms | 0.909ms | 0.25x |
| 500K | 0.605ms | 1.488ms | 0.41x |

The gap closes with scale (0.05x → 0.41x) but routed remains slower at all tested scales. The bottleneck is per-partition TQ search() call overhead (~0.03ms per partition), not the actual vector scoring.

### 4.4 Partition Hit Rate on Real Data

| M | R | Scan% | PartHit@10 | Recall (rr=25) |
|---|---|-------|------------|----------------|
| 1 | 16 | 25% | 0.87 | 0.888 |
| 2 | 16 | 25% | 0.95 | 0.892 |
| 2 | 32 | 50% | 0.98 | 0.900 |
| 4 | 16 | 25% | 0.98 | 0.896 |

On real embeddings, M=2 achieves 95%+ partition hit rate due to natural semantic cluster structure. The rerank step adds +2-3% recall on top of TQ-only scoring.

## 5. Discussion

### 5.1 Positive Results

1. **Float reranking breaks the quantization ceiling.** The single most important finding: reranking just 25 candidates with exact float inner product improves recall by +0.031 (3.6%) over TQ-only scoring. This is because TQ misranks ~14% of true neighbors due to 4-bit noise, and reranking corrects most of these errors.

2. **Near-IVF-Flat recall with quantized storage.** At 10K, routed-turboquant achieves 0.987 recall — within 0.002 of FAISS IVF-Flat (0.989) — while using 4-bit codes instead of full float32 vectors for candidate scoring.

3. **Beats FAISS HNSW at scale.** At 50K (0.936 vs 0.931) and 99K (0.900 vs 0.888), routed-turboquant exceeds HNSW recall because HNSW's graph connectivity degrades at scale while our routing + rerank approach maintains precision.

### 5.2 Negative Results

1. **Routing without reranking hurts.** Ablation shows -0.006 recall when routing is added without reranking. Partition misses lose neighbors that flat TQ would have scored. The routing is only justified by enabling the rerank step.

2. **Latency is worse at all tested scales.** At 500K, routed is still 2.5x slower than flat TQ. The per-partition TQ search() overhead (~0.03ms × 16 partitions = 0.48ms) dominates regardless of how many vectors are in each partition.

3. **QJL residual correction provides no benefit.** We implemented TurboQuant_prod-style 1-bit residual correction but found negligible recall improvement (+0.1% at 10K, -0.4% at 50K) because our rotation matrix doesn't match turbovec's internal rotation.

### 5.3 Bottleneck Analysis

The latency bottleneck is not routing accuracy or candidate quality — it's execution architecture. Each per-partition TurboQuantIndex::search() call has ~0.03ms fixed overhead for rotation matrix lookup and blocked cache access. With 16 partitions probed, that's 0.48ms of overhead regardless of partition size.

The ideal architecture would be: route → collect unique candidate IDs → score those IDs against a single TQ index with allowlist/mask support → rerank top-25. This would eliminate per-partition overhead entirely. However, turbovec's current Rust API does not support subset scoring.

### 5.4 Relation to MRQ

The concurrent MRQ work (Yang et al., 2024) takes a different approach: PCA-based dimension decomposition with multi-stage error-bound-driven distance correction. MRQ achieves 3x speedup over RaBitQ by exploiting the long-tailed variance distribution after PCA rotation. Our approach is complementary — we focus on improving recall over flat TQ scoring, while MRQ focuses on improving efficiency of distance correction. Combining PCA-ordered dimensions with our routing + rerank pipeline is a promising direction.

## 6. Limitations

1. Slower than flat TQ at all tested scales (up to 500K).
2. Multi-assignment increases TQ code storage 2-4×.
3. Float rerank requires original vectors in memory (~152 MB at 99K).
4. Build time is O(n × P) for k-means + multi-assignment (~80s at 99K).
5. Query selection (last 200 from same distribution) may overestimate recall in production settings.
6. Tested only on one embedding model (all-MiniLM-L6-v2, dim=384).

## 7. Conclusion

We presented routed-turboquant, a three-stage pipeline that improves TurboQuant recall by 3-9% on real embeddings through partition routing, multi-assignment, and lightweight float reranking. Ablation shows float reranking is the critical component — routing alone hurts recall. The approach achieves near-FAISS-IVF-Flat recall using quantized storage, but remains slower than flat TQ due to per-partition execution overhead. Future work should focus on fused subset-scoring support in compressed ANN systems to eliminate this overhead.

## References

1. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate. ICLR 2026.
2. RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound. SIGMOD 2024.
3. MRQ: Fast High-dimensional Approximate Nearest Neighbor Search with Efficient Index Time and Space. Yang et al., 2024.
4. HNSW: Efficient and Robust Approximate Nearest Neighbor Search. Malkov & Yashunin, 2020.
5. FAISS: Billion-scale similarity search with GPUs. Johnson et al., 2019.
6. DiskANN: Fast accurate billion-point nearest neighbor search. Subramanya et al., 2019.

## Reproducibility

Code: https://github.com/raghavenderreddygrudhanti/routed-turboquant
Dataset: 100K embeddings from all-MiniLM-L6-v2 (sentence-transformers)
Commands: `cargo run --release --example ablation` (ablation study), `cargo run --release --example bench_real_99k` (main results)
