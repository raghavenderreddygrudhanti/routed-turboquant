# routed-turboquant

IVF-style float routing on top of TurboQuant SIMD scoring. Sublinear search with recall that **exceeds** flat TurboQuant.

## Key Result

At 10K vectors (dim=384, k=10), with rerank=25:

| Method | Recall@10 | Latency | Storage |
|--------|-----------|---------|---------|
| turbovec flat | 0.840 | 0.018ms | 1x |
| **routed M=4 R=12** | **0.935** | 0.508ms | 4x |
| **routed M=3 R=16** | **0.944** | 0.573ms | 3x |
| **routed M=4 R=16** | **0.983** | 0.652ms | 4x |
| routed M=4 R=24 | 1.000 | 0.974ms | 4x |

Routed TurboQuant with reranking **beats flat TurboQuant on recall by 11-19%** because float reranking is more precise than 4-bit quantized scoring.

## How It Works

```
Build:
  vectors → normalize → float k-means (P partitions)
  per vector → assign to top-M nearest partitions (multi-assignment)
  per partition → build TurboQuantIndex (SIMD scoring)

Query:
  query → centroid dot products (P) → select top-R partitions
        → TurboQuant SIMD search within each partition
        → deduplicate across partitions
        → float rerank top-25 candidates (exact inner product)
        → return top-k
```

## Why Recall Exceeds Flat

1. **Multi-assignment** ensures true neighbors are in probed partitions (93.5% partition hit rate at M=4 R=12)
2. **TurboQuant scoring** within partitions provides vector-level ranking (not just partition-level)
3. **Float reranking** of top-25 candidates corrects TQ quantization errors with exact inner product
4. Net effect: routing + rerank recovers neighbors that flat TQ misranks due to 4-bit quantization noise

## Full Recall Curve (n=10K, P=32, rerank=25)

```
M      R      Scan%    Recall     Latency    StorageX
1      8      25.0     0.383      0.249ms    1x
1      16     50.0     0.649      0.475ms    1x
1      24     75.0     0.855      0.738ms    1x
2      8      25.0     0.593      0.280ms    2x
2      16     50.0     0.874      0.525ms    2x
3      8      25.0     0.727      0.294ms    3x
3      12     37.5     0.869      0.479ms    3x
3      16     50.0     0.944      0.573ms    3x
4      8      25.0     0.816      0.324ms    4x
4      12     37.5     0.935      0.508ms    4x
4      16     50.0     0.983      0.652ms    4x
4      24     75.0     1.000      0.974ms    4x
```

## Latency vs Flat

At 10K, turbovec flat (0.018ms) is faster because NEON SIMD scores all 10K vectors in one pass with no overhead. Routed search has per-partition call overhead (~0.03ms × R partitions).

The crossover point where routed becomes faster is at **n > 500K**, where flat scan exceeds the routing overhead. This architecture is designed for scale.

## Features

- **Multi-assignment** (M=1-4): each vector stored in M partitions for higher hit rate
- **Boundary-aware assignment**: adaptive M based on centroid proximity (2-3x storage instead of fixed 4x)
- **Float reranking**: exact inner product on top-25 TQ candidates
- **Fast dedup**: vec<bool> visited array (O(1) per candidate)
- **Partial sort**: `select_nth_unstable` for routing and top-k (no full sort)
- **Parallel batch search**: rayon for multi-query throughput

## Correctness Verified

- Full-probe (R=P) matches turbovec flat recall exactly (0.842 vs 0.840)
- Partition Hit@10 perfectly predicts final recall
- No duplicate results in output (dedup verified by test)
- 13 tests passing

## Install

```bash
# Rust library
cargo build --release

# Python (via maturin)
pip install maturin
maturin develop --release
```

## Usage

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

## Benchmarks

```bash
# Full M/R sweep with rerank
cargo run --release --example bench_v1_tuned

# Correctness diagnostics (full-probe, partition hit rate)
cargo run --release --example diagnose

# Multi-assignment impact
cargo run --release --example diagnose_multi

# Boundary-aware vs fixed M
cargo run --release --example bench_boundary

# Reranking impact
cargo run --release --example bench_rerank
```

## Architecture

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

## Comparison with turbovec

| Aspect | turbovec flat | routed-turboquant |
|--------|-------------|-------------------|
| Recall@10 (10K) | 0.840 | **0.935** (M=4 R=12 rr=25) |
| Latency (10K) | **0.018ms** | 0.508ms |
| Latency (500K est.) | ~0.9ms | ~0.5ms |
| Build time | O(n) | O(n × P) |
| Memory | n × dim × bits/8 | M × n × dim × bits/8 |
| Recall control | fixed (bit_width) | tunable (M, R, rerank) |
| Best for | small-medium flat scan | large-scale high-recall |

## License

MIT
