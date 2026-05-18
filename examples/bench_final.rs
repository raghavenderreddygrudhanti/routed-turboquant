//! Final benchmark: unique candidate %, scale test, proper timing.
//!
//! Addresses:
//! 1. Unique candidate % for each config
//! 2. Scale test at 50K, 100K, 500K
//! 3. Proper timing with warmup and multiple runs

extern crate blas_src;

use std::collections::HashSet;
use std::time::Instant;
use rand::rngs::StdRng;
use rand::SeedableRng;
use turbovec::TurboQuantIndex;
use routed_turboquant::index::{RoutedTQConfig, RoutedTurboQuantIndex};
use routed_turboquant::kmeans::kmeans_float;

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vecs = vec![0.0f32; n * dim];
    for i in 0..n {
        for d in 0..dim {
            vecs[i * dim + d] = rand::Rng::gen_range(&mut rng, -1.0..1.0);
        }
        let norm: f32 = vecs[i * dim..(i + 1) * dim].iter().map(|x| x * x).sum::<f32>().sqrt();
        for d in 0..dim {
            vecs[i * dim + d] /= norm;
        }
    }
    vecs
}

fn exact_topk(vectors: &[f32], query: &[f32], dim: usize, k: usize) -> Vec<usize> {
    let n = vectors.len() / dim;
    let mut scores: Vec<(f32, usize)> = (0..n)
        .map(|i| {
            let v = &vectors[i * dim..(i + 1) * dim];
            let sim: f32 = v.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
            (sim, i)
        })
        .collect();
    scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scores.iter().take(k).map(|(_, i)| *i).collect()
}

fn recall_score(predicted: &[usize], ground_truth: &[usize], k: usize) -> f64 {
    let pred_set: HashSet<usize> = predicted.iter().take(k).copied().collect();
    let gt_set: HashSet<usize> = ground_truth.iter().take(k).copied().collect();
    pred_set.intersection(&gt_set).count() as f64 / k as f64
}

/// Run timed search with warmup and multiple runs. Returns (mean_ms, p50_ms, p95_ms).
fn timed_search(
    idx: &RoutedTurboQuantIndex,
    queries: &[f32],
    dim: usize,
    nq: usize,
    k: usize,
    runs: usize,
) -> (f64, f64, f64) {
    // Warmup
    for i in 0..5.min(nq) {
        idx.search(&queries[i * dim..(i + 1) * dim], k);
    }

    let mut latencies: Vec<f64> = Vec::with_capacity(nq * runs);
    for _ in 0..runs {
        for i in 0..nq {
            let q = &queries[i * dim..(i + 1) * dim];
            let start = Instant::now();
            idx.search(q, k);
            latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }
    latencies.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
    (mean, p50, p95)
}

fn timed_flat_search(
    idx: &TurboQuantIndex,
    queries: &[f32],
    dim: usize,
    nq: usize,
    k: usize,
    runs: usize,
) -> (f64, f64, f64) {
    // Warmup
    idx.search(&queries[0..dim], k);

    let mut latencies: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        idx.search(queries, k);
        let per_query = start.elapsed().as_secs_f64() * 1000.0 / nq as f64;
        latencies.push(per_query);
    }
    latencies.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[latencies.len() - 1]; // few samples
    (mean, p50, p95)
}

fn main() {
    let dim = 384;
    let k = 10;
    let nq = 50;
    let runs = 3;

    // =========================================================
    // PART 1: Unique candidate % at 10K
    // =========================================================
    println!("=== PART 1: Unique Candidate % (n=10K, P=32) ===\n");
    let n = 10_000;
    let vectors = random_vectors(n, dim, 42);
    let queries = random_vectors(nq, dim, 123);

    println!("{:<22} {:<8} {:<10} {:<12} {:<12} {:<10} {:<10}",
             "Config", "Rerank", "Scan%", "RawEntries", "UniqueIDs", "Unique%n", "DupRatio");
    println!("{}", "-".repeat(84));

    let test_configs: Vec<(&str, usize, usize, usize)> = vec![
        ("M=3 R=12", 3, 12, 25),
        ("M=4 R=12", 4, 12, 25),
        ("M=3 R=16", 3, 16, 25),
        ("M=4 R=16", 4, 16, 25),
    ];

    for (label, m, r, rerank) in &test_configs {
        let config = RoutedTQConfig {
            dim, n_partitions: 32, n_probe: *r, bit_width: 4,
            kmeans_iter: 10, seed: 42, multi_assign: *m,
            boundary_threshold: None, max_assign: 4, rerank_top: *rerank,
        };
        let idx = RoutedTurboQuantIndex::build(&vectors, config);

        let mut raw_sum: usize = 0;
        let mut tq_sum: usize = 0;
        let mut unique_sum: usize = 0;
        let mut rerank_sum: usize = 0;
        for i in 0..nq {
            let q = &queries[i * dim..(i + 1) * dim];
            let (_, stats) = idx.search_stats(q, k);
            raw_sum += stats.raw_partition_entries;
            tq_sum += stats.tq_results_collected;
            unique_sum += stats.unique_ids;
            rerank_sum += stats.rerank_k;
        }
        let avg_raw = raw_sum as f64 / nq as f64;
        let avg_unique = unique_sum as f64 / nq as f64;
        let unique_pct_n = (avg_unique / n as f64) * 100.0;
        let dup_ratio = avg_raw / avg_unique;
        let scan_pct = (*r as f64 / 32.0) * 100.0;

        println!("{:<22} {:<8} {:<10.1} {:<12.0} {:<12.0} {:<10.1} {:<10.1}",
                 label, rerank, scan_pct, avg_raw, avg_unique, unique_pct_n, dup_ratio);
    }

    // =========================================================
    // PART 2: Scale benchmark
    // =========================================================
    println!("\n\n=== PART 2: Scale Benchmark ===");
    println!("Best configs vs turbovec flat at increasing scale\n");

    for &n in &[10_000, 50_000, 100_000] {
        println!("--------------------------------------------------------------");
        println!("  n = {:>7}  dim={}  k={}  nq={}  runs={}", n, dim, k, nq, runs);
        println!("--------------------------------------------------------------");

        let vectors = random_vectors(n, dim, 42);
        let queries = random_vectors(nq, dim, 123);

        // Exact ground truth
        let exact_gt: Vec<Vec<usize>> = (0..nq)
            .map(|i| exact_topk(&vectors, &queries[i * dim..(i + 1) * dim], dim, k))
            .collect();

        // Turbovec flat
        let mut flat_idx = TurboQuantIndex::new(dim, 4);
        flat_idx.add(&vectors);
        flat_idx.prepare();

        let (flat_mean, flat_p50, flat_p95) = timed_flat_search(&flat_idx, &queries, dim, nq, k, runs);

        let flat_results = flat_idx.search(&queries, k);
        let mut flat_recall_sum = 0.0;
        for i in 0..nq {
            let pred: Vec<usize> = flat_results.indices_for_query(i)
                .iter().filter(|&&x| x >= 0).map(|&x| x as usize).collect();
            flat_recall_sum += recall_score(&pred, &exact_gt[i], k);
        }
        let flat_recall = flat_recall_sum / nq as f64;

        println!("  turbovec flat:       recall={:.3}  mean={:.3}ms  p50={:.3}ms  p95={:.3}ms",
                 flat_recall, flat_mean, flat_p50, flat_p95);

        // Routed configs — use appropriate P and R for each scale
        let p = if n <= 10_000 { 32 } else if n <= 50_000 { 64 } else { 128 };

        // Choose R values that give 25-50% scan
        let r_values: Vec<usize> = if n <= 10_000 {
            vec![12, 16]
        } else if n <= 50_000 {
            vec![16, 24, 32]
        } else {
            vec![24, 32, 48]
        };

        // Compute partition assignments for PartHit@10
        let mut unit_vectors = vectors.clone();
        for i in 0..n {
            let s = i * dim;
            let e = s + dim;
            let norm: f32 = unit_vectors[s..e].iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 { for x in unit_vectors[s..e].iter_mut() { *x /= norm; } }
        }
        let km = kmeans_float(&unit_vectors, n, dim, p, 10, 42);
        let centroids_flat: Vec<f32> = km.centroids.iter().flatten().copied().collect();

        // Multi-assign: compute top-4 partitions per vector
        let multi_asgn: Vec<Vec<u32>> = (0..n).map(|i| {
            let v = &unit_vectors[i * dim..(i + 1) * dim];
            let mut scores: Vec<(f32, u32)> = (0..p).map(|pid| {
                let c = &centroids_flat[pid * dim..(pid + 1) * dim];
                let sim: f32 = v.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
                (sim, pid as u32)
            }).collect();
            scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            scores.iter().take(4).map(|(_, pid)| *pid).collect()
        }).collect();

        for &r in &r_values {
            let config = RoutedTQConfig {
                dim, n_partitions: p, n_probe: r, bit_width: 4,
                kmeans_iter: 10, seed: 42, multi_assign: 4,
                boundary_threshold: None, max_assign: 4, rerank_top: 25,
            };

            let build_start = Instant::now();
            let idx = RoutedTurboQuantIndex::build(&vectors, config);
            let build_s = build_start.elapsed().as_secs_f64();

            // PartHit@10
            let mut part_hit_sum = 0.0;
            for i in 0..nq {
                let q = &queries[i * dim..(i + 1) * dim];
                let q_norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
                let q_unit: Vec<f32> = q.iter().map(|x| x / q_norm).collect();
                let mut cs: Vec<(f32, usize)> = (0..p).map(|pid| {
                    let c = &centroids_flat[pid * dim..(pid + 1) * dim];
                    let sim: f32 = q_unit.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
                    (sim, pid)
                }).collect();
                cs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                let selected: HashSet<u32> = cs.iter().take(r).map(|(_, pid)| *pid as u32).collect();

                let mut hits = 0;
                for &gt_id in exact_gt[i].iter().take(k) {
                    if multi_asgn[gt_id].iter().any(|pid| selected.contains(pid)) {
                        hits += 1;
                    }
                }
                part_hit_sum += hits as f64 / k as f64;
            }
            let avg_part_hit = part_hit_sum / nq as f64;

            // Timed search
            let (mean, p50, p95) = timed_search(&idx, &queries, dim, nq, k, runs);
            let speedup = flat_mean / mean;

            // Recall
            let mut recall_sum = 0.0;
            for i in 0..nq {
                let q = &queries[i * dim..(i + 1) * dim];
                let (_, ids) = idx.search(q, k);
                recall_sum += recall_score(&ids, &exact_gt[i], k);
            }
            let avg_recall = recall_sum / nq as f64;
            let scan_pct = (r as f64 / p as f64) * 100.0;

            println!("  M=4 R={:<2} rr=25 P={:<3}: recall={:.3}  PartHit={:.3}  mean={:.3}ms  p50={:.3}ms  p95={:.3}ms  speedup={:.2}x  scan={:.1}%  build={:.1}s",
                     r, p, avg_recall, avg_part_hit, mean, p50, p95, speedup, scan_pct, build_s);
        }
        println!();
    }
}
