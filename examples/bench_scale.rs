//! Scale benchmark: find where routed multi-assign beats turbovec flat on latency.
//!
//! At small n, flat scan is faster (no routing overhead).
//! At large n, routed scan wins because it only touches R/P fraction of vectors.
//! This benchmark finds the crossover.

extern crate blas_src;

use std::collections::HashSet;
use std::time::Instant;
use rand::rngs::StdRng;
use rand::SeedableRng;
use turbovec::TurboQuantIndex;
use routed_turboquant::index::{RoutedTQConfig, RoutedTurboQuantIndex};

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

fn main() {
    let dim = 384;
    let k = 10;
    let nq = 50;

    println!("=== Scale Benchmark: Routed Multi-Assign vs Turbovec Flat ===");
    println!("dim={}, k={}, nq={}\n", dim, k, nq);

    for &n in &[10_000, 50_000, 100_000, 200_000] {
        println!("======================================================================");
        println!("  n = {:>7}", n);
        println!("======================================================================");

        let vectors = random_vectors(n, dim, 42);
        let queries = random_vectors(nq, dim, 123);

        // Exact ground truth (skip for 200K — too slow)
        let exact_gt: Option<Vec<Vec<usize>>> = if n <= 100_000 {
            Some((0..nq)
                .map(|i| exact_topk(&vectors, &queries[i * dim..(i + 1) * dim], dim, k))
                .collect())
        } else {
            None
        };

        // Turbovec flat
        let mut flat_idx = TurboQuantIndex::new(dim, 4);
        flat_idx.add(&vectors);
        flat_idx.prepare();

        // Warmup
        flat_idx.search(&queries[0..dim], k);

        let start = Instant::now();
        let flat_results = flat_idx.search(&queries, k);
        let flat_elapsed = start.elapsed();
        let flat_latency = flat_elapsed.as_secs_f64() * 1000.0 / nq as f64;
        let flat_qps = nq as f64 / flat_elapsed.as_secs_f64();

        let flat_recall = if let Some(ref gt) = exact_gt {
            let mut sum = 0.0;
            for i in 0..nq {
                let pred: Vec<usize> = flat_results.indices_for_query(i)
                    .iter().filter(|&&x| x >= 0).map(|&x| x as usize).collect();
                sum += recall_score(&pred, &gt[i], k);
            }
            sum / nq as f64
        } else {
            -1.0
        };

        if flat_recall >= 0.0 {
            println!("  turbovec flat:       recall={:.3}  latency={:.3}ms  QPS={:.0}",
                     flat_recall, flat_latency, flat_qps);
        } else {
            println!("  turbovec flat:       latency={:.3}ms  QPS={:.0}",
                     flat_latency, flat_qps);
        }

        // Routed configs: best from diagnostic (M=3 R=8, M=4 R=8)
        let p = if n <= 10_000 { 32 } else if n <= 50_000 { 64 } else { 128 };

        let configs: Vec<(usize, usize, &str)> = vec![
            (3, 8, "M=3 R=8"),
            (4, 8, "M=4 R=8"),
            (3, 12, "M=3 R=12"),
            (4, 12, "M=4 R=12"),
            (3, 16, "M=3 R=16"),
        ];

        for (m, r, label) in configs {
            if r > p {
                continue;
            }

            let config = RoutedTQConfig {
                dim,
                n_partitions: p,
                n_probe: r,
                bit_width: 4,
                kmeans_iter: 10,
                seed: 42,
                multi_assign: m, boundary_threshold: None, max_assign: 4, rerank_top: 0,
            };

            let build_start = Instant::now();
            let routed_idx = RoutedTurboQuantIndex::build(&vectors, config);
            let build_time = build_start.elapsed();

            // Warmup
            routed_idx.search(&queries[0..dim], k);

            let start = Instant::now();
            let mut recall_sum = 0.0;
            for i in 0..nq {
                let q = &queries[i * dim..(i + 1) * dim];
                let (_, routed_ids) = routed_idx.search(q, k);
                if let Some(ref gt) = exact_gt {
                    recall_sum += recall_score(&routed_ids, &gt[i], k);
                }
            }
            let elapsed = start.elapsed();
            let latency = elapsed.as_secs_f64() * 1000.0 / nq as f64;
            let qps = nq as f64 / elapsed.as_secs_f64();
            let scan_pct = (r as f64 / p as f64) * 100.0;
            let speedup = flat_latency / latency;

            if exact_gt.is_some() {
                let avg_recall = recall_sum / nq as f64;
                println!("  routed P={:<3} {:<8}: recall={:.3}  latency={:.3}ms  QPS={:.0}  scan={:.1}%  speedup={:.2}x  build={:.1}s",
                         p, label, avg_recall, latency, qps, scan_pct, speedup, build_time.as_secs_f64());
            } else {
                println!("  routed P={:<3} {:<8}: latency={:.3}ms  QPS={:.0}  scan={:.1}%  speedup={:.2}x  build={:.1}s",
                         p, label, latency, qps, scan_pct, speedup, build_time.as_secs_f64());
            }
        }
        println!();
    }
}
