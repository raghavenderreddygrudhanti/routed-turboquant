//! Benchmark: boundary-aware multi-assignment vs fixed M.
//!
//! Compares storage efficiency and recall for:
//! - Fixed M=1 (baseline)
//! - Fixed M=3, M=4
//! - Boundary-aware with various thresholds

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
    let nq = 100;
    let n = 10_000;
    let p = 32;
    let r = 12; // 37.5% scan

    println!("n={}, dim={}, k={}, nq={}, P={}, R={} (scan={:.1}%)",
             n, dim, k, nq, p, r, (r as f64 / p as f64) * 100.0);
    println!();

    let vectors = random_vectors(n, dim, 42);
    let queries = random_vectors(nq, dim, 123);

    // Ground truth
    let exact_gt: Vec<Vec<usize>> = (0..nq)
        .map(|i| exact_topk(&vectors, &queries[i * dim..(i + 1) * dim], dim, k))
        .collect();

    // Turbovec flat baseline
    let mut flat_idx = TurboQuantIndex::new(dim, 4);
    flat_idx.add(&vectors);
    flat_idx.prepare();
    let flat_results = flat_idx.search(&queries, k);
    let mut flat_recall_sum = 0.0;
    for i in 0..nq {
        let pred: Vec<usize> = flat_results.indices_for_query(i)
            .iter().filter(|&&x| x >= 0).map(|&x| x as usize).collect();
        flat_recall_sum += recall_score(&pred, &exact_gt[i], k);
    }
    let flat_recall = flat_recall_sum / nq as f64;
    println!("turbovec flat:  recall={:.3} (ceiling)\n", flat_recall);

    println!("{:<25} {:<12} {:<12} {:<12} {:<12}",
             "Method", "StorageX", "Recall", "Latency_ms", "Recall/Store");
    println!("{}", "-".repeat(73));

    // Helper to benchmark a config
    let bench = |label: &str, config: RoutedTQConfig| {
        let idx = RoutedTurboQuantIndex::build(&vectors, config);
        let sf = idx.storage_factor();

        let start = Instant::now();
        let mut recall_sum = 0.0;
        for i in 0..nq {
            let q = &queries[i * dim..(i + 1) * dim];
            let (_, ids) = idx.search(q, k);
            recall_sum += recall_score(&ids, &exact_gt[i], k);
        }
        let elapsed = start.elapsed();
        let latency = elapsed.as_secs_f64() * 1000.0 / nq as f64;
        let avg_recall = recall_sum / nq as f64;
        let efficiency = avg_recall / sf; // recall per unit storage

        println!("{:<25} {:<12.2} {:<12.3} {:<12.3} {:<12.3}",
                 label, sf, avg_recall, latency, efficiency);
    };

    // Fixed M=1
    bench("Fixed M=1", RoutedTQConfig {
        dim, n_partitions: p, n_probe: r, bit_width: 4,
        kmeans_iter: 10, seed: 42, multi_assign: 1,
        boundary_threshold: None, max_assign: 4, rerank_top: 0,
    });

    // Fixed M=2
    bench("Fixed M=2", RoutedTQConfig {
        dim, n_partitions: p, n_probe: r, bit_width: 4,
        kmeans_iter: 10, seed: 42, multi_assign: 2,
        boundary_threshold: None, max_assign: 4, rerank_top: 0,
    });

    // Fixed M=3
    bench("Fixed M=3", RoutedTQConfig {
        dim, n_partitions: p, n_probe: r, bit_width: 4,
        kmeans_iter: 10, seed: 42, multi_assign: 3,
        boundary_threshold: None, max_assign: 4, rerank_top: 0,
    });

    // Fixed M=4
    bench("Fixed M=4", RoutedTQConfig {
        dim, n_partitions: p, n_probe: r, bit_width: 4,
        kmeans_iter: 10, seed: 42, multi_assign: 4,
        boundary_threshold: None, max_assign: 4, rerank_top: 0,
    });

    println!();

    // Boundary-aware with various thresholds
    for &threshold in &[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.10] {
        let label = format!("Boundary t={:.3}", threshold);
        bench(&label, RoutedTQConfig {
            dim, n_partitions: p, n_probe: r, bit_width: 4,
            kmeans_iter: 10, seed: 42, multi_assign: 1,
            boundary_threshold: Some(threshold), max_assign: 4, rerank_top: 0,
        });
    }

    println!();

    // Boundary-aware with max_assign=6 (allow more copies for very boundary vectors)
    println!("--- With max_assign=6 ---");
    for &threshold in &[0.02, 0.03, 0.04, 0.05] {
        let label = format!("Boundary t={:.3} M<=6", threshold);
        bench(&label, RoutedTQConfig {
            dim, n_partitions: p, n_probe: r, bit_width: 4,
            kmeans_iter: 10, seed: 42, multi_assign: 1,
            boundary_threshold: Some(threshold), max_assign: 6, rerank_top: 0,
        });
    }
}
