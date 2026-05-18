//! Benchmark: float reranking impact on recall.
//!
//! Tests: Route → TQ score → take top N → rerank with float
//! Expected: reranking closes the gap between routed and turbovec flat.

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

    println!("n={}, dim={}, k={}, nq={}, P={}", n, dim, k, nq, p);

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
    println!("turbovec flat recall: {:.3}\n", flat_recall);

    println!("{:<20} {:<8} {:<12} {:<12} {:<12} {:<12}",
             "Config", "Rerank", "Recall", "vs Flat", "Latency_ms", "StorageX");
    println!("{}", "-".repeat(76));

    // Test matrix: M × R × rerank_top
    let configs: Vec<(&str, usize, usize)> = vec![
        ("M=3 R=12", 3, 12),
        ("M=4 R=12", 4, 12),
        ("M=3 R=16", 3, 16),
        ("M=4 R=16", 4, 16),
    ];

    for (label, m, r) in &configs {
        for &rerank in &[0, 25, 50, 100, 200] {
            let config = RoutedTQConfig {
                dim,
                n_partitions: p,
                n_probe: *r,
                bit_width: 4,
                kmeans_iter: 10,
                seed: 42,
                multi_assign: *m,
                boundary_threshold: None,
                max_assign: 4,
                rerank_top: rerank,
            };

            let idx = RoutedTurboQuantIndex::build(&vectors, config);

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
            let vs_flat = avg_recall / flat_recall;
            let sf = idx.storage_factor();

            println!("{:<20} {:<8} {:<12.3} {:<12.1}% {:<12.3} {:<12.1}",
                     label, rerank, avg_recall, vs_flat * 100.0, latency, sf);
        }
        println!();
    }

    // Best config: boundary-aware + reranking
    println!("\n=== Boundary-aware + Reranking ===");
    println!("{:<25} {:<8} {:<12} {:<12} {:<12} {:<12}",
             "Config", "Rerank", "Recall", "vs Flat", "Latency_ms", "StorageX");
    println!("{}", "-".repeat(81));

    for &threshold in &[0.05, 0.07, 0.10] {
        for &rerank in &[0, 50, 100] {
            let config = RoutedTQConfig {
                dim,
                n_partitions: p,
                n_probe: 12,
                bit_width: 4,
                kmeans_iter: 10,
                seed: 42,
                multi_assign: 1,
                boundary_threshold: Some(threshold),
                max_assign: 4,
                rerank_top: rerank,
            };

            let idx = RoutedTurboQuantIndex::build(&vectors, config);

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
            let vs_flat = avg_recall / flat_recall;
            let sf = idx.storage_factor();

            let label = format!("Boundary t={:.2} R=12", threshold);
            println!("{:<25} {:<8} {:<12.3} {:<12.1}% {:<12.3} {:<12.2}",
                     label, rerank, avg_recall, vs_flat * 100.0, latency, sf);
        }
        println!();
    }
}
