//! Benchmark with clustered data (simulates real embeddings with manifold structure).
//!
//! Random uniform vectors on a 384-d sphere have no neighborhood structure.
//! Real embeddings (sentence-transformers, OpenAI) cluster into semantic regions.
//! This benchmark generates clustered data to simulate that structure.

extern crate blas_src;

use std::collections::HashSet;
use std::time::Instant;
use rand::rngs::StdRng;
use rand::SeedableRng;
use turbovec::TurboQuantIndex;
use routed_turboquant::index::{RoutedTQConfig, RoutedTurboQuantIndex};

/// Generate clustered vectors: n_clusters cluster centers, each with members nearby.
fn clustered_vectors(n: usize, dim: usize, n_clusters: usize, spread: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vecs = vec![0.0f32; n * dim];

    // Generate cluster centers
    let mut centers = vec![0.0f32; n_clusters * dim];
    for i in 0..n_clusters {
        for d in 0..dim {
            centers[i * dim + d] = rand::Rng::gen_range(&mut rng, -1.0..1.0);
        }
        let norm: f32 = centers[i * dim..(i + 1) * dim].iter().map(|x| x * x).sum::<f32>().sqrt();
        for d in 0..dim {
            centers[i * dim + d] /= norm;
        }
    }

    // Generate vectors around cluster centers
    for i in 0..n {
        let cluster = i % n_clusters;
        for d in 0..dim {
            vecs[i * dim + d] = centers[cluster * dim + d]
                + rand::Rng::gen_range(&mut rng, -spread..spread);
        }
        // Normalize
        let norm: f32 = vecs[i * dim..(i + 1) * dim].iter().map(|x| x * x).sum::<f32>().sqrt();
        for d in 0..dim {
            vecs[i * dim + d] /= norm;
        }
    }

    vecs
}

/// Exact brute-force top-k.
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
    let nq = 200;

    println!("=== Routed TurboQuant vs Flat TurboQuant ===");
    println!("=== Clustered data (simulates real embeddings) ===\n");

    // Test with different cluster counts and spreads
    for &(n, n_clusters, spread, label) in &[
        (10_000usize, 50usize, 0.3f32, "10K, 50 clusters, tight"),
        (10_000, 50, 0.5, "10K, 50 clusters, medium"),
        (50_000, 200, 0.3, "50K, 200 clusters, tight"),
        (50_000, 200, 0.5, "50K, 200 clusters, medium"),
        (100_000, 500, 0.3, "100K, 500 clusters, tight"),
    ] {
        println!("\n======================================================================");
        println!("  {} (n={}, dim={}, nq={})", label, n, dim, nq);
        println!("======================================================================");

        let vectors = clustered_vectors(n, dim, n_clusters, spread, 42);
        // Queries from same distribution
        let queries = clustered_vectors(nq, dim, n_clusters, spread, 123);

        // Ground truth
        let exact_gt: Vec<Vec<usize>> = (0..nq)
            .map(|i| exact_topk(&vectors, &queries[i * dim..(i + 1) * dim], dim, k))
            .collect();

        // Turbovec flat
        let mut flat_idx = TurboQuantIndex::new(dim, 4);
        flat_idx.add(&vectors);
        flat_idx.prepare();

        let start = Instant::now();
        let flat_results = flat_idx.search(&queries, k);
        let flat_elapsed = start.elapsed();
        let flat_latency = flat_elapsed.as_secs_f64() * 1000.0 / nq as f64;
        let flat_qps = nq as f64 / flat_elapsed.as_secs_f64();

        let mut flat_recall_sum = 0.0;
        for i in 0..nq {
            let pred: Vec<usize> = flat_results.indices_for_query(i)
                .iter().filter(|&&x| x >= 0).map(|&x| x as usize).collect();
            flat_recall_sum += recall_score(&pred, &exact_gt[i], k);
        }
        let flat_recall = flat_recall_sum / nq as f64;

        println!("  turbovec flat:     recall={:.3}  latency={:.3}ms  QPS={:.0}",
                 flat_recall, flat_latency, flat_qps);

        // Routed variants
        let p = if n <= 10_000 { 32 } else if n <= 50_000 { 64 } else { 128 };

        for &r in &[4, 8, 16, 32] {
            if r > p {
                continue;
            }

            let config = RoutedTQConfig {
                dim,
                n_partitions: p,
                n_probe: r,
                bit_width: 4,
                kmeans_iter: 15,
                seed: 42, multi_assign: 1, boundary_threshold: None, max_assign: 4, rerank_top: 0,
            };

            let build_start = Instant::now();
            let routed_idx = RoutedTurboQuantIndex::build(&vectors, config);
            let build_time = build_start.elapsed();

            // Timed search
            let start = Instant::now();
            let mut recall_sum = 0.0;
            for i in 0..nq {
                let q = &queries[i * dim..(i + 1) * dim];
                let (_, routed_ids) = routed_idx.search(q, k);
                recall_sum += recall_score(&routed_ids, &exact_gt[i], k);
            }
            let elapsed = start.elapsed();
            let latency = elapsed.as_secs_f64() * 1000.0 / nq as f64;
            let qps = nq as f64 / elapsed.as_secs_f64();
            let avg_recall = recall_sum / nq as f64;
            let scan_pct = (r as f64 / p as f64) * 100.0;
            let speedup = flat_latency / latency;

            println!("  routed P={:<3} R={:<3}: recall={:.3}  latency={:.3}ms  QPS={:.0}  scan={:.1}%  speedup={:.1}x  build={:.2}s",
                     p, r, avg_recall, latency, qps, scan_pct, speedup, build_time.as_secs_f64());
        }
    }
}
