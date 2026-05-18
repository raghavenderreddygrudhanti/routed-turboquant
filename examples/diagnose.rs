//! Diagnostic benchmark: identify whether the problem is routing or scoring.
//!
//! Tests:
//! 1. Full-probe (R=P) — must match turbovec flat recall
//! 2. Partition Hit@10 — are true neighbors in the routed partitions?
//! 3. Recall curve across R values

extern crate blas_src;

use rand::rngs::StdRng;
use rand::SeedableRng;
use routed_turboquant::index::{RoutedTQConfig, RoutedTurboQuantIndex};
use std::collections::HashSet;
use std::time::Instant;
use turbovec::TurboQuantIndex;

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vecs = vec![0.0f32; n * dim];
    for i in 0..n {
        for d in 0..dim {
            vecs[i * dim + d] = rand::Rng::gen_range(&mut rng, -1.0..1.0);
        }
        let norm: f32 = vecs[i * dim..(i + 1) * dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        for d in 0..dim {
            vecs[i * dim + d] /= norm;
        }
    }
    vecs
}

/// Exact brute-force top-k by float inner product.
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

/// Get turbovec flat top-k.
fn turbovec_flat_topk(index: &TurboQuantIndex, query: &[f32], k: usize) -> Vec<usize> {
    let results = index.search(query, k);
    let indices = results.indices_for_query(0);
    indices
        .iter()
        .filter(|&&x| x >= 0)
        .map(|&x| x as usize)
        .collect()
}

/// Compute recall between two sets.
fn recall(predicted: &[usize], ground_truth: &[usize], k: usize) -> f64 {
    let pred_set: HashSet<usize> = predicted.iter().take(k).copied().collect();
    let gt_set: HashSet<usize> = ground_truth.iter().take(k).copied().collect();
    pred_set.intersection(&gt_set).count() as f64 / k as f64
}

fn main() {
    let dim = 384;
    let k = 10;
    let nq = 100;
    let n = 10_000;

    println!("Generating {} vectors, dim={}, {} queries", n, dim, nq);
    let vectors = random_vectors(n, dim, 42);
    let queries = random_vectors(nq, dim, 123);

    // Build turbovec flat
    println!("Building turbovec flat index...");
    let mut flat_idx = TurboQuantIndex::new(dim, 4);
    flat_idx.add(&vectors);
    flat_idx.prepare();

    // Compute ground truths
    println!("Computing exact ground truth...");
    let exact_gt: Vec<Vec<usize>> = (0..nq)
        .map(|i| exact_topk(&vectors, &queries[i * dim..(i + 1) * dim], dim, k))
        .collect();

    let tv_gt: Vec<Vec<usize>> = (0..nq)
        .map(|i| turbovec_flat_topk(&flat_idx, &queries[i * dim..(i + 1) * dim], k))
        .collect();

    // Turbovec flat recall vs exact
    let mut tv_recall_sum = 0.0;
    for i in 0..nq {
        tv_recall_sum += recall(&tv_gt[i], &exact_gt[i], k);
    }
    let tv_recall = tv_recall_sum / nq as f64;
    println!("\nturbovec flat recall vs exact: {:.3}", tv_recall);

    // =========================================================
    // TEST 1: Full-probe correctness check
    // =========================================================
    println!("\n=== TEST 1: Full-probe (R=P) correctness ===");
    println!(
        "{:<12} {:<12} {:<18} {:<18}",
        "P", "R=P", "recall vs exact", "recall vs tv_flat"
    );

    for &p in &[8, 16, 32, 64] {
        let config = RoutedTQConfig {
            dim,
            n_partitions: p,
            n_probe: p, // FULL PROBE
            bit_width: 4,
            kmeans_iter: 10,
            seed: 42,
            multi_assign: 1,
            boundary_threshold: None,
            max_assign: 4,
            rerank_top: 0,
        };

        let routed_idx = RoutedTurboQuantIndex::build(&vectors, config);

        let mut recall_vs_exact_sum = 0.0;
        let mut recall_vs_tv_sum = 0.0;
        for i in 0..nq {
            let q = &queries[i * dim..(i + 1) * dim];
            let (_, routed_ids) = routed_idx.search(q, k);
            recall_vs_exact_sum += recall(&routed_ids, &exact_gt[i], k);
            recall_vs_tv_sum += recall(&routed_ids, &tv_gt[i], k);
        }
        let r_exact = recall_vs_exact_sum / nq as f64;
        let r_tv = recall_vs_tv_sum / nq as f64;

        println!("{:<12} {:<12} {:<18.3} {:<18.3}", p, p, r_exact, r_tv);
    }

    // =========================================================
    // TEST 2: Partition Hit@10
    // =========================================================
    println!("\n=== TEST 2: Partition Hit@10 (are true neighbors in routed partitions?) ===");
    println!(
        "{:<8} {:<8} {:<10} {:<15} {:<15}",
        "P", "R", "Scan%", "PartHit@10", "Final Recall"
    );

    for &p in &[32, 64] {
        // Build index to get partition assignments
        // We need to know which partition each vector belongs to.
        let mut unit_vectors = vectors.clone();
        for i in 0..n {
            let start = i * dim;
            let end = start + dim;
            let slice = &mut unit_vectors[start..end];
            let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for x in slice.iter_mut() {
                    *x /= norm;
                }
            }
        }
        let km_result = routed_turboquant::kmeans::kmeans_float(&unit_vectors, n, dim, p, 10, 42);
        let assignments = &km_result.assignments;

        // Get centroids (flattened)
        let centroids: Vec<f32> = km_result.centroids.iter().flatten().copied().collect();

        for &r in &[4, 8, 16, 32, 64] {
            if r > p {
                continue;
            }

            // Build routed index with this P and R
            let config_r = RoutedTQConfig {
                dim,
                n_partitions: p,
                n_probe: r,
                bit_width: 4,
                kmeans_iter: 10,
                seed: 42,
                multi_assign: 1,
                boundary_threshold: None,
                max_assign: 4,
                rerank_top: 0,
            };
            let routed_r = RoutedTurboQuantIndex::build(&vectors, config_r);

            let mut part_hit_sum = 0.0;
            let mut recall_sum = 0.0;

            for i in 0..nq {
                let q = &queries[i * dim..(i + 1) * dim];

                // Normalize query
                let q_norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
                let q_unit: Vec<f32> = q.iter().map(|x| x / q_norm).collect();

                // Route: find top-R partitions
                let mut centroid_scores: Vec<(f32, usize)> = (0..p)
                    .map(|pid| {
                        let c = &centroids[pid * dim..(pid + 1) * dim];
                        let sim: f32 = q_unit.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
                        (sim, pid)
                    })
                    .collect();
                centroid_scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                let selected_partitions: HashSet<u32> = centroid_scores
                    .iter()
                    .take(r)
                    .map(|(_, pid)| *pid as u32)
                    .collect();

                // Partition Hit@10: how many of exact top-10 are in selected partitions?
                let mut hits = 0;
                for &gt_id in exact_gt[i].iter().take(k) {
                    let vec_partition = assignments[gt_id];
                    if selected_partitions.contains(&vec_partition) {
                        hits += 1;
                    }
                }
                part_hit_sum += hits as f64 / k as f64;

                // Actual routed recall
                let (_, routed_ids) = routed_r.search(q, k);
                recall_sum += recall(&routed_ids, &exact_gt[i], k);
            }

            let avg_part_hit = part_hit_sum / nq as f64;
            let avg_recall = recall_sum / nq as f64;
            let scan_pct = (r as f64 / p as f64) * 100.0;

            println!(
                "{:<8} {:<8} {:<10.1} {:<15.3} {:<15.3}",
                p, r, scan_pct, avg_part_hit, avg_recall
            );
        }
    }

    // =========================================================
    // TEST 3: Recall curve for P=32
    // =========================================================
    println!("\n=== TEST 3: Recall curve P=32 ===");
    println!(
        "{:<8} {:<10} {:<18} {:<18} {:<12}",
        "R", "Scan%", "recall vs exact", "recall vs tv_flat", "latency_ms"
    );

    for &r in &[1, 2, 4, 8, 16, 32] {
        let config = RoutedTQConfig {
            dim,
            n_partitions: 32,
            n_probe: r,
            bit_width: 4,
            kmeans_iter: 10,
            seed: 42,
            multi_assign: 1,
            boundary_threshold: None,
            max_assign: 4,
            rerank_top: 0,
        };

        let routed_idx = RoutedTurboQuantIndex::build(&vectors, config);

        let start = Instant::now();
        let mut recall_vs_exact_sum = 0.0;
        let mut recall_vs_tv_sum = 0.0;
        for i in 0..nq {
            let q = &queries[i * dim..(i + 1) * dim];
            let (_, routed_ids) = routed_idx.search(q, k);
            recall_vs_exact_sum += recall(&routed_ids, &exact_gt[i], k);
            recall_vs_tv_sum += recall(&routed_ids, &tv_gt[i], k);
        }
        let elapsed = start.elapsed();
        let latency_ms = elapsed.as_secs_f64() * 1000.0 / nq as f64;

        let r_exact = recall_vs_exact_sum / nq as f64;
        let r_tv = recall_vs_tv_sum / nq as f64;
        let scan_pct = (r as f64 / 32.0) * 100.0;

        println!(
            "{:<8} {:<10.1} {:<18.3} {:<18.3} {:<12.3}",
            r, scan_pct, r_exact, r_tv, latency_ms
        );
    }
}
