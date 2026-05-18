//! Diagnostic: multi-assignment impact on partition hit rate and recall.
//!
//! Tests M=1,2,3 across various P and R combinations.
//! Shows how multi-assignment directly increases partition hit rate.

extern crate blas_src;

use rand::rngs::StdRng;
use rand::SeedableRng;
use routed_turboquant::index::{RoutedTQConfig, RoutedTurboQuantIndex};
use routed_turboquant::kmeans::kmeans_float;
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
    println!("Generating data...");

    let vectors = random_vectors(n, dim, 42);
    let queries = random_vectors(nq, dim, 123);

    // Ground truth
    println!("Computing exact ground truth...");
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
        let pred: Vec<usize> = flat_results
            .indices_for_query(i)
            .iter()
            .filter(|&&x| x >= 0)
            .map(|&x| x as usize)
            .collect();
        flat_recall_sum += recall_score(&pred, &exact_gt[i], k);
    }
    let flat_recall = flat_recall_sum / nq as f64;
    println!("turbovec flat recall vs exact: {:.3}\n", flat_recall);

    // Compute partition hit rate for each M
    // Need centroids + assignments for hit rate calculation
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
    let km_result = kmeans_float(&unit_vectors, n, dim, p, 10, 42);
    let centroids: Vec<f32> = km_result.centroids.iter().flatten().copied().collect();

    // For each vector, compute its top-M partition assignments
    // (needed for partition hit rate calculation)
    fn get_multi_assignments(
        unit_vectors: &[f32],
        centroids: &[f32],
        n: usize,
        dim: usize,
        p: usize,
        m: usize,
    ) -> Vec<Vec<u32>> {
        let mut assignments = Vec::with_capacity(n);
        for i in 0..n {
            let v = &unit_vectors[i * dim..(i + 1) * dim];
            let mut scores: Vec<(f32, u32)> = (0..p)
                .map(|pid| {
                    let c = &centroids[pid * dim..(pid + 1) * dim];
                    let sim: f32 = v.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
                    (sim, pid as u32)
                })
                .collect();
            scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            assignments.push(scores.iter().take(m).map(|(_, pid)| *pid).collect());
        }
        assignments
    }

    println!("=== Multi-Assignment Impact ===");
    println!(
        "{:<6} {:<6} {:<10} {:<15} {:<15} {:<12} {:<12}",
        "M", "R", "Scan%", "PartHit@10", "Recall", "Latency_ms", "StorageX"
    );

    for &m in &[1, 2, 3, 4] {
        // Compute multi-assignments for hit rate
        let multi_asgn = get_multi_assignments(&unit_vectors, &centroids, n, dim, p, m);

        for &r in &[4, 8, 12, 16, 32] {
            // Build routed index with this M
            let config = RoutedTQConfig {
                dim,
                n_partitions: p,
                n_probe: r,
                bit_width: 4,
                kmeans_iter: 10,
                seed: 42,
                multi_assign: m,
                boundary_threshold: None,
                max_assign: 4,
                rerank_top: 0,
            };

            let routed_idx = RoutedTurboQuantIndex::build(&vectors, config);

            // Compute partition hit rate
            let mut part_hit_sum = 0.0;
            for i in 0..nq {
                let q = &queries[i * dim..(i + 1) * dim];
                let q_norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
                let q_unit: Vec<f32> = q.iter().map(|x| x / q_norm).collect();

                // Route query to top-R partitions
                let mut centroid_scores: Vec<(f32, usize)> = (0..p)
                    .map(|pid| {
                        let c = &centroids[pid * dim..(pid + 1) * dim];
                        let sim: f32 = q_unit.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
                        (sim, pid)
                    })
                    .collect();
                centroid_scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                let selected: HashSet<u32> = centroid_scores
                    .iter()
                    .take(r)
                    .map(|(_, pid)| *pid as u32)
                    .collect();

                // How many of exact top-10 have at least one assignment in selected partitions?
                let mut hits = 0;
                for &gt_id in exact_gt[i].iter().take(k) {
                    let vec_partitions = &multi_asgn[gt_id];
                    if vec_partitions.iter().any(|pid| selected.contains(pid)) {
                        hits += 1;
                    }
                }
                part_hit_sum += hits as f64 / k as f64;
            }
            let avg_part_hit = part_hit_sum / nq as f64;

            // Compute actual recall
            let start = Instant::now();
            let mut recall_sum = 0.0;
            for i in 0..nq {
                let q = &queries[i * dim..(i + 1) * dim];
                let (_, routed_ids) = routed_idx.search(q, k);
                recall_sum += recall_score(&routed_ids, &exact_gt[i], k);
            }
            let elapsed = start.elapsed();
            let latency_ms = elapsed.as_secs_f64() * 1000.0 / nq as f64;
            let avg_recall = recall_sum / nq as f64;
            let scan_pct = (r as f64 / p as f64) * 100.0;

            println!(
                "{:<6} {:<6} {:<10.1} {:<15.3} {:<15.3} {:<12.3} {:<12}",
                m,
                r,
                scan_pct,
                avg_part_hit,
                avg_recall,
                latency_ms,
                format!("{}x", m)
            );
        }
        println!();
    }

    // Summary comparison
    println!("\n=== Best configs vs turbovec flat ===");
    println!("turbovec flat: recall={:.3}", flat_recall);
    println!();

    // Find configs where recall >= 0.8 * flat_recall
    let target = flat_recall * 0.95;
    println!("Target: {:.3} (95% of flat recall)", target);
    println!(
        "{:<6} {:<6} {:<10} {:<15} {:<12}",
        "M", "R", "Scan%", "Recall", "Latency_ms"
    );

    for &m in &[1, 2, 3, 4] {
        for &r in &[4, 8, 12, 16, 32] {
            let config = RoutedTQConfig {
                dim,
                n_partitions: p,
                n_probe: r,
                bit_width: 4,
                kmeans_iter: 10,
                seed: 42,
                multi_assign: m,
                boundary_threshold: None,
                max_assign: 4,
                rerank_top: 0,
            };
            let routed_idx = RoutedTurboQuantIndex::build(&vectors, config);

            let start = Instant::now();
            let mut recall_sum = 0.0;
            for i in 0..nq {
                let q = &queries[i * dim..(i + 1) * dim];
                let (_, routed_ids) = routed_idx.search(q, k);
                recall_sum += recall_score(&routed_ids, &exact_gt[i], k);
            }
            let elapsed = start.elapsed();
            let latency_ms = elapsed.as_secs_f64() * 1000.0 / nq as f64;
            let avg_recall = recall_sum / nq as f64;

            if avg_recall >= target {
                let scan_pct = (r as f64 / p as f64) * 100.0;
                println!(
                    "{:<6} {:<6} {:<10.1} {:<15.3} {:<12.3}",
                    m, r, scan_pct, avg_recall, latency_ms
                );
            }
        }
    }
}
