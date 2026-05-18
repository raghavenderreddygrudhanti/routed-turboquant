//! Benchmark: QJL residual correction impact on recall.
//!
//! Compares:
//! 1. turbovec flat (TQ_mse only)
//! 2. turbovec flat + QJL correction (TQ_prod)
//! 3. routed + rerank (current best)
//! 4. routed + QJL + rerank (new)

extern crate blas_src;

use rand::SeedableRng;
use routed_turboquant::qjl::QjlCorrection;
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::time::Instant;
use turbovec::TurboQuantIndex;

fn load_npy(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("cannot open npy file");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("cannot read npy file");
    assert_eq!(&buf[0..6], b"\x93NUMPY");
    let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let header = std::str::from_utf8(&buf[10..10 + header_len]).unwrap();
    let shape_start = header.find("'shape': (").unwrap() + 10;
    let shape_end = header[shape_start..].find(')').unwrap() + shape_start;
    let dims: Vec<usize> = header[shape_start..shape_end]
        .split(',')
        .map(|s| s.trim().parse::<usize>().unwrap())
        .collect();
    let (n, dim) = (dims[0], dims[1]);
    let data_start = 10 + header_len;
    let floats: Vec<f32> = buf[data_start..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(floats.len(), n * dim);
    (floats, n, dim)
}

fn exact_topk(vectors: &[f32], query: &[f32], dim: usize, k: usize) -> Vec<usize> {
    let n = vectors.len() / dim;
    let mut scores: Vec<(f32, usize)> = (0..n)
        .map(|i| {
            let v = &vectors[i * dim..(i + 1) * dim];
            (
                v.iter().zip(query.iter()).map(|(a, b)| a * b).sum::<f32>(),
                i,
            )
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
    let k = 10;
    let nq = 200;

    // Try real data first, fall back to random
    let (vectors_full, n_total, dim) = if let Ok(_) = File::open("../data/minilm_100k.npy") {
        println!("Using real embeddings (all-MiniLM-L6-v2, 384d)");
        load_npy("../data/minilm_100k.npy")
    } else {
        println!("Real data not found, using random vectors");
        let dim = 384;
        let n = 10_000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
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
        (vecs, n, dim)
    };

    for &n in &[10_000usize, 50_000] {
        if n > n_total {
            continue;
        }

        println!("\n================================================================");
        println!("  n={}, dim={}, k={}, nq={}", n, dim, k, nq);
        println!("================================================================");

        let vectors = &vectors_full[..n * dim];
        let query_start = n - nq;
        let queries = &vectors_full[query_start * dim..n * dim];

        // Ground truth
        println!("  Computing ground truth...");
        let exact_gt: Vec<Vec<usize>> = (0..nq)
            .map(|i| exact_topk(vectors, &queries[i * dim..(i + 1) * dim], dim, k))
            .collect();

        // Build turbovec flat
        println!("  Building turbovec flat...");
        let mut flat_idx = TurboQuantIndex::new(dim, 4);
        flat_idx.add(vectors);
        flat_idx.prepare();

        // Build QJL correction
        println!("  Building QJL correction (this takes a moment)...");
        let build_start = Instant::now();
        let qjl = QjlCorrection::build(vectors, dim, 4, 42);
        let qjl_build_s = build_start.elapsed().as_secs_f64();
        println!(
            "  QJL build: {:.1}s, memory: {:.1} MB",
            qjl_build_s,
            qjl.memory_bytes() as f64 / 1e6
        );

        // --- Method 1: turbovec flat (TQ_mse) ---
        let flat_results = flat_idx.search(queries, k);
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

        // --- Method 2: turbovec flat + QJL correction (TQ_prod) ---
        // Get top-50 from TQ, apply QJL correction, re-sort, take top-10
        let mut qjl_recall_sum = 0.0;
        let start = Instant::now();
        for i in 0..nq {
            let q = &queries[i * dim..(i + 1) * dim];

            // Get top-50 from TQ
            let results = flat_idx.search(q, 50);
            let tq_scores = results.scores_for_query(0);
            let tq_indices = results.indices_for_query(0);

            // Apply QJL correction to each candidate
            let candidates: Vec<usize> = tq_indices
                .iter()
                .filter(|&&x| x >= 0)
                .map(|&x| x as usize)
                .collect();
            let corrections = qjl.batch_correction(q, &candidates);

            // Corrected scores = TQ score + QJL correction
            let mut corrected: Vec<(f32, usize)> = tq_scores
                .iter()
                .zip(candidates.iter())
                .zip(corrections.iter())
                .map(|((&score, &idx), &corr)| (score + corr, idx))
                .collect();

            corrected.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let pred: Vec<usize> = corrected.iter().take(k).map(|(_, i)| *i).collect();
            qjl_recall_sum += recall_score(&pred, &exact_gt[i], k);
        }
        let qjl_latency = start.elapsed().as_secs_f64() * 1000.0 / nq as f64;
        let qjl_recall = qjl_recall_sum / nq as f64;

        // --- Method 3: turbovec flat + QJL + rerank top-25 with float ---
        let mut qjl_rerank_recall_sum = 0.0;
        let start = Instant::now();
        for i in 0..nq {
            let q = &queries[i * dim..(i + 1) * dim];

            // Get top-50 from TQ
            let results = flat_idx.search(q, 50);
            let tq_scores = results.scores_for_query(0);
            let tq_indices = results.indices_for_query(0);

            let candidates: Vec<usize> = tq_indices
                .iter()
                .filter(|&&x| x >= 0)
                .map(|&x| x as usize)
                .collect();
            let corrections = qjl.batch_correction(q, &candidates);

            // Corrected scores
            let mut corrected: Vec<(f32, usize)> = tq_scores
                .iter()
                .zip(candidates.iter())
                .zip(corrections.iter())
                .map(|((&score, &idx), &corr)| (score + corr, idx))
                .collect();
            corrected.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            // Float rerank top-25
            let top25: Vec<(f32, usize)> = corrected
                .iter()
                .take(25)
                .map(|&(_, idx)| {
                    let v = &vectors[idx * dim..(idx + 1) * dim];
                    let sim: f32 = q.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                    (sim, idx)
                })
                .collect();
            let mut final_results = top25;
            final_results.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let pred: Vec<usize> = final_results.iter().take(k).map(|(_, i)| *i).collect();
            qjl_rerank_recall_sum += recall_score(&pred, &exact_gt[i], k);
        }
        let qjl_rerank_latency = start.elapsed().as_secs_f64() * 1000.0 / nq as f64;
        let qjl_rerank_recall = qjl_rerank_recall_sum / nq as f64;

        // --- Results ---
        println!("\n  {:<45} {:<10} {:<12}", "Method", "Recall@10", "Latency");
        println!("  {}", "-".repeat(67));
        println!(
            "  {:<45} {:<10.3} {:<12}",
            "turbovec flat (TQ_mse)", flat_recall, "baseline"
        );
        println!(
            "  {:<45} {:<10.3} {:<12.3}ms",
            "turbovec flat + QJL (TQ_prod)", qjl_recall, qjl_latency
        );
        println!(
            "  {:<45} {:<10.3} {:<12.3}ms",
            "turbovec flat + QJL + rerank25", qjl_rerank_recall, qjl_rerank_latency
        );
        println!(
            "\n  QJL improvement over flat: {:.3} ({:+.1}%)",
            qjl_recall - flat_recall,
            (qjl_recall - flat_recall) / flat_recall * 100.0
        );
        println!(
            "  QJL+rerank improvement: {:.3} ({:+.1}%)",
            qjl_rerank_recall - flat_recall,
            (qjl_rerank_recall - flat_recall) / flat_recall * 100.0
        );
    }
}
