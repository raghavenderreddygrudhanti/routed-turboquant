//! RoutedTurboQuantIndex: IVF-style partitioned search with TurboQuant scoring.
//!
//! Architecture:
//!   1. Float k-means partitions vectors into P clusters
//!   2. Each partition stores vectors in a TurboQuantIndex
//!   3. At query time, route to top-R partitions by centroid similarity
//!   4. Score within those partitions using TurboQuant SIMD kernels
//!   5. Merge results across partitions, return top-k
//!
//! Complexity: O(n/P * R) per query instead of O(n).
//! At P=128, R=8: scans 6.2% of vectors.

use turbovec::TurboQuantIndex;
use rayon::prelude::*;

use crate::kmeans::{kmeans_float, KMeansResult};

/// Configuration for building the routed index.
pub struct RoutedTQConfig {
    /// Vector dimensionality (must be multiple of 8).
    pub dim: usize,
    /// Number of partitions.
    pub n_partitions: usize,
    /// Partitions to probe per query.
    pub n_probe: usize,
    /// TurboQuant bit width (2, 3, or 4).
    pub bit_width: usize,
    /// K-means iterations.
    pub kmeans_iter: usize,
    /// Random seed for k-means.
    pub seed: u64,
    /// Multi-assignment factor: each vector is inserted into its top-M
    /// nearest partitions. M=1 is standard IVF. M=2-3 significantly
    /// improves partition hit rate at the cost of M× storage.
    /// Ignored when `boundary_threshold` is set (adaptive mode).
    pub multi_assign: usize,
    /// Boundary-aware adaptive multi-assignment threshold.
    /// When set to Some(t), vectors are assigned to all partitions
    /// whose centroid similarity is within `t` of the best centroid.
    /// Specifically: assign to partition p if sim(v, c_p) >= best_sim - t.
    /// This gives M=1 for clear cluster members and M=2-4 for boundary vectors.
    /// Typical values: 0.01 to 0.05.
    /// When None, uses fixed `multi_assign`.
    pub boundary_threshold: Option<f32>,
    /// Maximum assignments per vector in boundary-aware mode.
    /// Caps storage even for vectors equidistant to many centroids.
    pub max_assign: usize,
    /// Float reranking: after TQ scoring, take top `rerank_top` candidates
    /// and rescore with exact float inner product against stored vectors.
    /// 0 = no reranking (use TQ scores directly).
    /// Typical values: 50, 100, 200.
    pub rerank_top: usize,
}

impl Default for RoutedTQConfig {
    fn default() -> Self {
        Self {
            dim: 384,
            n_partitions: 128,
            n_probe: 8,
            bit_width: 4,
            kmeans_iter: 10,
            seed: 42,
            multi_assign: 1,
            boundary_threshold: None,
            max_assign: 4,
            rerank_top: 0,
        }
    }
}

/// Per-partition data: TurboQuant index + mapping to global IDs.
struct Partition {
    /// TurboQuant index for this partition's vectors.
    tq_index: TurboQuantIndex,
    /// Global vector IDs for vectors in this partition (ordered by insertion).
    global_ids: Vec<usize>,
}

/// Diagnostic stats from a single search query.
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    /// Total vector entries in probed partitions (actual TQ scan volume).
    pub raw_partition_entries: usize,
    /// Results returned by TQ scoring (capped by collect_k per partition).
    pub tq_results_collected: usize,
    /// Unique database IDs in TQ results (after dedup).
    pub unique_ids: usize,
    /// Candidates sent to float rerank.
    pub rerank_k: usize,
}

/// IVF-routed TurboQuant index.
///
/// Combines float-space k-means routing with per-partition TurboQuant
/// SIMD scoring for sublinear approximate nearest neighbor search.
pub struct RoutedTurboQuantIndex {
    config: RoutedTQConfig,
    /// Unit-normalized centroids (n_partitions, dim), flattened row-major.
    centroids: Vec<f32>,
    /// Per-partition data.
    partitions: Vec<Partition>,
    /// Total vectors indexed.
    n_vectors: usize,
    /// Total vector copies stored (>= n_vectors when multi-assign > 1).
    total_stored: usize,
    /// Original vectors stored for float reranking (n * dim, row-major).
    /// Only populated when rerank_top > 0.
    vectors: Option<Vec<f32>>,
}

impl RoutedTurboQuantIndex {
    /// Build the index from a set of vectors.
    ///
    /// `vectors` is a flat f32 slice of shape (n, dim) in row-major order.
    /// All vectors are normalized internally before clustering.
    ///
    /// With `multi_assign > 1`, each vector is inserted into its top-M
    /// nearest partitions. This increases partition hit rate at the cost
    /// of M× storage and slightly larger partitions to scan.
    pub fn build(vectors: &[f32], config: RoutedTQConfig) -> Self {
        let n = vectors.len() / config.dim;
        assert_eq!(vectors.len(), n * config.dim, "vectors length must be n * dim");
        assert!(n > 0, "need at least one vector");
        assert!(config.multi_assign >= 1, "multi_assign must be >= 1");
        assert!(config.multi_assign <= config.n_partitions,
                "multi_assign must be <= n_partitions");

        // Normalize vectors for clustering
        let mut unit_vectors = vectors.to_vec();
        for i in 0..n {
            let start = i * config.dim;
            let end = start + config.dim;
            let slice = &mut unit_vectors[start..end];
            let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for x in slice.iter_mut() {
                    *x /= norm;
                }
            }
        }

        // Run k-means
        let KMeansResult { centroids, assignments: _ } = kmeans_float(
            &unit_vectors,
            n,
            config.dim,
            config.n_partitions,
            config.kmeans_iter,
            config.seed,
        );

        // Flatten centroids for scoring
        let flat_centroids: Vec<f32> = centroids.iter().flatten().copied().collect();

        // Assignment: either fixed multi-assign or boundary-aware adaptive
        let mut partition_vecs: Vec<Vec<usize>> = vec![Vec::new(); config.n_partitions];

        if let Some(threshold) = config.boundary_threshold {
            // Boundary-aware adaptive multi-assignment:
            // Assign vector to all partitions within `threshold` of best similarity.
            let max_m = config.max_assign.min(config.n_partitions);

            for i in 0..n {
                let v = &unit_vectors[i * config.dim..(i + 1) * config.dim];

                let mut scores: Vec<(f32, usize)> = (0..config.n_partitions)
                    .map(|p| {
                        let c = &flat_centroids[p * config.dim..(p + 1) * config.dim];
                        let sim: f32 = v.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
                        (sim, p)
                    })
                    .collect();

                scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                let best_sim = scores[0].0;
                let cutoff = best_sim - threshold;

                // Always assign to best partition
                partition_vecs[scores[0].1].push(i);

                // Assign to additional partitions within threshold, up to max_m
                for &(sim, pid) in scores.iter().skip(1).take(max_m - 1) {
                    if sim >= cutoff {
                        partition_vecs[pid].push(i);
                    } else {
                        break; // sorted descending, no more will qualify
                    }
                }
            }
        } else {
            // Fixed multi-assignment
            let m = config.multi_assign;

            for i in 0..n {
                let v = &unit_vectors[i * config.dim..(i + 1) * config.dim];

                let mut scores: Vec<(f32, usize)> = (0..config.n_partitions)
                    .map(|p| {
                        let c = &flat_centroids[p * config.dim..(p + 1) * config.dim];
                        let sim: f32 = v.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
                        (sim, p)
                    })
                    .collect();

                scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                for &(_, pid) in scores.iter().take(m) {
                    partition_vecs[pid].push(i);
                }
            }
        }

        // Build per-partition TurboQuant indices
        // Use original (unnormalized) vectors for TQ — it stores norms internally
        let total_stored: usize = partition_vecs.iter().map(|v| v.len()).sum();

        let partitions: Vec<Partition> = partition_vecs
            .into_iter()
            .map(|global_ids| {
                let mut tq_index = TurboQuantIndex::new(config.dim, config.bit_width);
                if !global_ids.is_empty() {
                    // Gather partition vectors into contiguous buffer
                    let mut buf = vec![0.0f32; global_ids.len() * config.dim];
                    for (local_i, &global_i) in global_ids.iter().enumerate() {
                        let src = &vectors[global_i * config.dim..(global_i + 1) * config.dim];
                        buf[local_i * config.dim..(local_i + 1) * config.dim].copy_from_slice(src);
                    }
                    tq_index.add(&buf);
                    tq_index.prepare();
                }
                Partition { tq_index, global_ids }
            })
            .collect();

        // Store original vectors if reranking is enabled
        let stored_vectors = if config.rerank_top > 0 {
            Some(vectors.to_vec())
        } else {
            None
        };

        Self {
            config,
            centroids: flat_centroids,
            partitions,
            n_vectors: n,
            total_stored,
            vectors: stored_vectors,
        }
    }

    /// Search for the k nearest neighbors of a single query.
    ///
    /// Returns (scores, global_indices) sorted by descending score.
    /// When rerank_top > 0, collects extra candidates from TQ scoring
    /// then rescores them with exact float inner product.
    pub fn search(&self, query: &[f32], k: usize) -> (Vec<f32>, Vec<usize>) {
        assert_eq!(query.len(), self.config.dim);

        if self.n_vectors == 0 {
            return (vec![], vec![]);
        }

        // Normalize query for routing
        let dim = self.config.dim;
        let mut query_unit = query.to_vec();
        let norm: f32 = query_unit.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in query_unit.iter_mut() {
                *x /= norm;
            }
        }

        // Route: compute similarity to all centroids, select top-R
        let n_partitions = self.config.n_partitions;
        let n_probe = self.config.n_probe.min(n_partitions);

        // Partial sort: only need top n_probe, not full sort
        let mut centroid_scores: Vec<(f32, usize)> = Vec::with_capacity(n_partitions);
        for p in 0..n_partitions {
            let c = &self.centroids[p * dim..(p + 1) * dim];
            let sim: f32 = query_unit.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
            centroid_scores.push((sim, p));
        }
        centroid_scores.select_nth_unstable_by(n_probe - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());

        // How many candidates to collect from TQ scoring
        let collect_k = if self.config.rerank_top > 0 {
            self.config.rerank_top
        } else {
            k
        };

        // Score within probed partitions
        let mut all_scores: Vec<f32> = Vec::new();
        let mut all_global_ids: Vec<usize> = Vec::new();

        for &(_, pid) in centroid_scores[..n_probe].iter() {
            let partition = &self.partitions[pid];
            if partition.global_ids.is_empty() {
                continue;
            }

            let local_k = collect_k.min(partition.global_ids.len());
            let results = partition.tq_index.search(query, local_k);

            let scores = results.scores_for_query(0);
            let local_ids = results.indices_for_query(0);

            for (&score, &local_id) in scores.iter().zip(local_ids.iter()) {
                if local_id < 0 {
                    continue;
                }
                let global_id = partition.global_ids[local_id as usize];
                all_scores.push(score);
                all_global_ids.push(global_id);
            }
        }

        if all_scores.is_empty() {
            return (vec![], vec![]);
        }

        // Deduplicate using a fast visited-vector (O(1) lookup by global ID)
        // Much faster than HashSet for dense integer IDs
        let total_candidates = all_scores.len();
        let mut combined: Vec<(f32, usize)> = all_scores
            .into_iter()
            .zip(all_global_ids.into_iter())
            .collect();
        combined.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let target = if self.config.rerank_top > 0 { self.config.rerank_top } else { k };
        let mut deduped: Vec<(f32, usize)> = Vec::with_capacity(target);

        if self.config.multi_assign > 1 || self.config.boundary_threshold.is_some() {
            // Multi-assignment: need dedup. Use vec<bool> for fast lookup.
            let mut visited = vec![false; self.n_vectors];
            for (score, gid) in combined {
                if !visited[gid] {
                    visited[gid] = true;
                    deduped.push((score, gid));
                    if deduped.len() == target {
                        break;
                    }
                }
            }
        } else {
            // Single assignment: no duplicates possible, just truncate
            deduped.extend(combined.into_iter().take(target));
        }

        let unique_count = deduped.len();
        let _ = (total_candidates, unique_count); // available for diagnostics

        // Float reranking
        if self.config.rerank_top > 0 {
            if let Some(ref vectors) = self.vectors {
                let mut reranked: Vec<(f32, usize)> = deduped.iter()
                    .map(|&(_, gid)| {
                        let v = &vectors[gid * dim..(gid + 1) * dim];
                        let sim: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                        (sim, gid)
                    })
                    .collect();
                reranked.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                reranked.truncate(k);

                let scores: Vec<f32> = reranked.iter().map(|(s, _)| *s).collect();
                let indices: Vec<usize> = reranked.iter().map(|(_, i)| *i).collect();
                return (scores, indices);
            }
        }

        deduped.truncate(k);
        let scores: Vec<f32> = deduped.iter().map(|(s, _)| *s).collect();
        let indices: Vec<usize> = deduped.iter().map(|(_, i)| *i).collect();
        (scores, indices)
    }

    /// Batch search: multiple queries at once.
    ///
    /// `queries` is flat f32 of shape (nq, dim).
    /// Returns (scores, indices) each of shape (nq, k).
    pub fn search_batch(&self, queries: &[f32], k: usize) -> (Vec<Vec<f32>>, Vec<Vec<usize>>) {
        let nq = queries.len() / self.config.dim;
        assert_eq!(queries.len(), nq * self.config.dim);

        let results: Vec<(Vec<f32>, Vec<usize>)> = (0..nq)
            .into_par_iter()
            .map(|i| {
                let q = &queries[i * self.config.dim..(i + 1) * self.config.dim];
                self.search(q, k)
            })
            .collect();

        let scores: Vec<Vec<f32>> = results.iter().map(|(s, _)| s.clone()).collect();
        let indices: Vec<Vec<usize>> = results.iter().map(|(_, i)| i.clone()).collect();

        (scores, indices)
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.n_vectors
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }

    /// Search with diagnostics: returns results + detailed stats.
    ///
    /// Stats struct:
    /// - raw_partition_entries: total vectors in probed partitions (actual scan volume)
    /// - tq_results_collected: results returned by TQ scoring (capped by collect_k per partition)
    /// - unique_ids: deduplicated candidate count
    /// - rerank_k: candidates sent to float rerank
    pub fn search_stats(&self, query: &[f32], k: usize) -> ((Vec<f32>, Vec<usize>), SearchStats) {
        assert_eq!(query.len(), self.config.dim);

        if self.n_vectors == 0 {
            return ((vec![], vec![]), SearchStats::default());
        }

        let dim = self.config.dim;
        let mut query_unit = query.to_vec();
        let norm: f32 = query_unit.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in query_unit.iter_mut() {
                *x /= norm;
            }
        }

        let n_partitions = self.config.n_partitions;
        let n_probe = self.config.n_probe.min(n_partitions);

        let mut centroid_scores: Vec<(f32, usize)> = Vec::with_capacity(n_partitions);
        for p in 0..n_partitions {
            let c = &self.centroids[p * dim..(p + 1) * dim];
            let sim: f32 = query_unit.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
            centroid_scores.push((sim, p));
        }
        centroid_scores.select_nth_unstable_by(n_probe - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());

        let collect_k = if self.config.rerank_top > 0 { self.config.rerank_top } else { k };

        // Count raw partition entries (actual scan volume)
        let mut raw_partition_entries: usize = 0;
        let mut all_global_ids: Vec<usize> = Vec::new();
        let mut all_scores: Vec<f32> = Vec::new();

        for &(_, pid) in centroid_scores[..n_probe].iter() {
            let partition = &self.partitions[pid];
            if partition.global_ids.is_empty() { continue; }

            // Raw entries = total vectors in this partition (what TQ actually scans)
            raw_partition_entries += partition.global_ids.len();

            let local_k = collect_k.min(partition.global_ids.len());
            let results = partition.tq_index.search(query, local_k);
            let scores = results.scores_for_query(0);
            let local_ids = results.indices_for_query(0);
            for (&score, &local_id) in scores.iter().zip(local_ids.iter()) {
                if local_id < 0 { continue; }
                all_global_ids.push(partition.global_ids[local_id as usize]);
                all_scores.push(score);
            }
        }

        let tq_results_collected = all_global_ids.len();

        if tq_results_collected == 0 {
            return ((vec![], vec![]), SearchStats {
                raw_partition_entries,
                tq_results_collected: 0,
                unique_ids: 0,
                rerank_k: 0,
            });
        }

        let mut combined: Vec<(f32, usize)> = all_scores.into_iter().zip(all_global_ids.into_iter()).collect();
        combined.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let target = if self.config.rerank_top > 0 { self.config.rerank_top } else { k };
        let mut deduped: Vec<(f32, usize)> = Vec::with_capacity(target);
        let mut visited = vec![false; self.n_vectors];
        let mut total_unique_in_tq = 0usize;
        for (score, gid) in &combined {
            if !visited[*gid] {
                visited[*gid] = true;
                total_unique_in_tq += 1;
                if deduped.len() < target {
                    deduped.push((*score, *gid));
                }
            }
        }

        let rerank_k = deduped.len();

        // Rerank
        if self.config.rerank_top > 0 {
            if let Some(ref vectors) = self.vectors {
                let mut reranked: Vec<(f32, usize)> = deduped.iter()
                    .map(|&(_, gid)| {
                        let v = &vectors[gid * dim..(gid + 1) * dim];
                        let sim: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                        (sim, gid)
                    })
                    .collect();
                reranked.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                reranked.truncate(k);
                let scores: Vec<f32> = reranked.iter().map(|(s, _)| *s).collect();
                let indices: Vec<usize> = reranked.iter().map(|(_, i)| *i).collect();
                return ((scores, indices), SearchStats {
                    raw_partition_entries,
                    tq_results_collected,
                    unique_ids: total_unique_in_tq,
                    rerank_k,
                });
            }
        }

        deduped.truncate(k);
        let scores: Vec<f32> = deduped.iter().map(|(s, _)| *s).collect();
        let indices: Vec<usize> = deduped.iter().map(|(_, i)| *i).collect();
        ((scores, indices), SearchStats {
            raw_partition_entries,
            tq_results_collected,
            unique_ids: total_unique_in_tq,
            rerank_k,
        })
    }

    /// Approximate memory usage in bytes.
    /// With multi_assign=M, TQ codes are stored M× (once per assigned partition).
    pub fn memory_bytes(&self) -> usize {
        // TQ codes: total_stored * dim * bit_width / 8
        let codes = self.total_stored * self.config.dim * self.config.bit_width / 8;
        // Norms: total_stored * 4
        let norms = self.total_stored * 4;
        // Centroids: P * dim * 4
        let centroids = self.config.n_partitions * self.config.dim * 4;
        // Global ID mappings: total_stored * 8
        let id_maps = self.total_stored * 8;
        codes + norms + centroids + id_maps
    }

    /// Average storage multiplier (total copies / unique vectors).
    /// 1.0 for standard IVF, higher for multi-assignment.
    pub fn storage_factor(&self) -> f64 {
        if self.n_vectors == 0 {
            return 1.0;
        }
        self.total_stored as f64 / self.n_vectors as f64
    }

    /// Total vector copies stored across all partitions.
    pub fn total_stored(&self) -> usize {
        self.total_stored
    }

    /// Vectors scanned per query (approximate).
    pub fn vectors_scanned_per_query(&self) -> usize {
        if self.n_vectors == 0 {
            return 0;
        }
        self.n_vectors * self.config.n_probe / self.config.n_partitions
    }

    /// Scan percentage per query.
    pub fn scan_percentage(&self) -> f64 {
        if self.n_vectors == 0 {
            return 0.0;
        }
        (self.config.n_probe as f64 / self.config.n_partitions as f64) * 100.0
    }

    /// Get config reference.
    pub fn config(&self) -> &RoutedTQConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut vecs = vec![0.0f32; n * dim];
        for i in 0..n {
            for d in 0..dim {
                vecs[i * dim + d] = rand::Rng::gen_range(&mut rng, -1.0..1.0);
            }
            // Normalize
            let norm: f32 = vecs[i * dim..(i + 1) * dim].iter().map(|x| x * x).sum::<f32>().sqrt();
            for d in 0..dim {
                vecs[i * dim + d] /= norm;
            }
        }
        vecs
    }

    #[test]
    fn test_build_and_search() {
        let dim = 32;
        let n = 1000;
        let k = 10;

        let vectors = random_vectors(n, dim, 42);
        let config = RoutedTQConfig {
            dim,
            n_partitions: 8,
            n_probe: 4,
            bit_width: 4,
            kmeans_iter: 5,
            seed: 42,
            multi_assign: 1, boundary_threshold: None, max_assign: 4, rerank_top: 0,
        };

        let index = RoutedTurboQuantIndex::build(&vectors, config);
        assert_eq!(index.len(), n);

        let query = &vectors[0..dim];
        let (scores, indices) = index.search(query, k);

        assert!(scores.len() <= k);
        assert_eq!(scores.len(), indices.len());
        // Scores should be descending
        for w in scores.windows(2) {
            assert!(w[0] >= w[1]);
        }
        // All indices valid
        for &idx in &indices {
            assert!(idx < n);
        }
    }

    #[test]
    fn test_self_retrieval() {
        // Query with a vector that's in the index — should find itself
        let dim = 64;
        let n = 500;
        let k = 1;

        let vectors = random_vectors(n, dim, 99);
        let config = RoutedTQConfig {
            dim,
            n_partitions: 8,
            n_probe: 8, // probe all partitions
            bit_width: 4,
            kmeans_iter: 10,
            seed: 42,
            multi_assign: 1, boundary_threshold: None, max_assign: 4, rerank_top: 0,
        };

        let index = RoutedTurboQuantIndex::build(&vectors, config);
        let query = &vectors[0..dim];
        let (_, indices) = index.search(query, k);

        // With all partitions probed, should find itself as top-1
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_batch_search() {
        let dim = 32;
        let n = 500;
        let nq = 10;
        let k = 5;

        let vectors = random_vectors(n, dim, 42);
        let queries = random_vectors(nq, dim, 123);

        let config = RoutedTQConfig {
            dim,
            n_partitions: 4,
            n_probe: 2,
            bit_width: 4,
            kmeans_iter: 5,
            seed: 42,
            multi_assign: 1, boundary_threshold: None, max_assign: 4, rerank_top: 0,
        };

        let index = RoutedTurboQuantIndex::build(&vectors, config);
        let (scores, indices) = index.search_batch(&queries, k);

        assert_eq!(scores.len(), nq);
        assert_eq!(indices.len(), nq);
        for i in 0..nq {
            assert!(scores[i].len() <= k);
            assert_eq!(scores[i].len(), indices[i].len());
        }
    }

    #[test]
    fn test_scan_percentage() {
        let dim = 16;
        let n = 200;

        let vectors = random_vectors(n, dim, 42);
        let config = RoutedTQConfig {
            dim,
            n_partitions: 16,
            n_probe: 4,
            bit_width: 4,
            kmeans_iter: 3,
            seed: 42,
            multi_assign: 1, boundary_threshold: None, max_assign: 4, rerank_top: 0,
        };

        let index = RoutedTurboQuantIndex::build(&vectors, config);
        let pct = index.scan_percentage();
        assert!((pct - 25.0).abs() < 0.01); // 4/16 = 25%
    }

    #[test]
    fn test_memory_estimate() {
        let dim = 384;

        let vectors = random_vectors(100, dim, 42); // small for test speed
        let config = RoutedTQConfig {
            dim,
            n_partitions: 4,
            n_probe: 2,
            bit_width: 4,
            kmeans_iter: 2,
            seed: 42,
            multi_assign: 1, boundary_threshold: None, max_assign: 4, rerank_top: 0,
        };

        let index = RoutedTurboQuantIndex::build(&vectors, config);
        let mem = index.memory_bytes();
        // 100 vectors * 384 * 4/8 = 19200 (codes)
        // + 100 * 4 = 400 (norms)
        // + 4 * 384 * 4 = 6144 (centroids)
        // + 100 * 8 = 800 (id maps)
        assert!(mem > 0);
    }

    #[test]
    fn test_multi_assign_improves_recall() {
        // Multi-assignment should improve recall at same n_probe
        let dim = 32;
        let n = 1000;
        let k = 10;

        let vectors = random_vectors(n, dim, 42);
        let query = &vectors[0..dim];

        // M=1
        let config1 = RoutedTQConfig {
            dim,
            n_partitions: 8,
            n_probe: 2,
            bit_width: 4,
            kmeans_iter: 10,
            seed: 42,
            multi_assign: 1, boundary_threshold: None, max_assign: 4, rerank_top: 0,
        };
        let idx1 = RoutedTurboQuantIndex::build(&vectors, config1);
        let (_, ids1) = idx1.search(query, k);

        // M=3
        let config3 = RoutedTQConfig {
            dim,
            n_partitions: 8,
            n_probe: 2,
            bit_width: 4,
            kmeans_iter: 10,
            seed: 42,
            multi_assign: 3, boundary_threshold: None, max_assign: 4, rerank_top: 0,
        };
        let idx3 = RoutedTurboQuantIndex::build(&vectors, config3);
        let (_, ids3) = idx3.search(query, k);

        // Full probe baseline
        let config_full = RoutedTQConfig {
            dim,
            n_partitions: 8,
            n_probe: 8,
            bit_width: 4,
            kmeans_iter: 10,
            seed: 42,
            multi_assign: 1, boundary_threshold: None, max_assign: 4, rerank_top: 0,
        };
        let idx_full = RoutedTurboQuantIndex::build(&vectors, config_full);
        let (_, ids_full) = idx_full.search(query, k);

        // Compute recall vs full-probe
        let gt_set: std::collections::HashSet<usize> = ids_full.iter().copied().collect();
        let recall1 = ids1.iter().filter(|id| gt_set.contains(id)).count() as f64 / k as f64;
        let recall3 = ids3.iter().filter(|id| gt_set.contains(id)).count() as f64 / k as f64;

        // M=3 should have >= recall of M=1 (more vectors in probed partitions)
        assert!(recall3 >= recall1,
                "multi_assign=3 recall ({}) should be >= multi_assign=1 recall ({})",
                recall3, recall1);
    }

    #[test]
    fn test_multi_assign_deduplication() {
        // With multi-assignment, same vector appears in multiple partitions.
        // Search results should not contain duplicates.
        let dim = 32;
        let n = 200;
        let k = 10;

        let vectors = random_vectors(n, dim, 42);
        let config = RoutedTQConfig {
            dim,
            n_partitions: 4,
            n_probe: 4, // probe all
            bit_width: 4,
            kmeans_iter: 5,
            seed: 42,
            multi_assign: 3, boundary_threshold: None, max_assign: 4, rerank_top: 0,
        };

        let index = RoutedTurboQuantIndex::build(&vectors, config);
        let query = &vectors[0..dim];
        let (_, indices) = index.search(query, k);

        // No duplicates
        let unique: std::collections::HashSet<usize> = indices.iter().copied().collect();
        assert_eq!(unique.len(), indices.len(), "search results contain duplicates");
    }

    #[test]
    fn test_boundary_aware_assignment() {
        // Boundary-aware should use less storage than fixed M=4
        // but more than M=1
        let dim = 32;
        let n = 500;

        let vectors = random_vectors(n, dim, 42);

        // Fixed M=1
        let config1 = RoutedTQConfig {
            dim,
            n_partitions: 8,
            n_probe: 4,
            bit_width: 4,
            kmeans_iter: 10,
            seed: 42,
            multi_assign: 1,
            boundary_threshold: None,
            max_assign: 4, rerank_top: 0,
        };
        let idx1 = RoutedTurboQuantIndex::build(&vectors, config1);

        // Fixed M=4
        let config4 = RoutedTQConfig {
            dim,
            n_partitions: 8,
            n_probe: 4,
            bit_width: 4,
            kmeans_iter: 10,
            seed: 42,
            multi_assign: 4,
            boundary_threshold: None,
            max_assign: 4, rerank_top: 0,
        };
        let idx4 = RoutedTurboQuantIndex::build(&vectors, config4);

        // Boundary-aware with threshold
        let config_ba = RoutedTQConfig {
            dim,
            n_partitions: 8,
            n_probe: 4,
            bit_width: 4,
            kmeans_iter: 10,
            seed: 42,
            multi_assign: 1, // ignored in boundary mode
            boundary_threshold: Some(0.03),
            max_assign: 4, rerank_top: 0,
        };
        let idx_ba = RoutedTurboQuantIndex::build(&vectors, config_ba);

        // Boundary-aware storage should be between M=1 and M=4
        let sf1 = idx1.storage_factor();
        let sf4 = idx4.storage_factor();
        let sf_ba = idx_ba.storage_factor();

        assert!((sf1 - 1.0).abs() < 0.01, "M=1 should have storage_factor ~1.0, got {}", sf1);
        assert!((sf4 - 4.0).abs() < 0.01, "M=4 should have storage_factor ~4.0, got {}", sf4);
        assert!(sf_ba > 1.0, "boundary-aware should store more than M=1, got {}", sf_ba);
        assert!(sf_ba < 4.0, "boundary-aware should store less than M=4, got {}", sf_ba);
    }
}
