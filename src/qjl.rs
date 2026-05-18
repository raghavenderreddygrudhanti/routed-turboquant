//! QJL (Quantized Johnson-Lindenstrauss) residual correction.
//!
//! After TurboQuant quantizes each coordinate to b bits, there's a residual:
//!   residual[i] = rotated[i] - centroid[quantized_code[i]]
//!
//! QJL stores sign(residual[i]) as 1 bit per coordinate. At search time,
//! the correction term is:
//!   correction = norm_db * norm_q * scale * sum(sign_db[i] * q_rotated[i])
//!
//! This makes the inner product estimator unbiased (TurboQuant_mse is biased).
//!
//! For vector search at dim=384, this adds 384/8 = 48 bytes per vector
//! (12.5% overhead on top of 4-bit codes which are 192 bytes/vector).

use rand::rngs::StdRng;
use rand::SeedableRng;

/// QJL correction data for a set of vectors.
pub struct QjlCorrection {
    /// Sign bits of residuals: n_vectors × n_bytes, packed 8 signs per byte.
    sign_bits: Vec<u8>,
    /// Number of vectors.
    n_vectors: usize,
    /// Dimensionality.
    dim: usize,
    /// Bytes per vector (dim / 8).
    n_bytes: usize,
    /// Per-vector residual magnitude scale (E[|residual|] per coordinate).
    /// Used to weight the correction term.
    scales: Vec<f32>,
    /// Random sign vector for the QJL projection (dim elements, ±1).
    /// Shared across all vectors (deterministic from seed).
    random_signs: Vec<i8>,
    /// Rotation matrix (dim × dim) — same as turbovec uses.
    rotation: Vec<f32>,
}

impl QjlCorrection {
    /// Build QJL correction data from original vectors.
    ///
    /// This re-encodes vectors through the same rotation + Lloyd-Max pipeline
    /// as turbovec, computes residuals, and stores their signs.
    ///
    /// `vectors`: flat f32 of shape (n, dim)
    /// `bit_width`: same as turbovec (2, 3, or 4)
    /// `seed`: random seed for rotation matrix (must match turbovec's seed=42)
    pub fn build(vectors: &[f32], dim: usize, bit_width: usize, seed: u64) -> Self {
        let n = vectors.len() / dim;
        assert_eq!(vectors.len(), n * dim);
        let n_bytes = dim / 8;

        // Generate rotation matrix (same as turbovec: random orthogonal via QR)
        let rotation = make_rotation_matrix(dim, seed);

        // Generate random signs for QJL projection
        let mut rng = StdRng::seed_from_u64(seed + 1); // different seed from rotation
        let random_signs: Vec<i8> = (0..dim)
            .map(|_| {
                if rand::Rng::gen_bool(&mut rng, 0.5) {
                    1
                } else {
                    -1
                }
            })
            .collect();

        // Compute Lloyd-Max boundaries and centroids
        let (boundaries, centroids) = lloyd_max_codebook(bit_width, dim);
        let n_levels = 1 << bit_width;

        // Encode each vector: normalize → rotate → quantize → compute residual → store sign
        let mut sign_bits = vec![0u8; n * n_bytes];
        let mut scales = vec![0.0f32; n];

        for i in 0..n {
            let row = &vectors[i * dim..(i + 1) * dim];

            // Normalize
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            let inv_norm = if norm > 1e-10 { 1.0 / norm } else { 0.0 };

            // Rotate
            let mut rotated = vec![0.0f32; dim];
            for j in 0..dim {
                let mut sum = 0.0f32;
                for k in 0..dim {
                    sum += row[k] * inv_norm * rotation[j * dim + k];
                }
                rotated[j] = sum;
            }

            // Quantize each coordinate and compute residual
            let mut residual_abs_sum = 0.0f32;
            for j in 0..dim {
                let val = rotated[j];

                // Find quantization level (binary search in boundaries)
                let mut level = 0usize;
                for l in 0..n_levels - 1 {
                    if val > boundaries[l] {
                        level = l + 1;
                    }
                }
                let centroid_val = centroids[level];
                let residual = val - centroid_val;

                // Store sign of (residual * random_sign)
                let projected = residual * random_signs[j] as f32;
                if projected >= 0.0 {
                    sign_bits[i * n_bytes + j / 8] |= 1 << (7 - j % 8);
                }

                residual_abs_sum += residual.abs();
            }

            // Scale = average absolute residual (E[|r|] per coordinate)
            scales[i] = residual_abs_sum / dim as f32;
        }

        Self {
            sign_bits,
            n_vectors: n,
            dim,
            n_bytes,
            scales,
            random_signs,
            rotation,
        }
    }

    /// Compute QJL correction for a query against a candidate vector.
    ///
    /// Returns the correction term to ADD to the TQ score.
    /// The corrected score = tq_score + qjl_correction.
    ///
    /// `query`: raw query vector (dim,)
    /// `vec_idx`: index of the database vector
    pub fn correction(&self, query: &[f32], vec_idx: usize) -> f32 {
        assert_eq!(query.len(), self.dim);
        assert!(vec_idx < self.n_vectors);

        // Rotate query (same rotation as encoding)
        let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let inv_norm = if q_norm > 1e-10 { 1.0 / q_norm } else { 0.0 };

        // Compute: sum(sign_bits[vec_idx][j] * random_signs[j] * q_rotated[j])
        // This is the QJL inner product estimate of the residual component
        let sign_row = &self.sign_bits[vec_idx * self.n_bytes..(vec_idx + 1) * self.n_bytes];

        let mut correction_sum = 0.0f32;
        for j in 0..self.dim {
            // Get stored sign bit
            let byte_idx = j / 8;
            let bit_idx = 7 - j % 8;
            let stored_sign: f32 = if (sign_row[byte_idx] >> bit_idx) & 1 == 1 {
                1.0
            } else {
                -1.0
            };

            // Rotate query coordinate
            let mut q_rot_j = 0.0f32;
            for k in 0..self.dim {
                q_rot_j += query[k] * inv_norm * self.rotation[j * self.dim + k];
            }

            // QJL correction: stored_sign * random_sign * q_rotated
            correction_sum += stored_sign * self.random_signs[j] as f32 * q_rot_j;
        }

        // Scale by database vector's residual magnitude and query norm
        // The expected correction magnitude is: scale * sqrt(2/pi) * correction_sum
        let scale_factor = self.scales[vec_idx] * q_norm * (2.0 / std::f32::consts::PI).sqrt();
        scale_factor * correction_sum / self.dim as f32
    }

    /// Batch correction for multiple candidates.
    /// Returns correction values for each candidate index.
    pub fn batch_correction(&self, query: &[f32], candidate_indices: &[usize]) -> Vec<f32> {
        // Pre-rotate query once
        let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let inv_norm = if q_norm > 1e-10 { 1.0 / q_norm } else { 0.0 };

        let mut q_rotated = vec![0.0f32; self.dim];
        for j in 0..self.dim {
            let mut sum = 0.0f32;
            for k in 0..self.dim {
                sum += query[k] * inv_norm * self.rotation[j * self.dim + k];
            }
            q_rotated[j] = sum;
        }

        // Pre-multiply q_rotated by random_signs
        let q_rot_signed: Vec<f32> = q_rotated
            .iter()
            .zip(self.random_signs.iter())
            .map(|(&q, &s)| q * s as f32)
            .collect();

        // Compute correction for each candidate
        candidate_indices
            .iter()
            .map(|&idx| {
                let sign_row = &self.sign_bits[idx * self.n_bytes..(idx + 1) * self.n_bytes];

                let mut dot = 0.0f32;
                for j in 0..self.dim {
                    let byte_idx = j / 8;
                    let bit_idx = 7 - j % 8;
                    let stored_sign: f32 = if (sign_row[byte_idx] >> bit_idx) & 1 == 1 {
                        1.0
                    } else {
                        -1.0
                    };
                    dot += stored_sign * q_rot_signed[j];
                }

                let scale_factor = self.scales[idx] * q_norm * (2.0 / std::f32::consts::PI).sqrt();
                scale_factor * dot / self.dim as f32
            })
            .collect()
    }

    /// Memory overhead in bytes.
    pub fn memory_bytes(&self) -> usize {
        // sign_bits + scales + random_signs + rotation
        self.n_vectors * self.n_bytes + self.n_vectors * 4 + self.dim + self.dim * self.dim * 4
    }

    pub fn n_vectors(&self) -> usize {
        self.n_vectors
    }
}

/// Generate a random orthogonal rotation matrix via QR decomposition.
/// Uses the same approach as turbovec (Gram-Schmidt on random Gaussian matrix).
fn make_rotation_matrix(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut matrix = vec![0.0f32; dim * dim];

    // Fill with random Gaussian
    for i in 0..dim * dim {
        // Box-Muller transform for Gaussian
        let u1: f32 = rand::Rng::gen_range(&mut rng, 0.0001f32..1.0);
        let u2: f32 = rand::Rng::gen_range(&mut rng, 0.0f32..std::f32::consts::TAU);
        matrix[i] = (-2.0 * u1.ln()).sqrt() * u2.cos();
    }

    // Gram-Schmidt orthogonalization
    for i in 0..dim {
        // Subtract projections onto previous vectors
        for j in 0..i {
            let mut dot = 0.0f32;
            for k in 0..dim {
                dot += matrix[i * dim + k] * matrix[j * dim + k];
            }
            for k in 0..dim {
                matrix[i * dim + k] -= dot * matrix[j * dim + k];
            }
        }
        // Normalize
        let norm: f32 = (0..dim)
            .map(|k| matrix[i * dim + k] * matrix[i * dim + k])
            .sum::<f32>()
            .sqrt();
        if norm > 1e-10 {
            for k in 0..dim {
                matrix[i * dim + k] /= norm;
            }
        }
    }

    matrix
}

/// Compute Lloyd-Max codebook boundaries and centroids for a given bit width.
/// Assumes the input distribution is approximately N(0, 1/dim) after rotation.
fn lloyd_max_codebook(bit_width: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let n_levels = 1usize << bit_width;
    let sigma = 1.0 / (dim as f32).sqrt();

    // For Gaussian N(0, sigma^2), Lloyd-Max boundaries are symmetric
    // Use pre-computed optimal boundaries for common bit widths
    let (raw_boundaries, raw_centroids) = match bit_width {
        2 => {
            // 4 levels: boundaries at [-0.9816, 0, 0.9816] * sigma
            let b = vec![-0.9816f32, 0.0, 0.9816];
            let c = vec![-1.51f32, -0.4528, 0.4528, 1.51];
            (b, c)
        }
        3 => {
            // 8 levels
            let b = vec![-1.748f32, -1.050, -0.5006, 0.0, 0.5006, 1.050, 1.748];
            let c = vec![
                -2.152f32, -1.344, -0.7560, -0.2451, 0.2451, 0.7560, 1.344, 2.152,
            ];
            (b, c)
        }
        4 => {
            // 16 levels
            let b = vec![
                -2.401f32, -1.844, -1.437, -1.099, -0.7975, -0.5176, -0.2510, 0.0, 0.2510, 0.5176,
                0.7975, 1.099, 1.437, 1.844, 2.401,
            ];
            let c = vec![
                -2.733f32, -2.069, -1.618, -1.256, -0.9424, -0.6568, -0.3881, -0.1284, 0.1284,
                0.3881, 0.6568, 0.9424, 1.256, 1.618, 2.069, 2.733,
            ];
            (b, c)
        }
        _ => panic!("unsupported bit_width: {}", bit_width),
    };

    // Scale by sigma
    let boundaries: Vec<f32> = raw_boundaries.iter().map(|&b| b * sigma).collect();
    let centroids: Vec<f32> = raw_centroids.iter().map(|&c| c * sigma).collect();

    (boundaries, centroids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qjl_builds() {
        let dim = 32;
        let n = 100;
        let mut rng = StdRng::seed_from_u64(42);
        let mut vectors = vec![0.0f32; n * dim];
        for v in vectors.iter_mut() {
            *v = rand::Rng::gen_range(&mut rng, -1.0..1.0);
        }

        let qjl = QjlCorrection::build(&vectors, dim, 4, 42);
        assert_eq!(qjl.n_vectors(), n);
        assert!(qjl.memory_bytes() > 0);
    }

    #[test]
    fn test_qjl_correction_bounded() {
        let dim = 32;
        let n = 50;
        let mut rng = StdRng::seed_from_u64(42);
        let mut vectors = vec![0.0f32; n * dim];
        for v in vectors.iter_mut() {
            *v = rand::Rng::gen_range(&mut rng, -1.0..1.0);
        }

        let qjl = QjlCorrection::build(&vectors, dim, 4, 42);

        // Correction should be small relative to inner product magnitude
        let query = &vectors[0..dim];
        for i in 0..n {
            let corr = qjl.correction(query, i);
            // Correction should be bounded (not exploding)
            assert!(
                corr.abs() < 1.0,
                "correction too large: {} for vec {}",
                corr,
                i
            );
        }
    }

    #[test]
    fn test_batch_correction_matches_single() {
        let dim = 32;
        let n = 20;
        let mut rng = StdRng::seed_from_u64(99);
        let mut vectors = vec![0.0f32; n * dim];
        for v in vectors.iter_mut() {
            *v = rand::Rng::gen_range(&mut rng, -1.0..1.0);
        }

        let qjl = QjlCorrection::build(&vectors, dim, 4, 42);
        let query = &vectors[0..dim];
        let indices: Vec<usize> = (0..n).collect();

        let batch = qjl.batch_correction(query, &indices);
        for i in 0..n {
            let single = qjl.correction(query, i);
            assert!(
                (batch[i] - single).abs() < 1e-5,
                "mismatch at {}: batch={} single={}",
                i,
                batch[i],
                single
            );
        }
    }
}
