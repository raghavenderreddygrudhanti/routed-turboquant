//! Routed TurboQuant: IVF-style float routing + TurboQuant SIMD scoring.
//!
//! Achieves sublinear search (O(n/P * R) instead of O(n)) while maintaining
//! the same recall and memory footprint as flat TurboQuant.

extern crate blas_src;

pub mod kmeans;
pub mod index;
pub mod index_v2;

#[cfg(feature = "python")]
pub mod python;

pub use index::RoutedTurboQuantIndex;
pub use index::SearchStats;
pub use index_v2::RoutedV2Index;
