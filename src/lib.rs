//! Routed TurboQuant: IVF-style float routing + TurboQuant SIMD scoring.
//!
//! Achieves sublinear search with recall that exceeds flat TurboQuant
//! by combining partition routing, multi-assignment, and float reranking.

extern crate blas_src;

pub mod kmeans;
pub mod index;

pub use index::RoutedTurboQuantIndex;
pub use index::SearchStats;
