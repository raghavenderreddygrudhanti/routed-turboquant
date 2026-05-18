"""Routed TurboQuant: IVF-style float routing + TurboQuant SIMD scoring.

Sublinear approximate nearest neighbor search that scans only 6.2% of vectors
while maintaining near-lossless recall compared to flat TurboQuant.
"""

from routed_turboquant._core import RoutedTurboQuantIndex

__all__ = ["RoutedTurboQuantIndex"]
__version__ = "0.1.0"
