"""
voronoiq.region
===============

VoronoiRegion — represents a single cell in a weighted Voronoi diagram.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class VoronoiRegion:
    """
    A single weighted Voronoi cell.

    Attributes
    ----------
    index : int
        Index of the generator that owns this region.
    generator : ndarray of shape (2,)
        (x, y) coordinates of the generator point.
    weight : float
        Weight associated with the generator.
    pixel_mask : ndarray of shape (H, W), dtype bool
        Boolean mask — True where pixels belong to this region.
    color : tuple of 3 floats in [0, 1]
        RGB colour used when plotting.
    area : int  (computed property)
        Number of pixels in this region.
    centroid : ndarray of shape (2,)  (computed property)
        Centroid of the region in pixel-index space.
    boundary_pixels : ndarray of shape (K, 2)  (computed property)
        Row/col indices of boundary pixels.
        Requires scipy — install with: pip install voronoiq[full]
    """

    index:      int
    generator:  np.ndarray
    weight:     float
    pixel_mask: np.ndarray
    color:      tuple = field(default_factory=lambda: (0.5, 0.5, 0.5))

    # ---------------------------------------------------------------- #
    #  Derived properties                                                #
    # ---------------------------------------------------------------- #

    @property
    def area(self) -> int:
        """Number of pixels in this region."""
        return int(np.sum(self.pixel_mask))

    @property
    def centroid(self) -> np.ndarray:
        """
        Centroid of the region in (col, row) pixel coordinates.
        Returns the generator position if the region is empty.
        """
        ys, xs = np.where(self.pixel_mask)
        if len(xs) == 0:
            return self.generator.copy()
        return np.array([xs.mean(), ys.mean()])

    @property
    def boundary_pixels(self) -> np.ndarray:
        """
        Indices (row, col) of pixels on the boundary of the region.
        A pixel is on the boundary if at least one of its 4-neighbours
        belongs to a different region.

        Requires scipy. Install with:
            pip install voronoiq[full]

        Returns
        -------
        ndarray of shape (K, 2)

        Raises
        ------
        ImportError  if scipy is not installed.
        """
        try:
            from scipy.ndimage import binary_erosion
        except ImportError:
            raise ImportError(
                "boundary_pixels requer scipy, que é uma dependência opcional.\n"
                "Instale com:  pip install voronoiq[full]"
            )
        eroded   = binary_erosion(self.pixel_mask)
        boundary = self.pixel_mask & ~eroded
        return np.array(np.where(boundary)).T  # shape (K, 2)

    # ---------------------------------------------------------------- #
    #  String representation                                             #
    # ---------------------------------------------------------------- #

    def __repr__(self):
        gx, gy = self.generator
        return (
            f"VoronoiRegion(index={self.index}, "
            f"generator=({gx:.3f}, {gy:.3f}), "
            f"weight={self.weight:.3f}, "
            f"area={self.area} px)"
        )