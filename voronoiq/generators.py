"""
voronoiq.generators
===================

Utility functions for creating sets of generator points (seeds) and
their associated weights, ready to feed into ``WeightedVoronoi``.

All functions return a tuple ``(points, weights)`` where

  * ``points``  — ndarray of shape (N, 2), values in [0, 1]²
  * ``weights`` — ndarray of shape (N,),  values in ``weight_range``
"""

import numpy as np


# ------------------------------------------------------------------ #
#  Random uniform                                                      #
# ------------------------------------------------------------------ #

def random_generators(
    n: int = 20,
    weight_range: tuple = (0.5, 2.0),
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate *n* random points inside the unit square with random weights.

    Parameters
    ----------
    n            : number of generators
    weight_range : (min_weight, max_weight) for uniform sampling
    seed         : optional random seed for reproducibility

    Returns
    -------
    points  : ndarray (n, 2)
    weights : ndarray (n,)

    Example
    -------
    >>> from voronoiq.generators import random_generators
    >>> pts, w = random_generators(10, seed=42)
    >>> pts.shape
    (10, 2)
    """
    rng = np.random.default_rng(seed)
    points  = rng.random((n, 2))
    lo, hi  = weight_range
    weights = rng.uniform(lo, hi, n)
    return points, weights


# ------------------------------------------------------------------ #
#  Regular grid                                                        #
# ------------------------------------------------------------------ #

def grid_generators(
    nx: int = 5,
    ny: int = 5,
    weight_range: tuple = (0.5, 2.0),
    jitter: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Place generators on a regular *nx × ny* grid inside [0, 1]².
    An optional random jitter is applied to break the symmetry.

    Parameters
    ----------
    nx, ny       : grid dimensions
    weight_range : (min_weight, max_weight) for uniform weight sampling
    jitter       : maximum displacement in each axis (in normalised units)
    seed         : optional random seed

    Returns
    -------
    points  : ndarray (nx*ny, 2)
    weights : ndarray (nx*ny,)

    Example
    -------
    >>> from voronoiq.generators import grid_generators
    >>> pts, w = grid_generators(4, 4, jitter=0.05, seed=0)
    >>> pts.shape
    (16, 2)
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 1, nx + 2)[1:-1]
    ys = np.linspace(0, 1, ny + 2)[1:-1]
    gx, gy = np.meshgrid(xs, ys)
    points = np.stack([gx.ravel(), gy.ravel()], axis=1)
    if jitter > 0:
        points += rng.uniform(-jitter, jitter, points.shape)
        points = np.clip(points, 0.0, 1.0)
    lo, hi  = weight_range
    weights = rng.uniform(lo, hi, len(points))
    return points, weights


# ------------------------------------------------------------------ #
#  Poisson-disk sampling (blue noise)                                  #
# ------------------------------------------------------------------ #

def poisson_disk_generators(
    min_dist: float = 0.1,
    weight_range: tuple = (0.5, 2.0),
    max_attempts: int = 30,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate well-distributed generator points using Bridson's Poisson-disk
    sampling algorithm (blue noise).  Points are guaranteed to be at least
    ``min_dist`` apart.

    Parameters
    ----------
    min_dist     : minimum distance between any two generators
    weight_range : (min_weight, max_weight) for uniform weight sampling
    max_attempts : number of candidate samples per active point
    seed         : optional random seed

    Returns
    -------
    points  : ndarray (N, 2),  N varies depending on min_dist
    weights : ndarray (N,)

    Example
    -------
    >>> from voronoiq.generators import poisson_disk_generators
    >>> pts, w = poisson_disk_generators(min_dist=0.15, seed=7)
    >>> len(pts) > 0
    True
    """
    rng = np.random.default_rng(seed)

    cell_size = min_dist / np.sqrt(2)
    grid_w = int(np.ceil(1.0 / cell_size))
    grid   = {}          # (ci, cj) → point index

    def _cell(p):
        return int(p[0] / cell_size), int(p[1] / cell_size)

    def _neighbours_ok(p, pts):
        cx, cy = _cell(p)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                key = (cx + di, cy + dj)
                if key in grid:
                    q = pts[grid[key]]
                    if np.hypot(p[0] - q[0], p[1] - q[1]) < min_dist:
                        return False
        return True

    first = rng.random(2)
    points = [first]
    grid[_cell(first)] = 0
    active = [0]

    while active:
        idx = rng.integers(0, len(active))
        p   = points[active[idx]]
        found = False
        for _ in range(max_attempts):
            angle = rng.uniform(0, 2 * np.pi)
            r     = rng.uniform(min_dist, 2 * min_dist)
            q     = p + r * np.array([np.cos(angle), np.sin(angle)])
            if 0 <= q[0] <= 1 and 0 <= q[1] <= 1 and _neighbours_ok(q, points):
                grid[_cell(q)] = len(points)
                active.append(len(points))
                points.append(q)
                found = True
                break
        if not found:
            active.pop(idx)

    pts_arr = np.array(points)
    lo, hi  = weight_range
    weights = rng.uniform(lo, hi, len(pts_arr))
    return pts_arr, weights
