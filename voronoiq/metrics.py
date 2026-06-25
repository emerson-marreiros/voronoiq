"""
voronoiq.metrics
================

Distance functions used to assign pixels/cells to generators in
weighted Voronoi diagrams.

Each function follows the signature::

    distance(px, py, gx, gy, weight) -> float

where (px, py) is the query point and (gx, gy, weight) describe one
generator.

Usage example
-------------
    from voronoiq.metrics import multiplicative_weighted_distance
    d = multiplicative_weighted_distance(0.5, 0.3, 1.0, 1.0, 2.0)
"""

import numpy as np


# ------------------------------------------------------------------ #
#  Core scalar functions                                               #
# ------------------------------------------------------------------ #

def multiplicative_weighted_distance(px, py, gx, gy, weight):
    """
    Multiplicatively Weighted Voronoi distance.

    d_mw(p, g) = euclidean(p, g) / weight(g)

    A generator with a *larger* weight has a *larger* region because
    the effective distance is divided by the weight.

    Parameters
    ----------
    px, py : float   — query point coordinates
    gx, gy : float   — generator coordinates
    weight : float   — positive weight of the generator

    Returns
    -------
    float
    """
    if weight <= 0:
        raise ValueError("Weight must be strictly positive.")
    return np.sqrt((px - gx) ** 2 + (py - gy) ** 2) / weight


def additive_weighted_distance(px, py, gx, gy, weight):
    """
    Additively Weighted Voronoi distance.

    d_aw(p, g) = euclidean(p, g) - weight(g)

    A generator with a *larger* weight has a *larger* region because
    the effective distance is reduced by the weight.

    Parameters
    ----------
    px, py : float   — query point coordinates
    gx, gy : float   — generator coordinates
    weight : float   — weight (can be any real number, but typically ≥ 0)

    Returns
    -------
    float
    """
    return np.sqrt((px - gx) ** 2 + (py - gy) ** 2) - weight


def power_distance(px, py, gx, gy, weight):
    """
    Power (Laguerre–Voronoi) distance.

    d_pow(p, g) = euclidean(p, g)^2 - weight(g)^2

    This is the classical *power diagram* / *weighted Delaunay* distance.
    For equal weights it degenerates to the standard Voronoi diagram.

    Parameters
    ----------
    px, py : float   — query point coordinates
    gx, gy : float   — generator coordinates
    weight : float   — radius / weight of the generator

    Returns
    -------
    float
    """
    return (px - gx) ** 2 + (py - gy) ** 2 - weight ** 2


# ------------------------------------------------------------------ #
#  Vectorised batch variants (numpy arrays)                            #
# ------------------------------------------------------------------ #

def batch_multiplicative(px, py, generators):
    """
    Compute multiplicative weighted distances from (px, py) to every
    generator.

    Parameters
    ----------
    px, py      : float
    generators  : ndarray of shape (N, 3) — columns [gx, gy, weight]

    Returns
    -------
    ndarray of shape (N,)
    """
    gx, gy, w = generators[:, 0], generators[:, 1], generators[:, 2]
    return np.sqrt((px - gx) ** 2 + (py - gy) ** 2) / w


def batch_additive(px, py, generators):
    """
    Compute additive weighted distances from (px, py) to every generator.

    Parameters
    ----------
    px, py      : float
    generators  : ndarray of shape (N, 3) — columns [gx, gy, weight]

    Returns
    -------
    ndarray of shape (N,)
    """
    gx, gy, w = generators[:, 0], generators[:, 1], generators[:, 2]
    return np.sqrt((px - gx) ** 2 + (py - gy) ** 2) - w


def batch_power(px, py, generators):
    """
    Compute power distances from (px, py) to every generator.

    Parameters
    ----------
    px, py      : float
    generators  : ndarray of shape (N, 3) — columns [gx, gy, weight]

    Returns
    -------
    ndarray of shape (N,)
    """
    gx, gy, w = generators[:, 0], generators[:, 1], generators[:, 2]
    return (px - gx) ** 2 + (py - gy) ** 2 - w ** 2


# ------------------------------------------------------------------ #
#  Registry — map mode name → batch function                           #
# ------------------------------------------------------------------ #

METRIC_REGISTRY = {
    "multiplicative": batch_multiplicative,
    "additive":       batch_additive,
    "power":          batch_power,
}
