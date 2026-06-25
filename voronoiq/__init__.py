"""
voronoiq — Weighted Voronoi Diagram Library
============================================

A Python library for constructing and visualizing weighted Voronoi diagrams,
including:

  * Multiplicatively Weighted Voronoi (power diagram variant)
  * Additively Weighted Voronoi
  * Power Diagrams (Laguerre–Voronoi)
  * Standard (unweighted) Voronoi — as a special case

Main API
--------
  from voronoiq import WeightedVoronoi

  wv = WeightedVoronoi(points, weights, mode="multiplicative")
  wv.compute()
  wv.plot()
  regions = wv.regions         # list of VoronoiRegion
  wv.to_svg("output.svg")
"""

from .diagram import WeightedVoronoi
from .region import VoronoiRegion
from .generators import random_generators, grid_generators, poisson_disk_generators
from .metrics import (
    multiplicative_weighted_distance,
    additive_weighted_distance,
    power_distance,
)

__version__ = "0.1.0"
__author__ = "voronoiq contributors"

__all__ = [
    "WeightedVoronoi",
    "VoronoiRegion",
    "random_generators",
    "grid_generators",
    "poisson_disk_generators",
    "multiplicative_weighted_distance",
    "additive_weighted_distance",
    "power_distance",
]
