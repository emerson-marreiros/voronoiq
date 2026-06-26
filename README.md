# voronoiq — Weighted Voronoi Diagrams for Python

**voronoiq** is a Python library for calculating Weighted Voronoi Diagrams—or Power Diagrams—with real-time weight updates.

Unlike classical Voronoi diagrams, **voronoiq** adjusts cell sizes based on weights associated with each site, enabling dynamic load balancing. It is implemented using NumPy and SciPy for O(n log n) performance and features integrated visualization via Matplotlib.

## Applications

Weighted Voronoi Diagrams are used whenever spatial regions must adapt to heterogeneous capacity or cost. `voronoiq` enables these use cases in Python:

### Scientific Research
- **Computational Biology**: Tissue modeling where cell growth rates are encoded as weights. Applied in microscopy image segmentation and cell division modeling.
- **Urban Planning**: Coverage areas for schools, hospitals, and public services weighted by capacity to avoid overcrowding.
- **Cosmology**: Mapping cosmic voids and large-scale structure where galaxy mass acts as site weight.

### Industry
- **Logistics**: Dynamic delivery zone allocation where distribution centers with high order volume get reduced weight, pushing demand to nearby hubs.
- **AgriTech**: Precision farming with sensor-weighted irrigation zones based on soil moisture and fertility measurements.
- **Game Development**: Procedural map generation with weighted biomes and territories to balance resource distribution.

For implementation details and examples, see the `examples/` directory.

A lightweight, dependency-light Python library for constructing and
visualising **weighted Voronoi diagrams**, including:

| Mode | Distance function | Effect of larger weight |
|---|---|---|
| `"multiplicative"` | `dist(p,g) / w(g)` | larger region |
| `"additive"` | `dist(p,g) − w(g)` | larger region |
| `"power"` | `dist(p,g)² − w(g)²` | larger region (power diagram) |

---

## Installation

```bash
pip install numpy scipy matplotlib
# clone / copy voronoiq/ into your project
```

Dependencies: **numpy**, **scipy** (optional, for boundary_pixels),
**matplotlib** (for visualisation).

---

## Quick start

```python
import numpy as np
from voronoiq import WeightedVoronoi

pts = np.array([[0.2, 0.3],
                [0.7, 0.6],
                [0.5, 0.1],
                [0.1, 0.9]])
w   = np.array([1.0, 2.5, 0.5, 1.8])

wv = WeightedVoronoi(pts, w, mode="multiplicative", resolution=512)
wv.compute()
wv.plot()          # shows an interactive matplotlib figure
wv.to_png("out.png")
```

---

## API reference

### `WeightedVoronoi(points, weights, **kwargs)`

| Parameter | Default | Description |
|---|---|---|
| `points` | — | `(N, 2)` generator coordinates |
| `weights` | — | `(N,)` generator weights |
| `mode` | `"multiplicative"` | distance metric |
| `resolution` | `512` | pixels along longer axis |
| `domain` | auto (bounding box + 5 %) | `((xmin,xmax),(ymin,ymax))` |
| `palette` | `"tab20"` | matplotlib colormap name |
| `show_generators` | `True` | draw seed points |
| `show_weights` | `False` | annotate weights |
| `show_boundaries` | `True` | draw cell edges |

#### Methods

```python
wv.compute()                     # rasterise the diagram (required first)

wv.plot(kwargs)                # returns (fig, ax)
wv.plot_distance_field()         # heat-map of min weighted distance
wv.plot_comparison()             # side-by-side of all 3 modes

wv.owner(x, y)                   # generator index owning (x, y)
wv.region_of(x, y)               # VoronoiRegion containing (x, y)
wv.nearest_generators(x, y, k=3) # k nearest generators by weighted dist

wv.to_png("out.png", dpi=150)
wv.to_svg("out.svg")
wv.to_csv("out.csv")             # index, x, y, weight, area, centroid
wv.to_label_array()              # (H, W) int ndarray — copy
```

#### Key attributes (after `compute()`)

| Attribute | Type | Description |
|---|---|---|
| `label_grid` | `(H, W) int32` | generator index per pixel |
| `dist_grid` | `(H, W) float64` | minimum weighted distance per pixel |
| `regions` | `list[VoronoiRegion]` | one object per generator |

---

### `VoronoiRegion`

```python
r = wv.regions[0]

r.index          # int — generator index
r.generator      # (2,) float — (x, y)
r.weight         # float
r.pixel_mask     # (H, W) bool
r.color          # (R, G, B) tuple

r.area           # int — number of pixels
r.centroid       # (2,) float — mean (x, y) of mask pixels
r.boundary_pixels # (K, 2) row/col indices of boundary pixels
```

---

### `voronoiq.generators`

```python
from voronoiq.generators import (
    random_generators,           # uniform random
    grid_generators,             # regular grid with optional jitter
    poisson_disk_generators,     # Bridson blue-noise sampling
)

pts, w = random_generators(n=20, weight_range=(0.5, 2.0), seed=42)
pts, w = grid_generators(nx=6, ny=6, jitter=0.04, seed=0)
pts, w = poisson_disk_generators(min_dist=0.1, seed=7)
```

All functions return `(points, weights)` tuples ready for
`WeightedVoronoi`.

---

### `voronoiq.metrics`

```python
from voronoiq.metrics import (
    multiplicative_weighted_distance,  # scalar
    additive_weighted_distance,
    power_distance,
    batch_multiplicative,              # vectorised over generators
    batch_additive,
    batch_power,
)
```

---

## Examples

### Comparison of all three modes

```python
wv = WeightedVoronoi(pts, w, mode="multiplicative", resolution=400)
wv.compute()
fig, axes = wv.plot_comparison(figsize=(18, 6))
```

### Distance field heat-map

```python
wv.plot_distance_field(cmap="plasma")
```

### Querying which region owns a point

```python
idx = wv.owner(0.5, 0.5)
region = wv.region_of(0.5, 0.5)
print(region)
# VoronoiRegion(index=1, generator=(0.700, 0.600), weight=2.500, area=14832 px)
```

### Exporting

```python
wv.to_png("voronoi.png", dpi=200)
wv.to_svg("voronoi.svg")
wv.to_csv("voronoi.csv")
```

---

## Project structure

```
voronoiq/
├── __init__.py      # public API
├── diagram.py       # WeightedVoronoi class
├── region.py        # VoronoiRegion dataclass
├── generators.py    # random / grid / Poisson-disk seed generators
└── metrics.py       # distance functions + registry
tests/
└── test_voronoiq.py # full test suite (pytest)
README.md
```

---

## License

MIT
