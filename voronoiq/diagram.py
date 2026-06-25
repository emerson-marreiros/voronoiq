"""
voronoiq.diagram
================

WeightedVoronoi — the central class of the voronoiq library.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

from .metrics import METRIC_REGISTRY
from .region import VoronoiRegion


class WeightedVoronoi:
    """
    Weighted Voronoi diagram over a 2-D rectangular domain.

    Parameters
    ----------
    points : array-like of shape (N, 2)
    weights : array-like of shape (N,)
    mode : {"multiplicative", "additive", "power"}
    resolution : int
    domain : tuple or None
    palette : str or list
    show_generators : bool
    show_weights : bool
    show_boundaries : bool
    """

    def __init__(
        self,
        points,
        weights,
        mode: str = "multiplicative",
        resolution: int = 512,
        domain=None,
        palette="tab20",
        show_generators: bool = True,
        show_weights: bool = False,
        show_boundaries: bool = True,
    ):
        self.points  = np.asarray(points, dtype=float)
        self.weights = np.asarray(weights, dtype=float)

        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("points must be of shape (N, 2)")
        if self.weights.ndim != 1 or len(self.weights) != len(self.points):
            raise ValueError("weights must be a 1-D array with length == len(points)")
        if mode not in METRIC_REGISTRY:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {list(METRIC_REGISTRY)}")

        self.mode             = mode
        self.resolution       = resolution
        self.palette          = palette
        self.show_generators  = show_generators
        self.show_weights     = show_weights
        self.show_boundaries  = show_boundaries

        if domain is None:
            margin = 0.05
            xmin, xmax = self.points[:, 0].min(), self.points[:, 0].max()
            ymin, ymax = self.points[:, 1].min(), self.points[:, 1].max()
            dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
            self.domain = (
                (xmin - margin * dx, xmax + margin * dx),
                (ymin - margin * dy, ymax + margin * dy),
            )
        else:
            self.domain = domain

        self.label_grid = None
        self.dist_grid  = None
        self.regions    = None
        self._colors    = []
        self._xs        = np.array([])
        self._ys        = np.array([])

    # ---------------------------------------------------------------- #
    #  Core computation                                                  #
    # ---------------------------------------------------------------- #

    def compute(self, batch_size: int = 32):
        """
        Rasterise the weighted Voronoi diagram.

        Parameters
        ----------
        batch_size : int
            Number of generators processed per iteration.
            Reduce if you hit memory limits; increase for speed.
            Default 32 keeps peak RAM under ~70 MB for resolution=512.

        Returns
        -------
        self  (for method chaining)
        """
        (xmin, xmax), (ymin, ymax) = self.domain
        aspect = (xmax - xmin) / (ymax - ymin)

        if aspect >= 1.0:
            W = self.resolution
            H = max(1, int(self.resolution / aspect))
        else:
            H = self.resolution
            W = max(1, int(self.resolution * aspect))

        xs = np.linspace(xmin, xmax, W)
        ys = np.linspace(ymin, ymax, H)
        self._xs, self._ys = xs, ys

        gens = np.column_stack([self.points, self.weights])
        n    = len(self.points)

        XX, YY = np.meshgrid(xs, ys)
        XX = XX[:, :, np.newaxis]
        YY = YY[:, :, np.newaxis]

        label_grid = np.full((H, W), -1,      dtype=np.int32)
        dist_grid  = np.full((H, W), np.inf,  dtype=np.float64)

        for start in range(0, n, batch_size):
            end   = min(start + batch_size, n)
            batch = gens[start:end]

            gx = batch[:, 0][np.newaxis, np.newaxis, :]
            gy = batch[:, 1][np.newaxis, np.newaxis, :]
            gw = batch[:, 2][np.newaxis, np.newaxis, :]

            if self.mode == "multiplicative":
                D = np.sqrt((XX - gx) ** 2 + (YY - gy) ** 2) / gw
            elif self.mode == "additive":
                D = np.sqrt((XX - gx) ** 2 + (YY - gy) ** 2) - gw
            else:  # power
                D = (XX - gx) ** 2 + (YY - gy) ** 2 - gw ** 2

            batch_min_idx = np.argmin(D, axis=2)
            batch_min_val = D.min(axis=2)

            update = batch_min_val < dist_grid
            dist_grid[update]  = batch_min_val[update]
            label_grid[update] = (start + batch_min_idx)[update]

        self.label_grid = label_grid
        self.dist_grid  = dist_grid

        cmap = plt.get_cmap(self.palette)
        self._colors = [cmap(i / max(n - 1, 1))[:3] for i in range(n)]

        self.regions = []
        for i in range(n):
            mask = self.label_grid == i
            self.regions.append(
                VoronoiRegion(
                    index=i,
                    generator=self.points[i].copy(),
                    weight=float(self.weights[i]),
                    pixel_mask=mask,
                    color=self._colors[i],
                )
            )

        return self

    # ---------------------------------------------------------------- #
    #  Querying                                                          #
    # ---------------------------------------------------------------- #

    def owner(self, x: float, y: float) -> int:
        """Return the index of the generator that owns point (x, y)."""
        self._require_computed()
        (xmin, xmax), (ymin, ymax) = self.domain
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            raise ValueError(f"Point ({x}, {y}) is outside the domain {self.domain}.")
        ci = int(np.searchsorted(self._xs, x))
        ri = int(np.searchsorted(self._ys, y))
        ci = np.clip(ci, 0, self.label_grid.shape[1] - 1)
        ri = np.clip(ri, 0, self.label_grid.shape[0] - 1)
        return int(self.label_grid[ri, ci])

    def region_of(self, x: float, y: float) -> "VoronoiRegion":
        """Return the VoronoiRegion that contains point (x, y)."""
        return self.regions[self.owner(x, y)]

    def nearest_generators(self, x: float, y: float, k: int = 3) -> list:
        """Return the k nearest generators to (x, y) sorted by weighted distance."""
        self._require_computed()
        gens  = np.column_stack([self.points, self.weights])
        metric = METRIC_REGISTRY[self.mode]
        dists = metric(x, y, gens)
        idx   = np.argsort(dists)[:k]
        return [(float(dists[i]), int(i)) for i in idx]

    # ---------------------------------------------------------------- #
    #  Visualisation                                                     #
    # ---------------------------------------------------------------- #

    def plot(
        self,
        ax=None,
        figsize=(8, 8),
        title=None,
        alpha: float = 0.85,
        boundary_color: str = "white",
        boundary_lw: float = 0.8,
        generator_size: int = 60,
        weight_fontsize: int = 8,
        cmap_override=None,
    ):
        """Render the weighted Voronoi diagram. Returns (fig, ax)."""
        self._require_computed()

        if cmap_override:
            cmap = plt.get_cmap(cmap_override)
            n = len(self.points)
            self._colors = [cmap(i / max(n - 1, 1))[:3] for i in range(n)]
            for r in self.regions:
                r.color = self._colors[r.index]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        (xmin, xmax), (ymin, ymax) = self.domain
        extent = [xmin, xmax, ymin, ymax]

        H, W = self.label_grid.shape
        rgb = np.zeros((H, W, 4), dtype=float)
        for i, color in enumerate(self._colors):
            mask = self.label_grid == i
            rgb[mask, :3] = color
            rgb[mask,  3] = alpha

        ax.imshow(rgb, origin="lower", extent=extent,
                  aspect="auto", interpolation="nearest")

        if self.show_boundaries:
            L     = self.label_grid
            right = np.pad(L, ((0, 0), (1, 0)), mode="edge")[:, :-1]
            down  = np.pad(L, ((1, 0), (0, 0)), mode="edge")[:-1, :]
            bnd   = (L != right) | (L != down)
            bnd_rgb = np.zeros((H, W, 4), dtype=float)
            bnd_rgb[bnd] = mcolors.to_rgba(boundary_color, alpha=1.0)
            ax.imshow(bnd_rgb, origin="lower", extent=extent,
                      aspect="auto", interpolation="nearest")

        if self.show_generators:
            colors_list = [self._colors[i] for i in range(len(self.points))]
            ax.scatter(self.points[:, 0], self.points[:, 1],
                       c=colors_list, s=generator_size,
                       edgecolors="black", linewidths=0.8, zorder=5)

        if self.show_weights:
            for i, (pt, w) in enumerate(zip(self.points, self.weights)):
                ax.text(pt[0], pt[1] + (ymax - ymin) * 0.015,
                        f"w={w:.2f}", ha="center", va="bottom",
                        fontsize=weight_fontsize, color="black",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  alpha=0.6, lw=0), zorder=6)

        mode_labels = {
            "multiplicative": "Multiplicatively Weighted",
            "additive":       "Additively Weighted",
            "power":          "Power Diagram (Laguerre)",
        }
        if title is None:
            title = f"{mode_labels[self.mode]} Voronoi  |  N={len(self.points)}"
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        fig.tight_layout()
        return fig, ax

    def plot_distance_field(self, ax=None, figsize=(8, 8), cmap="viridis"):
        """Plot the minimum weighted distance field as a heat map. Returns (fig, ax)."""
        self._require_computed()
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        (xmin, xmax), (ymin, ymax) = self.domain
        im = ax.imshow(self.dist_grid, origin="lower",
                       extent=[xmin, xmax, ymin, ymax],
                       aspect="auto", cmap=cmap)
        plt.colorbar(im, ax=ax, label="weighted distance")
        ax.scatter(self.points[:, 0], self.points[:, 1],
                   c="red", s=40, zorder=5, label="generators")
        ax.set_title(f"Distance field  ({self.mode})")
        fig.tight_layout()
        return fig, ax

    def plot_comparison(self, figsize=(18, 6)):
        """Side-by-side comparison of all three modes. Returns (fig, axes)."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, mode in zip(axes, ["multiplicative", "additive", "power"]):
            tmp = WeightedVoronoi(
                self.points, self.weights,
                mode=mode, resolution=self.resolution,
                domain=self.domain, palette=self.palette,
                show_generators=self.show_generators,
                show_weights=self.show_weights,
                show_boundaries=self.show_boundaries,
            )
            tmp.compute()
            tmp.plot(ax=ax)
        fig.tight_layout()
        return fig, axes

    # ---------------------------------------------------------------- #
    #  Export                                                            #
    # ---------------------------------------------------------------- #

    def to_png(self, path: str, dpi: int = 150, **plot_kwargs):
        """Save the diagram as a PNG file."""
        fig, _ = self.plot(**plot_kwargs)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def to_svg(self, path: str, **plot_kwargs):
        """Save the diagram as an SVG file."""
        fig, _ = self.plot(**plot_kwargs)
        fig.savefig(path, format="svg", bbox_inches="tight")
        plt.close(fig)
        return path

    def to_csv(self, path: str):
        """Export generator metadata as CSV."""
        self._require_computed()
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "x", "y", "weight", "area_px",
                        "centroid_x", "centroid_y"])
            for r in self.regions:
                cx, cy = r.centroid
                w.writerow([r.index, r.generator[0], r.generator[1],
                             r.weight, r.area, cx, cy])
        return path

    def to_label_array(self) -> np.ndarray:
        """Return a copy of the (H, W) integer label grid."""
        self._require_computed()
        return self.label_grid.copy()

    # ---------------------------------------------------------------- #
    #  Internal helpers                                                  #
    # ---------------------------------------------------------------- #

    def _require_computed(self):
        if self.label_grid is None:
            raise RuntimeError(
                "Call .compute() before accessing diagram data or plotting."
            )

    def __repr__(self):
        n        = len(self.points)
        computed = "computed" if self.label_grid is not None else "not computed"
        return (
            f"WeightedVoronoi(n={n}, mode='{self.mode}', "
            f"resolution={self.resolution}, {computed})"
        )