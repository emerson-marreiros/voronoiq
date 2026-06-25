"""
Tests for the voronoiq library.
Run with:  python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voronoiq import WeightedVoronoi, VoronoiRegion
from voronoiq.generators import (
    random_generators,
    grid_generators,
    poisson_disk_generators,
)
from voronoiq.metrics import (
    multiplicative_weighted_distance,
    additive_weighted_distance,
    power_distance,
    batch_multiplicative,
    batch_additive,
    batch_power,
)


# ================================================================== #
#  Metric tests                                                        #
# ================================================================== #

class TestMetrics:

    def test_multiplicative_basic(self):
        d = multiplicative_weighted_distance(3, 4, 0, 0, 5)
        assert d == pytest.approx(1.0, rel=1e-6)

    def test_multiplicative_invalid_weight(self):
        with pytest.raises(ValueError):
            multiplicative_weighted_distance(1, 1, 0, 0, 0)

    def test_additive_basic(self):
        d = additive_weighted_distance(3, 4, 0, 0, 3)
        assert d == pytest.approx(2.0, rel=1e-6)

    def test_power_basic(self):
        d = power_distance(3, 4, 0, 0, 2)
        assert d == pytest.approx(21.0, rel=1e-6)  # 25 - 4

    def test_batch_multiplicative_shape(self):
        gens = np.array([[0, 0, 1.0], [1, 1, 2.0]])
        d = batch_multiplicative(0.5, 0.5, gens)
        assert d.shape == (2,)

    def test_batch_shapes_consistent(self):
        gens = np.random.rand(10, 3)
        gens[:, 2] = np.abs(gens[:, 2]) + 0.1
        d_m = batch_multiplicative(0.5, 0.5, gens)
        d_a = batch_additive(0.5, 0.5, gens)
        d_p = batch_power(0.5, 0.5, gens)
        assert d_m.shape == d_a.shape == d_p.shape == (10,)


# ================================================================== #
#  Generator tests                                                     #
# ================================================================== #

class TestGenerators:

    def test_random_shape(self):
        pts, w = random_generators(15, seed=0)
        assert pts.shape == (15, 2)
        assert w.shape  == (15,)

    def test_random_range(self):
        pts, w = random_generators(100, weight_range=(1.0, 3.0), seed=1)
        assert pts.min() >= 0.0
        assert pts.max() <= 1.0
        assert w.min()   >= 1.0
        assert w.max()   <= 3.0

    def test_grid_shape(self):
        pts, w = grid_generators(4, 3, seed=2)
        assert pts.shape == (12, 2)

    def test_grid_jitter_stays_in_bounds(self):
        pts, _ = grid_generators(5, 5, jitter=0.04, seed=3)
        assert pts.min() >= 0.0
        assert pts.max() <= 1.0

    def test_poisson_disk_min_distance(self):
        pts, _ = poisson_disk_generators(min_dist=0.12, seed=5)
        # Verify no two points are closer than min_dist
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                assert d >= 0.12 - 1e-9


# ================================================================== #
#  WeightedVoronoi tests                                               #
# ================================================================== #

class TestWeightedVoronoi:

    @pytest.fixture
    def simple_wv(self):
        pts = np.array([[0.2, 0.3], [0.7, 0.6], [0.5, 0.1], [0.1, 0.9]])
        w   = np.array([1.0, 2.0, 0.5, 1.5])
        wv  = WeightedVoronoi(pts, w, mode="multiplicative", resolution=64)
        wv.compute()
        return wv

    def test_repr_before_compute(self):
        pts, w = random_generators(5, seed=0)
        wv = WeightedVoronoi(pts, w, resolution=32)
        assert "not computed" in repr(wv)

    def test_repr_after_compute(self, simple_wv):
        assert "computed" in repr(simple_wv)

    def test_regions_count(self, simple_wv):
        assert len(simple_wv.regions) == 4

    def test_label_grid_shape(self, simple_wv):
        H, W = simple_wv.label_grid.shape
        assert H > 0 and W > 0

    def test_label_grid_values(self, simple_wv):
        n = len(simple_wv.points)
        assert simple_wv.label_grid.min() == 0
        assert simple_wv.label_grid.max() == n - 1

    def test_dist_grid_shape(self, simple_wv):
        assert simple_wv.dist_grid.shape == simple_wv.label_grid.shape

    def test_regions_are_VoronoiRegion(self, simple_wv):
        for r in simple_wv.regions:
            assert isinstance(r, VoronoiRegion)

    def test_region_area_positive(self, simple_wv):
        for r in simple_wv.regions:
            assert r.area > 0

    def test_region_masks_cover_all_pixels(self, simple_wv):
        total = sum(r.area for r in simple_wv.regions)
        H, W = simple_wv.label_grid.shape
        assert total == H * W

    def test_region_masks_non_overlapping(self, simple_wv):
        combined = sum(r.pixel_mask.astype(int) for r in simple_wv.regions)
        assert np.all(combined == 1)

    def test_owner_returns_valid_index(self, simple_wv):
        idx = simple_wv.owner(0.5, 0.5)
        assert 0 <= idx < len(simple_wv.points)

    def test_owner_outside_domain_raises(self, simple_wv):
        with pytest.raises(ValueError):
            simple_wv.owner(999, 999)

    def test_region_of(self, simple_wv):
        r = simple_wv.region_of(0.5, 0.5)
        assert isinstance(r, VoronoiRegion)

    def test_nearest_generators(self, simple_wv):
        result = simple_wv.nearest_generators(0.5, 0.5, k=2)
        assert len(result) == 2
        assert result[0][0] <= result[1][0]  # sorted ascending

    def test_all_modes_compute(self):
        pts, w = random_generators(8, seed=10)
        for mode in ["multiplicative", "additive", "power"]:
            wv = WeightedVoronoi(pts, w, mode=mode, resolution=32)
            wv.compute()
            assert wv.label_grid is not None

    def test_invalid_mode_raises(self):
        pts, w = random_generators(5, seed=0)
        with pytest.raises(ValueError):
            WeightedVoronoi(pts, w, mode="euclidean")

    def test_invalid_points_shape_raises(self):
        with pytest.raises(ValueError):
            WeightedVoronoi(np.ones((5, 3)), np.ones(5))

    def test_invalid_weights_length_raises(self):
        with pytest.raises(ValueError):
            WeightedVoronoi(np.ones((5, 2)), np.ones(3))

    def test_require_computed_raises(self):
        pts, w = random_generators(5, seed=0)
        wv = WeightedVoronoi(pts, w)
        with pytest.raises(RuntimeError):
            wv.owner(0.5, 0.5)

    def test_to_label_array(self, simple_wv):
        arr = simple_wv.to_label_array()
        assert arr.shape == simple_wv.label_grid.shape
        # Should be a copy
        arr[0, 0] = -999
        assert simple_wv.label_grid[0, 0] != -999

    def test_plot_returns_fig_ax(self, simple_wv):
        import matplotlib.pyplot as plt
        fig, ax = simple_wv.plot()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_distance_field(self, simple_wv):
        import matplotlib.pyplot as plt
        fig, ax = simple_wv.plot_distance_field()
        assert fig is not None
        plt.close(fig)

    def test_to_csv(self, simple_wv, tmp_path):
        path = str(tmp_path / "out.csv")
        simple_wv.to_csv(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 5  # header + 4 regions

    def test_to_png(self, simple_wv, tmp_path):
        path = str(tmp_path / "out.png")
        simple_wv.to_png(path)
        import os
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


# ================================================================== #
#  VoronoiRegion tests                                                 #
# ================================================================== #

class TestVoronoiRegion:

    def test_centroid_inside_mask(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True
        r = VoronoiRegion(
            index=0,
            generator=np.array([0.5, 0.5]),
            weight=1.0,
            pixel_mask=mask,
        )
        cx, cy = r.centroid
        assert 40 <= cx <= 59
        assert 40 <= cy <= 59

    def test_area(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:20, 10:20] = True
        r = VoronoiRegion(0, np.array([0.5, 0.5]), 1.0, mask)
        assert r.area == 100

    def test_repr(self):
        mask = np.zeros((10, 10), dtype=bool)
        r = VoronoiRegion(0, np.array([0.5, 0.5]), 1.0, mask)
        assert "VoronoiRegion" in repr(r)
