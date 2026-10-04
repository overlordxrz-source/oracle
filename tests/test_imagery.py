"""Offline tests for reading/rendering and vessel detection, on synthetic rasters."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from oracle.detect import Detection, _find_targets, _hull_like, _otsu
from oracle.geo import AOI
from oracle.imagery import Grid, colorize, read_href, read_render
from oracle.models import Render

UTM48N = "EPSG:32648"


def write_tif(path, data, x0=370000.0, y0=140000.0, res=10.0, crs=UTM48N, nodata=0):
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=from_origin(x0, y0, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return str(path)


def test_read_href_resamples_onto_grid(tmp_path):
    src = np.full((200, 200), 100, np.uint8)
    src[:, 100:] = 200
    href = write_tif(tmp_path / "a.tif", src)
    grid = Grid(rasterio.crs.CRS.from_string(UTM48N), from_origin(370000, 140000, 20, 20), 100, 100)
    out = read_href(href, grid, [1])
    assert out.shape == (1, 100, 100)
    assert out[0, 50, 10] == pytest.approx(100) and out[0, 50, 90] == pytest.approx(200)


def test_mosaic_first_valid_pixel_wins(tmp_path):
    left = np.zeros((3, 100, 100), np.uint8)
    left[:, :, :50] = 50  # right half is nodata
    right = np.full((3, 100, 100), 150, np.uint8)
    a, b = write_tif(tmp_path / "l.tif", left), write_tif(tmp_path / "r.tif", right)
    grid = Grid(rasterio.crs.CRS.from_string(UTM48N), from_origin(370000, 140000, 10, 10), 100, 100)
    raw = read_render(Render(kind="rgb8", hrefs=[a, b]), grid)
    assert raw[0, 50, 20] == pytest.approx(50) and raw[0, 50, 80] == pytest.approx(150)
    rgba = colorize(Render(kind="rgb8", hrefs=[a]), raw)
    assert rgba.shape == (100, 100, 4) and rgba[..., 3].min() == 255


def test_colorize_sar_and_nodata_alpha():
    raw = np.array([[[np.nan, 0.001, 0.1, 1.0]]], np.float32)
    rgba = colorize(Render(kind="sar", hrefs=["x"], power=True, vmin=-30, vmax=0), raw)
    assert rgba[0, 0, 3] == 0 and rgba[0, 1, 3] == 255
    assert rgba[0, 1, 0] < rgba[0, 2, 0] < rgba[0, 3, 0]


def test_grid_for_aoi_caps_pixels():
    g = Grid.for_aoi(AOI.from_point(1.2, 103.8, 10), res=0.3, max_pixels=2048)
    assert max(g.width, g.height) <= 2049 and g.res == pytest.approx(20000 / 2048, rel=0.02)


# --------------------------------------------------------------------------- detection


def _sea_scene(n=400, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.normal(0.02, 0.002, (n, n)).astype(np.float32)
    sea = np.ones((n, n), bool)
    grid = Grid(rasterio.crs.CRS.from_string(UTM48N), from_origin(370000, 140000, 10, 10), n, n)
    return values, sea, grid


def test_detects_elongated_hull_with_length_and_heading():
    values, sea, grid = _sea_scene()
    values[200, 150:183] = 0.25  # 33 px east-west line: a ~330 m hull
    values[201, 150:183] = 0.25
    land = np.zeros_like(sea)
    dets = _find_targets(values, sea, land, land, grid, k=6, min_contrast=0.03, sigma_m=400, min_area_m2=200)
    assert len(dets) == 1
    d = dets[0]
    assert d.length_m == pytest.approx(330, abs=20)
    assert d.width_m < 40
    assert d.heading_deg == pytest.approx(90, abs=5)  # east-west axis
    assert not d.near_shore


def test_ignores_noise_and_marks_shore_contacts():
    values, sea, grid = _sea_scene(seed=1)
    land = np.zeros_like(sea)
    land[:, :20] = True
    sea[:, :20] = False
    values[100, 22:30] = 0.3  # pier-like blob touching land
    dets = _find_targets(values, sea, land, np.zeros_like(sea), grid, k=6, min_contrast=0.03, sigma_m=400, min_area_m2=200)
    assert len(dets) == 1 and dets[0].near_shore


def _det(length, width, contrast, excess=None):
    return Detection(0, 0, 0, 0, length, width, 0, length * width, contrast, False, excess=excess)


def test_hull_filter_rules():
    hull = {"blue": 0.04, "red": 0.15, "nir": 0.2}
    cloud = {"blue": 0.07, "red": 0.07, "nir": 0.065}
    assert _hull_like(_det(330, 70, 80, hull))
    assert not _hull_like(_det(400, 280, 30, hull))  # fat: fish trap / cloud puff
    assert not _hull_like(_det(220, 80, 15, cloud))  # faint, flat, puffy -> cloud
    assert _hull_like(_det(150, 30, 20, cloud))  # flat but slender: grey warship / white boat
    assert _hull_like(_det(476, 55, 87, cloud))  # bright flat container ship
    assert not _hull_like(_det(280, 70, 9, {"blue": -0.07, "red": -0.09, "nir": 0.3}))  # mudflat
    assert _hull_like(_det(200, 50, 20))  # SAR: no spectral info, shape only


def test_otsu_separates_land_and_water():
    vals = np.concatenate([np.random.default_rng(0).normal(-20, 1, 500), np.random.default_rng(1).normal(-6, 1, 500)])
    thr, sep = _otsu(vals)
    assert -17 < thr < -9 and sep > 10
