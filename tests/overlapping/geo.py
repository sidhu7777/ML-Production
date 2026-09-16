"""Geometry helpers: distances, angles, polygon orientation and the analysis grid."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shapely
from shapely import wkt as shapely_wkt
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union

EARTH_RADIUS_M = 6371000.0
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQUATOR = 111_320.0


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.asarray(v, dtype=float) for v in (lat1, lon1, lat2, lon2))
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = (
        np.sin((phi2 - phi1) / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(np.radians(lon2 - lon1) / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def angle_diff_deg(a, b) -> np.ndarray:
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 360.0
    return np.minimum(d, 360.0 - d)


def circular_mean_deg(values) -> float:
    r = np.radians(np.asarray(values, dtype=float))
    return float((math.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean())) + 360.0) % 360.0)


@dataclass(frozen=True)
class LocalProjection:
    """Equirectangular metres around (lat0, lon0); accurate to <0.1% over a few km."""

    lat0: float
    lon0: float

    @property
    def m_per_deg_lon(self) -> float:
        return M_PER_DEG_LON_EQUATOR * math.cos(math.radians(self.lat0))

    def to_xy(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        x = (np.asarray(lon, dtype=float) - self.lon0) * self.m_per_deg_lon
        y = (np.asarray(lat, dtype=float) - self.lat0) * M_PER_DEG_LAT
        return x, y

    def to_latlon(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        lat = self.lat0 + np.asarray(y, dtype=float) / M_PER_DEG_LAT
        lon = self.lon0 + np.asarray(x, dtype=float) / self.m_per_deg_lon
        return lat, lon

    def geometry_to_xy(self, geom_lonlat: BaseGeometry) -> BaseGeometry:
        return transform(lambda x, y, z=None: self.to_xy(y, x), geom_lonlat)


def swap_xy(geom: BaseGeometry) -> BaseGeometry:
    return transform(lambda x, y, z=None: (y, x), geom)


def parse_geometry(value) -> BaseGeometry | None:
    """WKT or GeoJSON text -> valid shapely geometry (None when unusable)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        geom = shape(json.loads(text_value)) if text_value.startswith("{") else shapely_wkt.loads(text_value)
    except Exception:
        return None
    if geom.is_empty:
        return None
    return geom if geom.is_valid else shapely.make_valid(geom)


def orient_lonlat(geom: BaseGeometry, ref_lat, ref_lon) -> tuple[BaseGeometry, bool]:
    """
    Return the geometry in (lon, lat) order. Project polygons are stored as "lat lon" in
    project 193 while other sources use "lon lat"; production retries with swapped coordinates
    when a filter matches nothing. Here the order that contains more reference points (the
    project's sites) wins, so the choice is made from the data, not assumed.
    """
    ref_lat = np.asarray(ref_lat, dtype=float)
    ref_lon = np.asarray(ref_lon, dtype=float)
    as_is = int(shapely.contains_xy(geom, ref_lon, ref_lat).sum()) if ref_lat.size else 0
    swapped_geom = swap_xy(geom)
    swapped = int(shapely.contains_xy(swapped_geom, ref_lon, ref_lat).sum()) if ref_lat.size else 0
    if swapped > as_is:
        return swapped_geom, True
    if swapped == as_is == 0:
        minx, miny, maxx, maxy = geom.bounds
        x_is_lat = max(abs(minx), abs(maxx)) <= 90.0 < max(abs(miny), abs(maxy))
        if x_is_lat:
            return swapped_geom, True
    return geom, False


def orient_many_lonlat(geoms: list[BaseGeometry], area_lonlat: BaseGeometry) -> tuple[list[BaseGeometry], bool]:
    """Orient small geometries (buildings) by how many of their centroids fall in the area."""
    if not geoms:
        return geoms, False
    cx = np.array([g.centroid.x for g in geoms])
    cy = np.array([g.centroid.y for g in geoms])
    as_is = int(shapely.contains_xy(area_lonlat, cx, cy).sum())
    swapped = int(shapely.contains_xy(area_lonlat, cy, cx).sum())
    if swapped > as_is:
        return [swap_xy(g) for g in geoms], True
    return geoms, False


def union_area(wkts) -> BaseGeometry:
    geoms = [g for g in (parse_geometry(w) for w in wkts) if g is not None and g.area > 0]
    if not geoms:
        raise ValueError("No usable project polygon")
    return unary_union(geoms)


def build_grid(area_lonlat: BaseGeometry, resolution_m: float, buffer_m: float) -> tuple[pd.DataFrame, LocalProjection]:
    """Regular grid over the polygon buffered by buffer_m; `in_polygon` marks the original area."""
    centroid = area_lonlat.centroid
    proj = LocalProjection(lat0=centroid.y, lon0=centroid.x)
    area_xy = proj.geometry_to_xy(area_lonlat)
    eval_xy = area_xy.buffer(buffer_m) if buffer_m > 0 else area_xy
    minx, miny, maxx, maxy = eval_xy.bounds
    xs = np.arange(math.floor(minx / resolution_m), math.ceil(maxx / resolution_m) + 1) * resolution_m
    ys = np.arange(math.floor(miny / resolution_m), math.ceil(maxy / resolution_m) + 1) * resolution_m
    col_idx, row_idx = np.meshgrid(np.arange(xs.size), np.arange(ys.size))
    x = xs[col_idx.ravel()]
    y = ys[row_idx.ravel()]
    inside = shapely.contains_xy(eval_xy, x, y)
    x, y = x[inside], y[inside]
    lat, lon = proj.to_latlon(x, y)
    grid = pd.DataFrame(
        {
            "lat": lat,
            "lon": lon,
            "x_m": x,
            "y_m": y,
            "row": row_idx.ravel()[inside].astype(int),
            "col": col_idx.ravel()[inside].astype(int),
            "in_polygon": shapely.contains_xy(area_xy, x, y),
        }
    )
    grid.insert(0, "point_id", np.arange(len(grid)))
    return grid, proj
