"""Phase 43 v3 experimental geospatial fetch.

This stays outside production. It preserves the production Overture layer
inputs and clipping logic, but makes the cold Overture fetch less wasteful than
production without overloading the same process before physical scoring.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

import geopandas as gpd

from tools.lte_prediction_offset import geo_inputs as prod_geo


def _fetch_and_clip(kind: str, bbox: tuple[float, float, float, float], polygon) -> tuple[str, gpd.GeoDataFrame]:
    try:
        import overturemaps.core as overture
    except ImportError as exc:
        raise RuntimeError("overturemaps is required to build a Phase-27 clutter cache") from exc

    print(
        f"[LTE_OFFSET][PHASE43_V3_OVERTURE_FETCH] layer={kind} state=start "
        f"connect_timeout_s={prod_geo.OVERTURE_CONNECT_TIMEOUT_S} "
        f"request_timeout_s={prod_geo.OVERTURE_REQUEST_TIMEOUT_S}",
        flush=True,
    )
    layer = overture.geodataframe(
        kind,
        bbox=bbox,
        connect_timeout=prod_geo.OVERTURE_CONNECT_TIMEOUT_S,
        request_timeout=prod_geo.OVERTURE_REQUEST_TIMEOUT_S,
    )
    if layer.crs is None:
        layer = layer.set_crs("EPSG:4326")
    clipped = gpd.clip(layer, polygon)
    out = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()
    del layer, clipped
    gc.collect()
    print(f"[LTE_OFFSET][PHASE43_V3_OVERTURE_FETCH] layer={kind} state=done rows={len(out)}", flush=True)
    return kind, out


def fetch_overture_context_phase43_v3(grid: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    """Fetch Overture context with bounded parallelism.

    The previous v3 fetched all four layers at once. That reduced the network
    wait, but raised peak memory and made the later physical scorer slower. This
    version fetches the heavy road segment layer by itself, then fetches the
    three lighter polygon layers with two threads. The output contract and layer
    order remain production-equivalent.
    """
    polygon = grid.geometry.union_all()
    bbox = polygon.bounds
    layers: dict[str, gpd.GeoDataFrame] = {}

    kind, layer = _fetch_and_clip("segment", bbox, polygon)
    layers[kind] = layer

    light_kinds = ("water", "land_cover", "land_use")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_fetch_and_clip, kind, bbox, polygon) for kind in light_kinds]
        for future in as_completed(futures):
            kind, layer = future.result()
            layers[kind] = layer

    gc.collect()
    return {kind: layers[kind] for kind in ("segment", "water", "land_cover", "land_use")}
