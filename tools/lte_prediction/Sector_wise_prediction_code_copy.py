#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import atexit
import math
import os
import time
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape, Polygon
from shapely import wkt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import multiprocessing as mp
import sys
import datetime

# ==========================================================
# TERMINAL LOGGER
# ==========================================================
class DualLogger(object):
    """Writes output to both the terminal and a log file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ==========================================================
# GLOBAL CONSTANTS
# ==========================================================

KPI_RANGES = {
    "RSRP": (-140, -44),
    "RSRQ": (-20, -3),
    "SINR": (-10, 30),
}

BAND_TO_FREQ = {
    1: 2100, 3: 1800, 5: 850,
    8: 900, 20: 800, 28: 700,
}

_PREDICTION_POOL = None
_PREDICTION_POOL_WORKERS = None
_WORKER_SHARED_CONTEXT = None
_WORKER_POLYGONS = None
_WORKER_META = None


def _context_signature(shared_context):
    if not shared_context:
        return None
    polygons = shared_context.get("polygons")
    meta = shared_context.get("meta")
    return (
        len(polygons) if polygons is not None else 0,
        len(meta) if meta is not None else 0,
    )


def _init_prediction_worker(shared_context=None):
    global _WORKER_SHARED_CONTEXT, _WORKER_POLYGONS, _WORKER_META
    _WORKER_SHARED_CONTEXT = shared_context or {}
    _WORKER_POLYGONS = _WORKER_SHARED_CONTEXT.get("polygons")
    _WORKER_META = _WORKER_SHARED_CONTEXT.get("meta")


def _close_prediction_pool():
    global _PREDICTION_POOL, _PREDICTION_POOL_WORKERS, _WORKER_SHARED_CONTEXT
    if _PREDICTION_POOL is not None:
        _PREDICTION_POOL.close()
        _PREDICTION_POOL.join()
        _PREDICTION_POOL = None
        _PREDICTION_POOL_WORKERS = None
        _WORKER_SHARED_CONTEXT = None


def get_prediction_pool(n_workers, shared_context=None):
    global _PREDICTION_POOL, _PREDICTION_POOL_WORKERS, _WORKER_SHARED_CONTEXT
    shared_signature = _context_signature(shared_context)
    current_signature = _context_signature(_WORKER_SHARED_CONTEXT)
    if (
        _PREDICTION_POOL is None
        or _PREDICTION_POOL_WORKERS != n_workers
        or shared_signature != current_signature
    ):
        _close_prediction_pool()
        _PREDICTION_POOL = mp.Pool(
            n_workers,
            initializer=_init_prediction_worker,
            initargs=(shared_context,),
        )
        _PREDICTION_POOL_WORKERS = n_workers
        _WORKER_SHARED_CONTEXT = shared_context or {}
        print(f"[LTE][POOL] initialized_shared_pool workers={n_workers}")
    return _PREDICTION_POOL


atexit.register(_close_prediction_pool)

# ==========================================================
# GEO MATH FUNCTIONS (VECTORIZED)
# ==========================================================

def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def compute_bearing_vectorized(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    bearing = np.arctan2(x, y)
    return (np.degrees(bearing) + 360) % 360


def select_nearest_site_rows(site_df, serving_site_rows, limit):
    """
    Keep the serving rows plus the geographically nearest site rows.
    This preserves the existing prediction logic while reducing the
    interference candidate set from the full dataset.
    """
    if limit is None or limit <= 0 or len(site_df) <= limit:
        return site_df.to_dict("records")

    clat = float(serving_site_rows["lat"].mean())
    clon = float(serving_site_rows["lon"].mean())
    site_lat = site_df["lat"].to_numpy(dtype=float, copy=False)
    site_lon = site_df["lon"].to_numpy(dtype=float, copy=False)
    distances = haversine_vectorized(
        clat,
        clon,
        site_lat,
        site_lon,
    )
    nearest_idx = np.argpartition(distances, min(limit, len(distances) - 1))[:limit]
    nearest_df = site_df.iloc[nearest_idx]

    serving_ids = set(serving_site_rows["Node_Cell_ID"].astype(str))
    serving_df = site_df[
        site_df["Node_Cell_ID"].astype(str).isin(serving_ids)
    ]

    combined_df = (
        pd.concat([nearest_df, serving_df], ignore_index=True)
        .drop_duplicates(subset=["Node_Cell_ID"], keep="first")
    )
    return combined_df.to_dict("records")

# ==========================================================
# 3GPP ANTENNA MODEL (VECTORIZED)
# ==========================================================

def compute_3gpp_antenna_gain_vectorized(az_diff, elev_diff, max_gain=18.0,
                                         h_beamwidth=65.0, v_beamwidth=6.0,
                                         a_max=30.0, sla_v=20.0):
    ah = np.minimum(12.0 * (az_diff / h_beamwidth)**2, a_max)
    av = np.minimum(12.0 * (elev_diff / v_beamwidth)**2, sla_v)
    total_attenuation = np.minimum(ah + av, a_max)
    return max_gain - total_attenuation

# ==========================================================
# SINGLE SECTOR PATH LOSS + RSRP HELPER
# ==========================================================

def compute_sector_rsrp(site, p_lat, p_lon, freq, params):
    """
    Compute RSRP contribution of a single sector row to a point.
    Returns sector_rsrp in dBm (float).
    """
    s_lat       = site['lat']
    s_lon       = site['lon']
    s_az        = site['azimuth']
    s_etilt     = site['electrical_tilt']
    s_mtilt     = site['mechanical_tilt']
    s_htx       = site['antenna_height']
    tx_pwr      = site['tx_power']
    h_rx        = params.get('ue_height', 1.5)
    k1          = params.get('k1', 0)
    k2          = params.get('k2', 0)

    d_m  = haversine_vectorized(s_lat, s_lon, p_lat, p_lon)
    d_m  = max(d_m, 1.0)
    d_km = d_m / 1000.0

    # ---------- COST-231 Hata path loss ----------
    a_hm = (1.1 * np.log10(freq) - 0.7) * h_rx - (1.56 * np.log10(freq) - 0.8)
    CM   = 3.0
    if k1 and k1 != 0:
        base_PL    = k1
        slope_term = k2
    else:
        base_PL    = 46.3 + 33.9 * np.log10(freq) - 13.82 * np.log10(s_htx) - a_hm + CM
        slope_term = 44.9 - 6.55 * np.log10(s_htx)

    pathloss = base_PL + slope_term * np.log10(d_km)

    # ---------- 3GPP antenna gain ----------
    bearing      = compute_bearing_vectorized(s_lat, s_lon, p_lat, p_lon)
    az_diff      = (bearing - s_az + 180) % 360 - 180
    elev_angle   = np.degrees(np.arctan2(h_rx - s_htx, d_m))
    total_tilt   = s_etilt + s_mtilt
    elev_diff    = elev_angle + total_tilt
    gain_3gpp    = compute_3gpp_antenna_gain_vectorized(
                       az_diff, elev_diff, params.get('antenna_gain', 18.0))

    sector_rsrp = tx_pwr + gain_3gpp - pathloss - params.get('cable_loss', 2)
    return sector_rsrp

# ==========================================================
# PREDICTION CORE  ← KEY FIX: uses ALL sites for interference
# ==========================================================

def process_chunk_3gpp_antenna(args):
    test_pts, serving_site_rows, params = args

    rsrp_list, rsrq_list, sinr_list = [], [], []

    freq = params.get('frequency_mhz', 1800)
    bw = params.get('bandwidth_mhz', 10)

    rb_map = {1.4: 6, 3: 15, 5: 25, 10: 50, 15: 75, 20: 100}
    n_rb = rb_map.get(bw, 50)

    noise_linear = 10 ** (-104.0 / 10.0)

    all_sites = params.get('all_sites_rows', serving_site_rows)

    #  PRE-COMPUTE (VERY IMPORTANT)
    sites_array = all_sites   # already a list
    serving_ids = set(serving_site_rows['Node_Cell_ID'].astype(str))

    for pt in test_pts:
        p_lat = pt['lat']
        p_lon = pt['lon']

        iloss = detect_indoor(
            p_lat, p_lon,
            params.get("polygons"),
            params.get("meta")
        )

        best_rsrp = -150.0
        total_power_linear = 0.0

        for site in sites_array:
            site_freq = site.get('frequency_mhz', freq)

            sec_rsrp = compute_sector_rsrp(
                site, p_lat, p_lon, site_freq, params
            ) - iloss

            sec_linear = np.power(10, sec_rsrp / 10.0)
            total_power_linear += sec_linear

            if site['Node_Cell_ID'] in serving_ids:
                if sec_rsrp > best_rsrp:
                    best_rsrp = sec_rsrp

        if best_rsrp > -150.0:
            best_rsrp_linear = np.power(10, best_rsrp / 10.0)

            interference_linear = total_power_linear - best_rsrp_linear + noise_linear
            interference_linear = max(interference_linear, noise_linear)

            sinr_linear = best_rsrp_linear / interference_linear
            best_sinr = 10 * math.log10(sinr_linear)

            rssi_linear = total_power_linear + noise_linear
            rssi_dbm = 10 * math.log10(rssi_linear)

            best_rsrq = best_rsrp - rssi_dbm + 10 * math.log10(n_rb)
        else:
            best_rsrq = -20.0
            best_sinr = -10.0

        rsrp_list.append(best_rsrp)
        rsrq_list.append(best_rsrq)
        sinr_list.append(best_sinr)

    return rsrp_list, rsrq_list, sinr_list

# ==========================================================
# SAFE CSV LOADER
# ==========================================================

def safe_read(path):
    configs = [
        {},
        {"encoding": "latin-1"},
        {"sep": ";"},
        {"sep": "\t"},
        {"encoding": "utf-8", "engine": "python"},
    ]
    for kw in configs:
        try:
            return pd.read_csv(path, **kw)
        except:
            pass
    raise RuntimeError(f"Cannot read CSV: {path}")

def normcols(df):
    df.columns = [c.strip() for c in df.columns]
    return df

def standardize_latlon(df):
    if "lat" in df.columns and "lon" in df.columns:
        return df

    mapping = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("latitude", "lat_deg", "y") and "lat" not in mapping.values():
            mapping[c] = "lat"
        if lc in ("longitude", "lon_deg", "x", "long") and "lon" not in mapping.values():
            mapping[c] = "lon"

    return df.rename(columns=mapping)

# ==========================================================
# LOAD AREA POLYGON
# ==========================================================

def load_polygon_file(path):
    if path.lower().endswith(".json"):
        data = json.load(open(path))
        if "type" in data and data["type"] == "Polygon":
            return shape(data)
        if "polygon" in data:
            return Polygon(data["polygon"])
        raise ValueError("Unrecognized JSON polygon format.")

    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        wcol = None
        for c in ("wkt", "polygon", "geometry"):
            if c in df.columns:
                wcol = c
                break
        if wcol is None:
            raise ValueError("No WKT column found in polygon CSV.")
        w = df[wcol].dropna().iloc[0]
        return wkt.loads(w)

    raise ValueError("Unsupported polygon file.")

# ==========================================================
# LOAD BUILDING POLYGONS
# ==========================================================

def load_building_polygons(path):
    import re
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    geom_col = None
    for candidate in ("geometry_wkt", "geometry", "region"):
        if candidate in df.columns:
            geom_col = candidate
            break
    if geom_col is None:
        raise ValueError(" No supported geometry column found in building CSV")

    polygons = []
    meta     = []

    def swap_latlon(wkt_str):
        pattern = r"POLYGON\s*\(\((.*?)\)\)"
        match   = re.search(pattern, wkt_str, re.DOTALL | re.IGNORECASE)
        if not match:
            return None
        coords  = match.group(1).split(",")
        swapped = []
        for c in coords:
            parts = c.strip().split()
            if len(parts) == 2:
                lat, lon = parts
                swapped.append(f"{lon} {lat}")
        return "POLYGON((" + ",".join(swapped) + "))"

    def _load_polygon(raw_value):
        raw_text = str(raw_value).strip()
        if raw_text.lower() in ("nan", "none", ""):
            return None

        candidates = [raw_text]
        swapped = swap_latlon(raw_text)
        if swapped and swapped != raw_text:
            candidates.append(swapped)

        for candidate in candidates:
            try:
                geom = wkt.loads(candidate)
            except Exception:
                continue

            if geom.is_empty:
                continue

            if geom.geom_type == "Polygon":
                return geom if geom.is_valid else geom.buffer(0)
            if geom.geom_type == "MultiPolygon":
                pieces = list(geom.geoms)
                if not pieces:
                    continue
                poly = max(pieces, key=lambda g: g.area)
                return poly if poly.is_valid else poly.buffer(0)
        return None

    skipped = 0
    for _, row in df.iterrows():
        poly = _load_polygon(row[geom_col])
        if poly is None or poly.is_empty or not poly.is_valid:
            skipped += 1
            continue

        polygons.append(poly)
        minx, miny, maxx, maxy = poly.bounds
        meta.append({
            "loss": 15.0,
            "min_lon": float(minx),
            "min_lat": float(miny),
            "max_lon": float(maxx),
            "max_lat": float(maxy),
        })

    print(f"[OK] Loaded {len(polygons)} building polygons (skipped {skipped})")
    return polygons, meta

def detect_indoor(lat, lon, polygons, meta):
    if polygons is None:
        polygons = _WORKER_POLYGONS
        meta = _WORKER_META
    if polygons is None:
        return 0
    pt = Point(lon, lat)
    for poly, m in zip(polygons, meta):
        if not (
            m["min_lon"] <= lon <= m["max_lon"]
            and m["min_lat"] <= lat <= m["max_lat"]
        ):
            continue
        if poly.contains(pt):
            return m["loss"]
    return 0

# ==========================================================
# SMART COST231 CALIBRATION
# ==========================================================

def calibrate_site(drive_df, site_rows, tx, gain, closs, freq):
    """
    FIX: freq is now passed in (was implicitly 1800 before).
    Returns (K1, K2) — use k1=0,k2=0 to fall back to pure COST-231.
    """
    rcol = next((c for c in drive_df.columns if "rsrp" in c.lower()), None)
    if rcol is None:
        return 0, 0   # ← FIX: was (169, 35.2) — now triggers COST-231 formula

    dt = drive_df.dropna(subset=["lat", "lon"]).copy()
    dt["RSRP_meas"] = pd.to_numeric(dt[rcol], errors="coerce")
    dt = dt.dropna(subset=["RSRP_meas"])
    dt = dt[(dt["RSRP_meas"] >= -150) & (dt["RSRP_meas"] <= -30)]
    print(
        f"[LTE][CALIBRATION] serving_rows={len(site_rows)} valid_dt_rows={len(dt)} "
        f"tx={tx} gain={gain} cable_loss={closs} freq_mhz={freq}"
    )

    if len(dt) < 10:
        return 0, 0   # ← FIX: was (169, 35.2)

    dmat = np.zeros((len(dt), len(site_rows)))
    for i in range(len(dt)):
        for j in range(len(site_rows)):
            dmat[i, j] = haversine_vectorized(
                dt["lat"].values[i], dt["lon"].values[i],
                site_rows["lat"].values[j], site_rows["lon"].values[j]
            )

    serving     = dmat.min(axis=1)
    serving     = np.maximum(serving, 1.0)
    dkm         = serving / 1000.0
    spread_km   = dkm.max() - dkm.min()

    PLm  = tx + gain - closs - dt["RSRP_meas"].values
    logd = np.log10(dkm)

    DEFAULT_K2 = 35.2
    if spread_km < 0.5:
        K2 = DEFAULT_K2
        K1 = float(np.mean(PLm - K2 * logd))
    elif spread_km < 2.0:
        K2, K1 = np.polyfit(logd, PLm, 1)
        K2 = float(np.clip(K2, 25, 45))
    else:
        K2, K1 = np.polyfit(logd, PLm, 1)

    K1 = float(np.clip(K1, 120, 170))
    K2 = float(np.clip(K2, 20, 60))
    print(
        f"[LTE][CALIBRATION] calibrated_k1={K1:.4f} calibrated_k2={K2:.4f} "
        f"distance_km_range={dkm.min():.4f}..{dkm.max():.4f} spread_km={spread_km:.4f}"
    )
    return K1, K2

# ==========================================================
# GRID GENERATION
# ==========================================================

def generate_grid(site_rows, radius_m=5000, res=25):
    clat = site_rows["lat"].mean()
    clon = site_rows["lon"].mean()
    print(
        f"[LTE][GRID] center_lat={clat:.6f} center_lon={clon:.6f} "
        f"radius_m={radius_m} resolution_m={res} serving_rows={len(site_rows)}"
    )

    lat_step = res / 111320
    lon_step = res / (111320 * math.cos(math.radians(clat)))

    lat_min = clat - (radius_m / 111320)
    lat_max = clat + (radius_m / 111320)
    lon_min = clon - (radius_m / (111320 * math.cos(math.radians(clat))))
    lon_max = clon + (radius_m / (111320 * math.cos(math.radians(clat))))

    lat_range = np.arange(lat_min, lat_max, lat_step)
    lon_range = np.arange(lon_min, lon_max, lon_step)

    LAT, LON = np.meshgrid(lat_range, lon_range, indexing="ij")
    latf = LAT.flatten()
    lonf = LON.flatten()

    d = haversine_vectorized(clat, clon, latf, lonf)
    mask = d <= radius_m

    df = pd.DataFrame({"lat": latf[mask], "lon": lonf[mask]})
    print(
        f"[LTE][GRID] generated_points={len(df)} "
        f"lat_range={df['lat'].min():.6f}..{df['lat'].max():.6f} "
        f"lon_range={df['lon'].min():.6f}..{df['lon'].max():.6f}"
    )
    return df

# ==========================================================
# PARALLEL PROCESSING
# ==========================================================

def compute_predictions_parallel(test_pts, serving_site_rows, params, n_workers=None, pool=None, use_shared_pool=False):

    if n_workers is None or n_workers < 1:
        n_workers = max(1, mp.cpu_count() - 1)

    print(f" Using {n_workers} CPU cores for {len(test_pts)} points")

    print(
        f"[LTE][PATHLOSS] serving_rows={len(serving_site_rows)} "
        f"interference_site_rows={len(params.get('all_sites_rows', serving_site_rows))} "
        f"freq_mhz={params.get('frequency_mhz')} bandwidth_mhz={params.get('bandwidth_mhz')} "
        f"k1={params.get('k1')} k2={params.get('k2')}"
    )
    total_points = len(test_pts)

    #  SMART CHUNKING
    chunk_divisor = max(1, n_workers * 2)
    chunk_size = max(25, math.ceil(total_points / chunk_divisor))
    print(f"[LTE][PATHLOSS] total_points={total_points} chunk_size={chunk_size}")

    chunks = []
    if pool is not None or use_shared_pool:
        chunk_params = {k: v for k, v in params.items() if k not in ("polygons", "meta")}
    else:
        chunk_params = params
    for i in range(0, total_points, chunk_size):
        sub = test_pts.iloc[i:i+chunk_size]
        rows = sub.to_dict("records")
        chunks.append((rows, serving_site_rows, chunk_params))

    all_rsrp, all_rsrq, all_sinr = [], [], []

    #  FAST MULTIPROCESSING
    if pool is None and use_shared_pool:
        pool = get_prediction_pool(
            n_workers,
            shared_context={
                "polygons": params.get("polygons"),
                "meta": params.get("meta"),
            },
        )

    if pool is None:
        with mp.Pool(n_workers) as local_pool:
            results = local_pool.map(process_chunk_3gpp_antenna, chunks)
    else:
        results = pool.map(process_chunk_3gpp_antenna, chunks)

    for r, q, s in results:
        all_rsrp.extend(r)
        all_rsrq.extend(q)
        all_sinr.extend(s)

    return (
        np.array(all_rsrp),
        np.array(all_rsrq),
        np.array(all_sinr)
    )

# ==========================================================
# ACCURACY REPORT
# ==========================================================

def generate_accuracy_report(drive_df, site_df, params, pool=None):
    print("\n" + "="*60)
    print("GAUGE ACCURACY: MEASURED VS PREDICTED")
    print("="*60)

    dt   = drive_df.dropna(subset=["lat", "lon"]).copy()
    rcol = next((c for c in dt.columns if "rsrp" in c.lower()), None)
    qcol = next((c for c in dt.columns if "rsrq" in c.lower()), None)
    scol = next((c for c in dt.columns if "sinr" in c.lower()), None)

    if rcol is None:
        print("[WARN] No valid RSRP column found for validation.")
        return

    dt["RSRP_meas"] = pd.to_numeric(dt[rcol], errors="coerce")
    dt = dt.dropna(subset=["RSRP_meas"])
    if len(dt) == 0:
        return

    print(f"Running point-to-point validation for {len(dt)} DT points...")

    # Use ALL sites for interference during accuracy check too
    rsrp_pred, rsrq_pred, sinr_pred = compute_predictions_parallel(
        dt, site_df, params, n_workers=params.get('n_workers', 5), pool=pool, use_shared_pool=(pool is None)
    )

    dt["RSRP_pred"] = rsrp_pred
    dt["RSRQ_pred"] = rsrq_pred
    dt["SINR_pred"] = sinr_pred
    print(
        f"[LTE][RF_ACCURACY] site_rows={len(site_df)} dt_rows={len(dt)} "
        f"pred_rsrp_range={dt['RSRP_pred'].min():.4f}..{dt['RSRP_pred'].max():.4f}"
    )

    # RSRP
    print(f"\n RSRP ACCURACY:")
    print(f"   MAE  : {mean_absolute_error(dt['RSRP_meas'], dt['RSRP_pred']):.2f} dB")
    print(f"   RMSE : {np.sqrt(mean_squared_error(dt['RSRP_meas'], dt['RSRP_pred'])):.2f} dB")
    print(f"   R2   : {r2_score(dt['RSRP_meas'], dt['RSRP_pred']):.4f}")

    # RSRQ
    if qcol:
        dt["RSRQ_meas"] = pd.to_numeric(dt[qcol], errors="coerce")
        vq = dt.dropna(subset=["RSRQ_meas", "RSRQ_pred"])
        if len(vq) > 0:
            print(f"\n RSRQ ACCURACY:")
            print(f"   MAE  : {mean_absolute_error(vq['RSRQ_meas'], vq['RSRQ_pred']):.2f} dB")
            print(f"   RMSE : {np.sqrt(mean_squared_error(vq['RSRQ_meas'], vq['RSRQ_pred'])):.2f} dB")
            print(f"   R2   : {r2_score(vq['RSRQ_meas'], vq['RSRQ_pred']):.4f}")
    else:
        print("\n[WARN] No RSRQ column found in DT file.")

    # SINR
    if scol:
        dt["SINR_meas"] = pd.to_numeric(dt[scol], errors="coerce")
        vs = dt.dropna(subset=["SINR_meas", "SINR_pred"])
        if len(vs) > 0:
            print(f"\n SINR ACCURACY:")
            print(f"   MAE  : {mean_absolute_error(vs['SINR_meas'], vs['SINR_pred']):.2f} dB")
            print(f"   RMSE : {np.sqrt(mean_squared_error(vs['SINR_meas'], vs['SINR_pred'])):.2f} dB")
            print(f"   R2   : {r2_score(vs['SINR_meas'], vs['SINR_pred']):.4f}")
    else:
        print("\n[WARN] No SINR column found in DT file.")

    print("\n" + "="*60 + "\n")

# ==========================================================
# MAIN
# ==========================================================

def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(args.outdir, f"run_log_{timestamp}.txt")
    sys.stdout   = DualLogger(log_filename)

    print("\n============================================================")
    print(" LTE PREDICTOR — 3GPP 3D ANTENNA + INTER-CELL INTERFERENCE")
    print(f" Cores: {args.n_workers}   Frequency: {args.frequency} MHz   BW: {args.bandwidth} MHz")
    print("============================================================\n")

    # ── 1. LOAD SITE DATA ───────────────────────────────────────
    site_df = safe_read(args.site)
    site_df = standardize_latlon(normcols(site_df))

    if "cell_id" not in site_df.columns:
        raise ValueError(" 'cell_id' column missing in site file.")

    site_df["Node_Cell_ID"] = site_df["cell_id"].astype(str)
    site_df = site_df.rename(columns={
        "Etilt": "electrical_tilt",
        "Mtilt": "mechanical_tilt",
        "Height": "antenna_height",
    })

    # FIX #3 ── Assign frequency from args, not hardcoded 1800
    if "frequency_mhz" not in site_df.columns:
        site_df["frequency_mhz"] = args.frequency

    # ── 2. LOAD DRIVE TEST ──────────────────────────────────────
    drive_df = None
    if args.drive:
        drive_df = safe_read(args.drive)
        drive_df = standardize_latlon(normcols(drive_df))
        if "cell_id" in drive_df.columns:
            drive_df["Node_Cell_ID"] = drive_df["cell_id"].astype(str)
        elif "combined_id" in drive_df.columns:
            drive_df["Node_Cell_ID"] = drive_df["combined_id"].astype(str)

    # ── 3. LOAD BUILDINGS ───────────────────────────────────────
    polygons, poly_meta = None, None
    if args.building:
        polygons, poly_meta = load_building_polygons(args.building)

    # ── 4. LOAD AREA POLYGON ────────────────────────────────────
    area_polygon = None
    if args.polygon_area:
        area_polygon = load_polygon_file(args.polygon_area)

    # ── 5. CALIBRATE K1/K2 PER CELL ────────────────────────────
    calibrated_params = {}
    unique_cells = site_df["Node_Cell_ID"].unique()
    drive_groups = (
        drive_df.groupby("Node_Cell_ID")
        if drive_df is not None and "Node_Cell_ID" in drive_df.columns
        else None
    )
    max_interference_sites = int(getattr(args, "max_interference_sites", 50))

    if args.calibrate and drive_df is not None:
        print(" Calibrating K1/K2 from drive-test data...")
        total_calibration_cells = len(unique_cells)
        calibrated_count = 0
        skipped_count = 0
        for idx, cid in enumerate(unique_cells, start=1):
            calibration_start = time.perf_counter()
            print(
                f"[LTE][CALIBRATION_PROGRESS] current={idx}/{total_calibration_cells} "
                f"cell={cid} calibrated_done={calibrated_count} skipped={skipped_count} "
                f"remaining={total_calibration_cells - idx}"
            )
            site_rows_tmp = site_df[site_df["Node_Cell_ID"] == cid].copy()
            if drive_groups is not None and cid in drive_groups.groups:
                cell_dt_tmp = drive_groups.get_group(cid).copy()
            else:
                cell_dt_tmp = drive_df.copy()

            rcol = next((c for c in cell_dt_tmp.columns if "rsrp" in c.lower()), None)
            if rcol and len(cell_dt_tmp) > 0:
                valid_dt = pd.to_numeric(cell_dt_tmp[rcol], errors="coerce").dropna()
                if len(valid_dt) >= 10:
                    # FIX: pass frequency to calibrate_site
                    cell_freq = site_rows_tmp["frequency_mhz"].iloc[0]
                    k1_tmp, k2_tmp = calibrate_site(
                        cell_dt_tmp, site_rows_tmp,
                        site_rows_tmp["tx_power"].iloc[0],
                        args.antenna_gain, args.cable_loss, cell_freq
                    )
                    clat = site_rows_tmp["lat"].mean()
                    clon = site_rows_tmp["lon"].mean()
                    calibrated_params[cid] = (k1_tmp, k2_tmp, clat, clon)
                    calibrated_count += 1
                    print(
                        f"[LTE][CALIBRATION_TIMING] cell={cid} "
                        f"elapsed_sec={time.perf_counter() - calibration_start:.2f} "
                        f"status=calibrated"
                    )
                else:
                    skipped_count += 1
                    print(
                        f"[LTE][CALIBRATION_TIMING] cell={cid} "
                        f"elapsed_sec={time.perf_counter() - calibration_start:.2f} "
                        f"status=skipped reason=insufficient_valid_rsrp"
                    )
            else:
                skipped_count += 1
                print(
                    f"[LTE][CALIBRATION_TIMING] cell={cid} "
                    f"elapsed_sec={time.perf_counter() - calibration_start:.2f} "
                    f"status=skipped reason=missing_dt_for_cell"
                )

        print(f"    {len(calibrated_params)} cells calibrated from DT.\n")

    # ── 6. RUN CELL-WISE PREDICTION ─────────────────────────────
    final_list = []
    prediction_points_df = getattr(args, "prediction_points_df", None)
    if isinstance(prediction_points_df, pd.DataFrame) and not prediction_points_df.empty:
        prediction_points_df = prediction_points_df.copy()
        prediction_points_df["Node_Cell_ID"] = prediction_points_df["Node_Cell_ID"].astype(str)
        print(
            f"[LTE][PREDICTION_POINTS_OVERRIDE] enabled=True rows={len(prediction_points_df)} "
            f"cells={prediction_points_df['Node_Cell_ID'].nunique()} "
            f"grids={prediction_points_df['frontend_grid_id'].nunique() if 'frontend_grid_id' in prediction_points_df.columns else 'n/a'}"
        )
    else:
        prediction_points_df = pd.DataFrame()

    total_prediction_cells = len(unique_cells)
    for idx, cid in enumerate(unique_cells, start=1):
        cell_start = time.perf_counter()
        print("----------------------------------------------------")
        print(f"Processing Cell → {cid}")
        print("----------------------------------------------------")

        print(
            f"[LTE][PREDICTION_PROGRESS] current={idx}/{total_prediction_cells} "
            f"cell={cid} completed={idx - 1} remaining={total_prediction_cells - idx}"
        )
        site_rows = site_df[site_df["Node_Cell_ID"] == cid].copy()
        cell_freq = site_rows["frequency_mhz"].iloc[0]
        interference_site_rows = select_nearest_site_rows(
            site_df, site_rows, max_interference_sites
        )
        print(
            f"[LTE][SITE_FILTER] cell={cid} serving_rows={len(site_rows)} "
            f"candidate_interference_sites={len(interference_site_rows)} "
            f"from_total_sites={len(site_df)} limit={max_interference_sites}"
        )

        # ── Determine K1/K2 ──
        if args.calibrate and drive_df is not None:
            if cid in calibrated_params:
                k1, k2, _, _ = calibrated_params[cid]
                print(f"    Calibrated K1={k1:.2f}  K2={k2:.2f}")
            elif len(calibrated_params) > 0:
                clat = site_rows["lat"].mean()
                clon = site_rows["lon"].mean()
                closest_cid = min(
                    calibrated_params,
                    key=lambda c: haversine_vectorized(
                        clat, clon, calibrated_params[c][2], calibrated_params[c][3])
                )
                k1, k2 = calibrated_params[closest_cid][:2]
                dist_km = haversine_vectorized(
                    clat, clon,
                    calibrated_params[closest_cid][2],
                    calibrated_params[closest_cid][3]) / 1000.0
                print(f"     Inheriting K1={k1:.2f} K2={k2:.2f} from {dist_km:.2f} km away.")
            else:
                # FIX #4 ── default = 0,0 → use COST-231 formula (was 135/169, 35.2)
                k1, k2 = 0, 0
                print(f"     No reference cells. Using COST-231 default formula.")
        else:
            # FIX #4 ── non-calibrate mode: also use COST-231 (was 169, 35.2)
            k1, k2 = 0, 0
            print(f"   Non-calibrate mode: using COST-231 formula.")

        # ── Build params dict ──
        params = {
            "k1": k1, "k2": k2,
            "polygons": polygons, "meta": poly_meta,
            "antenna_gain": args.antenna_gain,
            "cable_loss":   args.cable_loss,
            "ue_height":    args.ue_height,
            "frequency_mhz":  cell_freq,          # FIX #3: use real freq
            "bandwidth_mhz":  args.bandwidth,      # FIX #5: from CLI arg
            # FIX #1 ── pass ALL site rows so neighbours contribute to RSSI
            "all_sites_rows": interference_site_rows,
            "n_workers": args.n_workers,
        }

        if not prediction_points_df.empty:
            point_cols = [
                col for col in [
                    "lat",
                    "lon",
                    "frontend_grid_id",
                    "grid_id",
                    "sample_index",
                    "grid_center_lat",
                    "grid_center_lon",
                    "assignment_source",
                ] if col in prediction_points_df.columns
            ]
            pts = prediction_points_df.loc[
                prediction_points_df["Node_Cell_ID"].astype(str) == str(cid),
                point_cols,
            ].copy()
            pts = pts.dropna(subset=["lat", "lon"]).drop_duplicates()
            point_source = "frontend_grid_samples"
        else:
            pts = generate_grid(site_rows, args.radius, args.grid_resolution)
            point_source = "circular_cell_grid"
        print(f" Grid points: {len(pts)} source={point_source}")
        if pts.empty:
            print(f"[LTE][CELL_SKIP] cell={cid} reason=no_prediction_points source={point_source}")
            continue

        rsrp, rsrq, sinr = compute_predictions_parallel(
            pts, site_rows, params, n_workers=args.n_workers, use_shared_pool=True
        )

        pts["pred_rsrp"]     = np.clip(rsrp, -140, -44)
        pts["pred_rsrq"]     = np.clip(rsrq, -20,   -3)
        pts["pred_sinr"]     = np.clip(sinr, -10,   30)
        pts["Node_Cell_ID"]  = cid

        final_list.append(pts)
        print(
            f"[LTE][CELL_TIMING] cell={cid} "
            f"elapsed_sec={time.perf_counter() - cell_start:.2f} "
            f"grid_points={len(pts)} completed={idx}/{total_prediction_cells}"
        )

    # ── 7. COMBINE & CLIP TO AREA ───────────────────────────────
    print("\nCombining all sector predictions…")
    final_df = pd.concat(final_list, ignore_index=True)

    if area_polygon is not None:
        final_df = final_df[
            final_df.apply(
                lambda r: area_polygon.contains(Point(r["lon"], r["lat"])), axis=1)
        ]

    # ── 8. ACCURACY REPORT ──────────────────────────────────────
    if drive_df is not None:
        # FIX #6 ── report uses proper params (not last cell's stale params)
        report_params = {
            "k1": 0, "k2": 0,
            "polygons": None, "meta": None,
            "antenna_gain":   args.antenna_gain,
            "cable_loss":     args.cable_loss,
            "ue_height":      args.ue_height,
            "frequency_mhz":  args.frequency,
            "bandwidth_mhz":  args.bandwidth,
            "all_sites_rows": select_nearest_site_rows(
                site_df, site_df, max_interference_sites
            ),
            "n_workers":      args.n_workers,
        }
        generate_accuracy_report(drive_df, site_df, report_params)

    # ── 9. SAVE OUTPUT ──────────────────────────────────────────
    outfile = os.path.join(args.outdir, "prediction_ALL_SITES.csv")
    final_df.to_csv(outfile, index=False)

    print("\n============================================================")
    print("  FINAL OUTPUT SAVED TO:")
    print(" ", outfile)
    print("============================================================")

def run_prediction_from_api(params):

    class Args:
        pass

    args = Args()

    for k, v in params.items():
        setattr(args, k, v)

    main(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site",            required=True)
    parser.add_argument("--drive",           default=None)
    parser.add_argument("--building",        default=None)
    parser.add_argument("--polygon_area",    default=None)
    parser.add_argument("--radius",          type=float, default=5000)
    parser.add_argument("--grid_resolution", type=float, default=25)

    # FIX #3 & #5 ── frequency & bandwidth now properly used throughout
    parser.add_argument("--frequency",       type=float, default=1800,
                        help="Carrier frequency in MHz (e.g. 900, 1800, 2100)")
    parser.add_argument("--bandwidth",       type=float, default=10,
                        help="Channel bandwidth in MHz (1.4/3/5/10/15/20)")

    parser.add_argument("--tx_power",       type=float, default=46)
    parser.add_argument("--antenna_gain",   type=float, default=18)
    parser.add_argument("--cable_loss",     type=float, default=2)
    parser.add_argument("--bs_height",      type=float, default=30)
    parser.add_argument("--ue_height",      type=float, default=1.5)
    parser.add_argument("--calibrate",      action="store_true")
    parser.add_argument("--n_workers", type=int, default=mp.cpu_count() - 1)
    parser.add_argument("--max_interference_sites", type=int, default=50)
    parser.add_argument("--outdir",         default="./output")

    args = parser.parse_args()
    main(args)
