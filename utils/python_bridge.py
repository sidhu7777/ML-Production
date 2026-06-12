from __future__ import annotations

import math
import os
import re
from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd
import requests


class PythonBridgeError(RuntimeError):
    pass


def bridge_enabled() -> bool:
    return bool(os.getenv("PYTHON_BRIDGE_BASE_URL") or os.getenv("SIGNAL_TRACKERS_BRIDGE_URL"))


def _base_url() -> str:
    raw = os.getenv("PYTHON_BRIDGE_BASE_URL") or os.getenv("SIGNAL_TRACKERS_BRIDGE_URL") or ""
    return raw.rstrip("/")


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    api_key = os.getenv("PYTHON_BRIDGE_API_KEY")
    if api_key:
        headers["X-Python-Bridge-Key"] = api_key
    return headers


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if value is pd.NA:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    safe_df = df.copy()
    safe_df = safe_df.replace({pd.NA: None})
    safe_df = safe_df.where(pd.notna(safe_df), None)
    return [
        {key: _json_safe(value) for key, value in row.items()}
        for row in safe_df.to_dict(orient="records")
    ]


class PythonBridgeClient:
    def __init__(self, timeout: int | None = None):
        base = _base_url()
        if not base:
            raise PythonBridgeError("PYTHON_BRIDGE_BASE_URL is not configured")
        self.api_root_url = base
        if not base.lower().endswith("/api/pythonbridge"):
            base = f"{base}/api/PythonBridge"
        self.base_url = base
        self.timeout = timeout or int(os.getenv("PYTHON_BRIDGE_TIMEOUT_SECONDS", "120"))

    def _request_url(self, method: str, url: str, **kwargs) -> dict[str, Any]:
        headers = _headers()
        extra_headers = kwargs.pop("headers", None)
        if extra_headers:
            headers.update({k: v for k, v in extra_headers.items() if v})
        if method.upper() == "POST":
            headers["Content-Type"] = "application/json"
        timeout = kwargs.pop("timeout", self.timeout)
        try:
            response = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            raise PythonBridgeError(f"{method} {url} failed: {exc}") from exc
        try:
            payload = response.json()
        except ValueError:
            payload = {"Status": 0, "Message": response.text}
        if not response.ok or payload.get("Status") == 0:
            message = payload.get("Message") or response.text or response.reason
            raise PythonBridgeError(f"{method} {url} failed: {response.status_code} {message}")
        return payload

    def _request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        return self._request_url(method, url, **kwargs)

    def get_rows(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        limit: int = 50000,
        cursor_param: str | None = None,
        cursor_column: str = "id",
        progress_label: str | None = None,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        offset = 0
        page_limit = int(limit)
        min_page_limit = int(os.getenv("PYTHON_BRIDGE_MIN_PAGE_SIZE", "250"))
        while True:
            page_params = dict(params or {})
            page_params["limit"] = page_limit
            if cursor_param:
                page_params[cursor_param] = offset
            else:
                page_params["offset"] = offset
            try:
                payload = self._request("GET", endpoint, params=page_params)
            except PythonBridgeError:
                if page_limit > min_page_limit:
                    page_limit = max(min_page_limit, page_limit // 2)
                    continue
                raise
            page_rows = payload.get("Data")
            if page_rows is None:
                page_rows = payload.get("Rows")
            page_rows = page_rows or []
            rows.extend(page_rows)
            count = int(payload.get("Count") or len(page_rows))
            page_limit = int(payload.get("Limit") or page_limit)
            if progress_label:
                print(
                    f"[PYTHON_BRIDGE_PAGE] label={progress_label} endpoint={endpoint} "
                    f"offset={offset} limit={page_limit} rows={count} total={len(rows)}",
                    flush=True,
                )
            if count < page_limit or not page_rows:
                break
            if cursor_param:
                next_offset = page_rows[-1].get(cursor_column)
                try:
                    next_offset = int(next_offset)
                except (TypeError, ValueError):
                    break
                if next_offset <= offset:
                    break
                offset = next_offset
            else:
                offset += page_limit
        return pd.DataFrame(rows)

    def post_rows(self, endpoint: str, body: dict[str, Any], limit: int = 50000) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            page_body = dict(body)
            page_body.setdefault("Limit", limit)
            page_body["Offset"] = offset
            payload = self._request("POST", endpoint, json=page_body)
            page_rows = payload.get("Data")
            if page_rows is None:
                page_rows = payload.get("Rows")
            page_rows = page_rows or []
            rows.extend(page_rows)
            count = int(payload.get("Count") or len(page_rows))
            page_limit = int(payload.get("Limit") or limit)
            if count < page_limit or not page_rows:
                break
            offset += page_limit
        return pd.DataFrame(rows)

    def save_dataframe(
        self,
        endpoint: str,
        df: pd.DataFrame,
        project_id: int,
        job_id: str,
        region: str | None = None,
        chunk_size: int = 3000,
        replace_existing: bool = False,
    ) -> int:
        total = 0
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start + chunk_size]
            payload = {
                "ProjectId": int(project_id),
                "JobId": str(job_id),
                "Region": region,
                "ReplaceExisting": bool(replace_existing and start == 0),
                "Rows": _records(chunk),
            }
            result = self._request("POST", endpoint, json=payload)
            total += int(result.get("Inserted") or result.get("Deleted") or 0)
        return total

    def get_grid_analytics(
        self,
        project_id: int,
        scenario_id: int | None = None,
        auth_header: str | None = None,
        cookie_header: str | None = None,
    ) -> tuple[pd.DataFrame, Any]:
        params: dict[str, Any] = {
            "projectId": int(project_id),
            "version": "original",
        }
        if scenario_id is not None:
            params["scenario_id"] = int(scenario_id)

        headers = {}
        if auth_header:
            headers["Authorization"] = auth_header
        if cookie_header:
            headers["Cookie"] = cookie_header

        payload = self._request_url(
            "GET",
            f"{self.api_root_url}/api/GridAnalytics/GetGridAnalytics",
            params=params,
            headers=headers,
            timeout=int(os.getenv("PYTHON_BRIDGE_GRID_TIMEOUT_SECONDS", "15")),
        )
        data = payload.get("Data") or payload.get("data") or {}
        rows = data.get("grids") if isinstance(data, dict) else None
        grid_size = (
            data.get("grid_size_meters")
            or data.get("gridSizeMeters")
            or data.get("GridSizeMeters")
            if isinstance(data, dict)
            else None
        )
        if not rows:
            return pd.DataFrame(), grid_size
        return _normalize_grid_analytics_df(pd.DataFrame(rows), grid_size), grid_size


def _snake_case(name):
    text = str(name or "").strip()
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", text)
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    return text.strip("_").lower()


def _first_present(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _copy_first_present(df, target, candidates):
    if target in df.columns:
        return
    source = _first_present(df, candidates)
    if source:
        df[target] = df[source]


def _normalize_grid_analytics_df(df, grid_size_meters=None):
    if df.empty:
        return df

    df = df.copy()
    df.columns = [_snake_case(col) for col in df.columns]

    for prefix in ["baseline", "optimized", "optimised", "difference"]:
        if prefix not in df.columns:
            continue
        expanded = df[prefix].apply(lambda value: value if isinstance(value, dict) else {}).apply(pd.Series)
        if expanded.empty:
            continue
        expanded.columns = [f"{prefix}_{_snake_case(col)}" for col in expanded.columns]
        df = pd.concat([df.drop(columns=[prefix]), expanded], axis=1)

    _copy_first_present(df, "baseline_point_count", ["baseline_point_count", "baseline_count", "point_count"])
    _copy_first_present(df, "baseline_avg_rsrp", ["baseline_avg_rsrp", "avg_rsrp"])
    _copy_first_present(df, "baseline_avg_rsrq", ["baseline_avg_rsrq", "avg_rsrq"])
    _copy_first_present(df, "baseline_avg_sinr", ["baseline_avg_sinr", "avg_sinr"])

    _copy_first_present(df, "grid_id", ["id", "gridid", "grid_cell_id", "cell_grid_id"])
    _copy_first_present(df, "center_lat", ["centerlat", "centre_lat", "latitude", "lat", "grid_lat"])
    _copy_first_present(
        df,
        "center_lon",
        ["centerlon", "center_lng", "centrelon", "centre_lon", "longitude", "lng", "lon", "grid_lon"],
    )
    _copy_first_present(df, "lat", ["center_lat", "latitude", "grid_lat"])
    _copy_first_present(df, "lon", ["center_lon", "longitude", "lng", "grid_lon"])

    _copy_first_present(df, "min_lat", ["minlat", "south", "south_lat", "bottom_lat"])
    _copy_first_present(df, "max_lat", ["maxlat", "north", "north_lat", "top_lat"])
    _copy_first_present(df, "min_lon", ["minlon", "min_lng", "west", "west_lon", "left_lon"])
    _copy_first_present(df, "max_lon", ["maxlon", "max_lng", "east", "east_lon", "right_lon"])

    for col in ["center_lat", "center_lon", "lat", "lon", "min_lat", "max_lat", "min_lon", "max_lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    grid_size = pd.to_numeric(grid_size_meters, errors="coerce")
    if pd.notna(grid_size) and {"center_lat", "center_lon"}.issubset(df.columns):
        half_meters = float(grid_size) / 2.0
        lat_delta = half_meters / 111320.0
        lon_delta = half_meters / (111320.0 * df["center_lat"].apply(lambda lat: max(math.cos(math.radians(lat)), 0.01)))
        if "min_lat" not in df.columns:
            df["min_lat"] = df["center_lat"] - lat_delta
        if "max_lat" not in df.columns:
            df["max_lat"] = df["center_lat"] + lat_delta
        if "min_lon" not in df.columns:
            df["min_lon"] = df["center_lon"] - lon_delta
        if "max_lon" not in df.columns:
            df["max_lon"] = df["center_lon"] + lon_delta

    if "grid_id" not in df.columns:
        df["grid_id"] = range(1, len(df) + 1)

    return df


def get_bridge_client() -> PythonBridgeClient | None:
    if not bridge_enabled():
        return None
    return PythonBridgeClient()
