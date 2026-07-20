from __future__ import annotations

import math
import os
import re
import json
from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd
import requests


class PythonBridgeError(RuntimeError):
    pass


def bridge_enabled() -> bool:
    return bool(os.getenv("PYTHON_BRIDGE_BASE_URL") or os.getenv("SIGNAL_TRACKERS_BRIDGE_URL"))


def _normalize_url(value: Any) -> str:
    return str(value or "").strip().lstrip("=").strip().rstrip("/")


def _base_url() -> str:
    raw = os.getenv("PYTHON_BRIDGE_BASE_URL") or os.getenv("SIGNAL_TRACKERS_BRIDGE_URL") or ""
    return _normalize_url(raw)


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    api_key = os.getenv("PYTHON_BRIDGE_API_KEY")
    if api_key:
        headers["X-Python-Bridge-Key"] = api_key
    return headers


def _mapview_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    authorization = os.getenv("MAPVIEW_API_AUTHORIZATION")
    cookie = os.getenv("MAPVIEW_API_COOKIE")
    if authorization:
        headers["Authorization"] = authorization
    if cookie:
        headers["Cookie"] = cookie
    return headers


def _service_root_url() -> str:
    base = _normalize_url(os.getenv("BASE_URL") or _base_url() or "")
    if base.lower().endswith("/api/pythonbridge"):
        return base[: -len("/api/PythonBridge")]
    return base


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
    try:
        return json.loads(safe_df.to_json(orient="records", date_format="iso", date_unit="s"))
    except Exception:
        return [
            {key: _json_safe(value) for key, value in row.items()}
            for row in safe_df.to_dict(orient="records")
        ]


_SITE_PREDICTION_ENDPOINTS = {
    "getltesitepredictionrows",
    "getltetiltantennarows",
    "getsitepredictionoptimized",
    "getsitepredictionoptimised",
    "getsiteprediction",
}


def _clean_identity_text(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    lowered = text.str.lower()
    invalid = text.isna() | text.eq("") | lowered.isin({"nan", "none", "null", "undefined", "unknown", "n/a", "na"})
    return text.mask(invalid, "")


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_to_original = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        column = lower_to_original.get(str(candidate).lower())
        if column is not None:
            return column
    return None


def _filter_complete_site_prediction_identity(df: pd.DataFrame, endpoint: str | None = None) -> pd.DataFrame:
    if df.empty:
        return df

    site_col = _first_existing_column(df, ["site", "site_id", "siteId"])
    cell_col = _first_existing_column(df, ["cell_id", "cellId"])
    sector_col = _first_existing_column(df, ["sector", "sector_id", "sectorId"])
    band_col = _first_existing_column(df, ["band", "frequency_band", "Band"])
    operator_col = _first_existing_column(df, ["operator", "operator_name", "operatorName", "provider", "cluster", "network", "Network"])
    required = [site_col, cell_col, sector_col, band_col, operator_col]
    if any(col is None for col in required):
        missing = [
            name
            for name, col in [
                ("site", site_col),
                ("cell_id", cell_col),
                ("sector", sector_col),
                ("band", band_col),
                ("operator", operator_col),
            ]
            if col is None
        ]
        print(
            f"[PYTHON_BRIDGE_SITE_IDENTITY_FILTER] endpoint={endpoint or 'unknown'} "
            f"rows_in={len(df)} rows_out=0 missing_columns={','.join(missing)}",
            flush=True,
        )
        return df.iloc[0:0].copy()

    work = df.copy()
    site = _clean_identity_text(work[site_col])
    cell = _clean_identity_text(work[cell_col])
    sector = _clean_identity_text(work[sector_col])
    band = _clean_identity_text(work[band_col])
    operator = _clean_identity_text(work[operator_col])
    keep = site.ne("") & cell.ne("") & sector.ne("") & band.ne("") & operator.ne("")
    filtered = work.loc[keep].copy()
    if not filtered.empty:
        filtered["site_prediction_key"] = (
            site.loc[keep]
            + "|"
            + cell.loc[keep]
            + "|"
            + sector.loc[keep]
            + "|"
            + band.loc[keep]
            + "|"
            + operator.loc[keep]
        )
        filtered["site_cell_sector_band_operator_key"] = filtered["site_prediction_key"]
        if "operator" not in filtered.columns:
            filtered["operator"] = operator.loc[keep]
        if "operator_name" not in filtered.columns:
            filtered["operator_name"] = operator.loc[keep]
        if "provider" not in filtered.columns:
            filtered["provider"] = operator.loc[keep]

    removed = int(len(work) - len(filtered))
    if removed:
        print(
            f"[PYTHON_BRIDGE_SITE_IDENTITY_FILTER] endpoint={endpoint or 'unknown'} "
            f"rows_in={len(work)} rows_out={len(filtered)} removed={removed} "
            "required=site,cell_id,sector,band,operator",
            flush=True,
        )
    return filtered


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
        df = pd.DataFrame(rows)
        endpoint_key = str(endpoint or "").strip("/").split("/")[-1].lower()
        if endpoint_key in _SITE_PREDICTION_ENDPOINTS:
            df = _filter_complete_site_prediction_identity(df, endpoint=endpoint)
        return df

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
        df = pd.DataFrame(rows)
        endpoint_key = str(endpoint or "").strip("/").split("/")[-1].lower()
        if endpoint_key in _SITE_PREDICTION_ENDPOINTS:
            df = _filter_complete_site_prediction_identity(df, endpoint=endpoint)
        return df

    def get_project(self, project_id: int) -> dict[str, Any] | None:
        payload = self._request("GET", "GetProject", params={"projectId": int(project_id)})
        return payload.get("Data")

    def get_project_regions(self, project_id: int) -> list[dict[str, Any]]:
        payload = self._request("GET", "GetProjectRegions", params={"projectId": int(project_id)})
        return payload.get("Data") or []

    def get_report_network_logs(
        self,
        session_ids: Iterable[int],
        limit: int = 50000,
        project_id: int | None = None,
        provider: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        session_ids = [int(v) for v in session_ids if int(v) > 0]
        print(
            "[ReportLogs] requesting logs via PythonBridge "
            f"sessions={len(session_ids)} project_id={project_id} "
            f"provider={provider!r} start_date={start_date} end_date={end_date} limit={limit}"
        )
        if os.getenv("REPORT_USE_MAPVIEW_NETWORKLOG_API", "0") == "1":
            try:
                df = self.get_mapview_network_logs(session_ids, limit=limit, project_id=project_id)
                print(f"[ReportLogs] received rows from MapView/GetNetworkLog: {len(df)}")
                return df
            except PythonBridgeError as exc:
                print(f"[ReportLogs] MapView/GetNetworkLog failed, falling back to PythonBridge: {exc}")
        df = self.post_rows(
            "GetReportNetworkLogs",
            {
                "SessionIds": session_ids,
                "ProjectId": int(project_id) if project_id is not None else None,
                "Provider": provider,
                "StartDate": start_date,
                "EndDate": end_date,
                "Limit": int(limit),
            },
            limit=limit,
        )
        print(f"[ReportLogs] received rows from PythonBridge/GetReportNetworkLogs: {len(df)}")
        return df

    def get_mapview_network_logs(
        self,
        session_ids: Iterable[int],
        limit: int = 50000,
        project_id: int | None = None,
    ) -> pd.DataFrame:
        session_ids = [int(v) for v in session_ids if int(v) > 0]
        if not session_ids:
            return pd.DataFrame()

        service_root = _service_root_url()
        if not service_root:
            raise PythonBridgeError("BASE_URL is not configured for MapView network-log access")

        url = f"{service_root}/api/MapView/GetNetworkLog"
        params: dict[str, Any] = {
            "session_ids": ",".join(str(v) for v in session_ids),
            "limit": int(limit),
            "page": 1,
        }
        if project_id is not None:
            params["project_id"] = int(project_id)

        payload = self._request_url(
            "GET",
            url,
            params=params,
            headers=_mapview_headers(),
        )
        rows = payload.get("data") or payload.get("Data") or []
        return pd.DataFrame(rows)

    def get_sessions(self, session_ids: Iterable[int]) -> pd.DataFrame:
        session_ids = [int(v) for v in session_ids if int(v) > 0]
        payload = self._request("POST", "GetSessions", json={"SessionIds": session_ids})
        rows = payload.get("Data") or []
        return pd.DataFrame(rows)

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        payload = self._request("GET", "GetUser", params={"userId": int(user_id)})
        return payload.get("Data")

    def get_user_thresholds(self, user_id: int) -> dict[str, Any] | None:
        payload = self._request("GET", "GetUserThresholds", params={"userId": int(user_id)})
        return payload.get("Data")

    def update_project_download_path(self, project_id: int, download_path: str) -> bool:
        payload = self._request(
            "POST",
            "UpdateProjectDownloadPath",
            json={"ProjectId": int(project_id), "DownloadPath": str(download_path)},
        )
        return bool(payload.get("Updated", False) or payload.get("Status") == 1)

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
        progress_enabled = os.getenv("PYTHON_BRIDGE_SAVE_PROGRESS", "1") != "0"
        for start in range(0, len(df), chunk_size):
            chunk_started = datetime.now()
            chunk = df.iloc[start:start + chunk_size]
            payload = {
                "ProjectId": int(project_id),
                "JobId": str(job_id),
                "Region": region,
                "ReplaceExisting": bool(replace_existing and start == 0),
                "Rows": _records(chunk),
            }
            result = self._request("POST", endpoint, json=payload)
            written = int(result.get("Inserted") or result.get("Deleted") or 0)
            total += written
            if progress_enabled:
                elapsed = (datetime.now() - chunk_started).total_seconds()
                print(
                    f"[PYTHON_BRIDGE_SAVE] endpoint={endpoint} offset={start} "
                    f"rows={len(chunk)} written={written} total={total}/{len(df)} "
                    f"elapsed_sec={elapsed:.2f}",
                    flush=True,
                )
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
