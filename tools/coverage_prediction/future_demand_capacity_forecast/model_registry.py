from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


ML_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = Path(__file__).resolve().parent / "weights"
MODEL_CANDIDATE_ROOTS = [
    MODEL_ROOT,
    ML_ROOT / "models" / "model2_hybrid_target_experiment",
    ML_ROOT / "models" / "model2",
]

TARGETS = {
    "model2a_demand": {
        "output": "demand_index",
        "model_file": "model2a_demand_xgb.pkl",
    },
    "model2b_users": {
        "output": "active_users_est",
        "model_file": "model2b_users_xgb.pkl",
    },
    "model2c_traffic": {
        "output": "traffic_demand_est",
        "model_file": "model2c_traffic_xgb.pkl",
    },
}

FORBIDDEN_PRODUCTION_FEATURES = {"bucket_seq", "time_bucket"}


@dataclass(frozen=True)
class Model2Bundle:
    models: dict[str, Any]
    metadata: dict[str, dict[str, Any]]
    weights_paths: dict[str, str]
    numeric_features: list[str]
    categorical_features: list[str]
    model_version: str


def _load_from_root(model_root: Path) -> Model2Bundle:
    models: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}
    weights_paths: dict[str, str] = {}

    for key, spec in TARGETS.items():
        target_dir = model_root / key
        model_path = target_dir / spec["model_file"]
        metadata_path = target_dir / "metadata.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing Model 2 weights: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing Model 2 metadata: {metadata_path}")
        models[key] = joblib.load(model_path)
        metadata[key] = json.loads(metadata_path.read_text(encoding="utf-8"))
        weights_paths[key] = str(model_path)

    first_meta = metadata["model2a_demand"]
    features = first_meta.get("features", {})
    numeric = list(features.get("numeric") or [])
    categorical = list(features.get("categorical") or [])
    if not numeric and hasattr(models["model2a_demand"], "feature_names_in_"):
        feature_names = list(models["model2a_demand"].feature_names_in_)
        categorical = [name for name in feature_names if name in {"clutter_class", "dominant_band_class"}]
        numeric = [name for name in feature_names if name not in categorical]
    forbidden = sorted(FORBIDDEN_PRODUCTION_FEATURES.intersection(set(numeric + categorical)))
    if forbidden:
        raise RuntimeError(
            "Loaded Model 2 weights are not production-clean. "
            f"Forbidden training-only features present: {forbidden}. "
            f"Replace weights under {model_root} with no-bucket/no-time-sequence Model 2 weights."
        )
    trained_at = str(first_meta.get("trained_at") or "")
    model_type = str(first_meta.get("model_type") or "future_demand_capacity_forecast")
    return Model2Bundle(
        models=models,
        metadata=metadata,
        weights_paths=weights_paths,
        numeric_features=numeric,
        categorical_features=categorical,
        model_version=f"{model_type}_{trained_at}".strip("_"),
    )


def load_latest_model2_bundle(model_root: Path | None = None) -> Model2Bundle:
    roots = [model_root] if model_root is not None else MODEL_CANDIDATE_ROOTS
    failures: list[str] = []
    for root in roots:
        try:
            return _load_from_root(root)
        except RuntimeError as exc:
            failures.append(f"{root}: {exc}")
    raise RuntimeError(
        "No production-clean Model 2 weights found. "
        "Every available candidate still contains training-only bucket/time features. "
        + " | ".join(failures)
    )
