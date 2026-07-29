from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


ML_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = ML_ROOT / "models" / "model1_hybrid_target_experiment" / "physical_no_teacher_summary"
TARGETS = ["pred_rsrp", "pred_rsrq", "pred_sinr"]
FORBIDDEN_PRODUCTION_FEATURES = {"bucket_seq", "time_bucket"}
KNOWN_CATEGORICAL_FEATURES = {"clutter_class", "dominant_band_class"}


@dataclass(frozen=True)
class Model1Bundle:
    models: dict[str, Any]
    metadata: dict[str, dict[str, Any]]
    weights_paths: dict[str, str]
    numeric_features: list[str]
    categorical_features: list[str]
    model_version: str


def load_latest_model1_bundle(model_root: Path = MODEL_ROOT) -> Model1Bundle:
    models: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}
    weights_paths: dict[str, str] = {}

    for target in TARGETS:
        target_dir = model_root / target
        root_model_path = model_root / f"{target}.joblib"
        model_path = root_model_path if root_model_path.exists() else target_dir / f"{target}.joblib"
        metadata_path = target_dir / "metadata.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing Model 1 weights: {model_path}")
        models[target] = joblib.load(model_path)
        metadata[target] = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        weights_paths[target] = str(model_path)

    first_meta = metadata[TARGETS[0]]
    features = first_meta.get("features", {})
    numeric = list(features.get("numeric") or [])
    categorical = list(features.get("categorical") or [])
    if not numeric and hasattr(models[TARGETS[0]], "feature_names_in_"):
        feature_names = list(models[TARGETS[0]].feature_names_in_)
        categorical = [name for name in feature_names if name in KNOWN_CATEGORICAL_FEATURES]
        numeric = [name for name in feature_names if name not in categorical]
    else:
        misplaced_categorical = [name for name in numeric if name in KNOWN_CATEGORICAL_FEATURES]
        if misplaced_categorical:
            numeric = [name for name in numeric if name not in KNOWN_CATEGORICAL_FEATURES]
            categorical = list(dict.fromkeys([*categorical, *misplaced_categorical]))
    forbidden = sorted(FORBIDDEN_PRODUCTION_FEATURES.intersection(set(numeric + categorical)))
    if forbidden:
        raise RuntimeError(
            "Loaded Model 1 weights are not production-clean. "
            f"Forbidden training-only features present: {forbidden}. "
            f"Use no-bucket/no-time-sequence Model 1 weights under {model_root}."
        )
    trained_at = str(first_meta.get("trained_at") or "")
    model_type = str(first_meta.get("model_type") or "model1")
    return Model1Bundle(
        models=models,
        metadata=metadata,
        weights_paths=weights_paths,
        numeric_features=numeric,
        categorical_features=categorical,
        model_version=f"{model_type}_{trained_at}".strip("_"),
    )
