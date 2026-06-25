"""
Model 1 interaction-coverage inspection.

This test answers whether important feature combinations are present often
enough in training data, or whether many appear only in PART_3.

It does not modify existing Model 1 artifacts. It writes a new inspection
folder under models/model1_interaction_coverage.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ML_ROOT = Path(__file__).resolve().parents[2]
os.chdir(ML_ROOT)
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))


DATASET_CSV = Path("data") / "model1_coverage_training.csv"
MODEL_ROOT = Path("models") / "model1_retrain" / "remove_coordinates"
OUTPUT_ROOT = Path("models") / "model1_interaction_coverage"

# Start from the coordinate-free feature set and focus on the features that
# repeatedly appear high in SHAP for all three targets.
KEY_FEATURES = [
    "bucket_seq",
    "dominant_band_class",
    "los_blocked_ratio",
    "morphology_cluster",
    "serving_distance_m",
    "site_count_500m",
    "site_count_250m",
    "azimuth_delta_deg",
    "prev_obs_rsrp",
    "prev_obs_rsrq",
    "prev_obs_sinr",
]


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def load_features() -> list[str]:
    metadata_path = MODEL_ROOT / "pred_rsrp" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = list(metadata["features"]["numeric"]) + list(metadata["features"]["categorical"])
    missing = [feature for feature in KEY_FEATURES if feature not in features]
    if missing:
        raise RuntimeError(f"Expected features missing from coordinate-free model: {missing}")
    return features


def quantile_bins(reference: pd.Series, bins: int = 4) -> np.ndarray:
    values = pd.to_numeric(reference, errors="coerce").dropna()
    if values.empty:
        return np.array([-np.inf, np.inf])
    edges = np.unique(np.nanquantile(values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        lo = float(values.min())
        hi = float(values.max())
        if hi <= lo:
            return np.array([-np.inf, np.inf])
        edges = np.linspace(lo, hi, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def bin_feature(series: pd.Series, feature: str, train_reference: pd.Series) -> pd.Series:
    if feature == "dominant_band_class":
        return series.fillna("Unknown").astype(str)
    if feature in {"bucket_seq", "morphology_cluster", "site_count_250m", "site_count_500m"}:
        return pd.to_numeric(series, errors="coerce").fillna(-1).round(0).astype("Int64").astype(str)
    if feature in {"prev_obs_rsrp", "prev_obs_rsrq", "prev_obs_sinr"}:
        return pd.cut(pd.to_numeric(series, errors="coerce"), quantile_bins(train_reference, bins=4), include_lowest=True).astype(str)
    if feature in {"serving_distance_m", "los_blocked_ratio", "azimuth_delta_deg"}:
        return pd.cut(pd.to_numeric(series, errors="coerce"), quantile_bins(train_reference, bins=4), include_lowest=True).astype(str)
    return series.fillna("Unknown").astype(str)


def signature_frame(df: pd.DataFrame, train_reference: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    work = pd.DataFrame(index=df.index)
    for feature in features:
        work[feature] = bin_feature(df[feature], feature, train_reference[feature])
    return work


def combo_stats(train_sig: pd.DataFrame, part3_sig: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    rows = []
    train_cols = [f"f{i}" for i in range(len(features))]
    train_keyed = train_sig.copy()
    part3_keyed = part3_sig.copy()
    train_keyed.columns = train_cols
    part3_keyed.columns = train_cols

    train_keys = train_keyed.astype(str).agg("||".join, axis=1)
    part3_keys = part3_keyed.astype(str).agg("||".join, axis=1)

    train_counts = Counter(train_keys)
    part3_counts = Counter(part3_keys)

    unseen_in_train = sorted(set(part3_counts) - set(train_counts))
    rare_in_train = [key for key, count in train_counts.items() if count <= 2]

    def decode(key: str) -> dict[str, str]:
        parts = key.split("||")
        return {features[i]: parts[i] for i in range(len(features))}

    unseen_rows = []
    for key in unseen_in_train[:30]:
        unseen_rows.append(
            {
                "signature": key,
                "part3_count": int(part3_counts[key]),
                **decode(key),
            }
        )

    train_mass = sum(train_counts[key] for key in set(train_counts) & set(part3_counts))
    part3_mass = sum(part3_counts[key] for key in set(train_counts) & set(part3_counts))

    return {
        "feature_set": features,
        "train_unique_combinations": int(len(train_counts)),
        "part3_unique_combinations": int(len(part3_counts)),
        "shared_combinations": int(len(set(train_counts) & set(part3_counts))),
        "unseen_part3_combinations": int(len(unseen_in_train)),
        "unseen_part3_rows": int(sum(part3_counts[key] for key in unseen_in_train)),
        "unseen_part3_row_fraction": float(sum(part3_counts[key] for key in unseen_in_train) / max(sum(part3_counts.values()), 1)),
        "train_rare_combination_count_le_2": int(len(rare_in_train)),
        "train_covering_mass_for_part3_rows": float(part3_mass / max(sum(part3_counts.values()), 1)),
        "top_unseen_part3_combinations": unseen_rows,
    }


def pairwise_new_combo_rates(train_sig: pd.DataFrame, part3_sig: pd.DataFrame, features: list[str]) -> list[dict[str, Any]]:
    train_pairs = {}
    part3_pairs = {}
    for i, left in enumerate(features):
        for right in features[i + 1 :]:
            train_pairs[(left, right)] = set(
                train_sig[left].astype(str) + "||" + train_sig[right].astype(str)
            )
            part3_pairs[(left, right)] = set(
                part3_sig[left].astype(str) + "||" + part3_sig[right].astype(str)
            )

    rows = []
    for pair, part3_set in part3_pairs.items():
        train_set = train_pairs[pair]
        unseen = part3_set - train_set
        rows.append(
            {
                "feature_left": pair[0],
                "feature_right": pair[1],
                "train_unique_pairs": int(len(train_set)),
                "part3_unique_pairs": int(len(part3_set)),
                "part3_unseen_pairs": int(len(unseen)),
                "part3_unseen_pair_fraction": float(len(unseen) / max(len(part3_set), 1)),
            }
        )
    return rows


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    features = load_features()
    df = pd.read_csv(DATASET_CSV)

    train_df = df[df["time_bucket"].isin(["PART_1", "PART_2"])].copy()
    part3_df = df[df["time_bucket"] == "PART_3"].copy()

    for feature in KEY_FEATURES:
        if feature in train_df.columns:
            train_df[feature] = pd.to_numeric(train_df[feature], errors="ignore")
        if feature in part3_df.columns:
            part3_df[feature] = pd.to_numeric(part3_df[feature], errors="ignore")

    sig_features = KEY_FEATURES
    train_sig = signature_frame(train_df, train_df, sig_features)
    part3_sig = signature_frame(part3_df, train_df, sig_features)

    combo = combo_stats(train_sig, part3_sig, sig_features)
    pair_rows = pairwise_new_combo_rates(train_sig, part3_sig, sig_features)
    pair_df = pd.DataFrame(pair_rows).sort_values("part3_unseen_pair_fraction", ascending=False)

    combo["top_pairwise_new_combo_rates"] = pair_df.head(20).to_dict(orient="records")
    combo["notes"] = [
        "A combination is considered new if it appears in PART_3 but never in PART_1+PART_2.",
        "Continuous features are binned using training quantiles before combination counting.",
    ]

    pair_df.to_csv(OUTPUT_ROOT / "pairwise_new_combo_rates.csv", index=False)
    save_json(combo, OUTPUT_ROOT / "interaction_coverage_summary.json")
    print(json.dumps(combo, indent=2))


if __name__ == "__main__":
    main()
