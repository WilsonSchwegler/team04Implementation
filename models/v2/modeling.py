from __future__ import annotations
from dataclasses import dataclass
import pickle
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

if __package__ in {None, ""}:
    package = Path(__file__).resolve().parents[2]
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    from models.v2.config import (
        event_tree_evaluator,
        event_tree_model_path,
        planner_eval_metadata,
        planner_metadata_path,
        random_seed,
    )
    from models.v2.cache_helpers import assign_crossfit_fold
    from models.v2.location_grid import (
        bucket_ids,
        bucket_center,
        default_bucket_prior_rows,
        locate_bucket_id,
    )
    from models.v2.data import (
        batter_specific_target_scope,
        broad_target_scope,
        default_target_scope,
        handidness_target_scope,
        V2Tables,
        build_pitch2_event_premium_lookup,
        build_empirical_count_state_values,
        build_true_continuation_count_state_values,
        derive_pitch_zone_bucket,
        load_available_pitch_level_history,
    )
else:
    from .config import (
        event_tree_evaluator,
        event_tree_model_path,
        planner_eval_metadata,
        planner_metadata_path,
        random_seed,
    )
    from .cache_helpers import assign_crossfit_fold
    from .location_grid import (
        bucket_ids,
        bucket_center,
        default_bucket_prior_rows,
        locate_bucket_id,
    )
    from .data import (
        batter_specific_target_scope,
        broad_target_scope,
        default_target_scope,
        handidness_target_scope,
        V2Tables,
        build_pitch2_event_premium_lookup,
        build_empirical_count_state_values,
        build_true_continuation_count_state_values,
        derive_pitch_zone_bucket,
        load_available_pitch_level_history,
    )


#shared state features
numeric_features = [
    "balls_before_p2",
    "strikes_before_p2",
    "pitch_1_velo",
    "pitch_1_h_mov",
    "pitch_1_v_mov",
    "pitch_1_spin_rate",
    "pitch_1_extension",
    "pitch_1_spin_axis",
    "pitch_1_release_x",
    "pitch_1_release_y",
    "pitch_1_release_z",
    "pitch_1_perceived_velo",
    "pitch_1_zone_distance",
    "pitch_1_delta_velo_vs_primary_fastball",
    "pitch_1_delta_h_mov_vs_primary_fastball",
    "pitch_1_delta_v_mov_vs_primary_fastball",
    "same_pitch_type",
    "same_pitch_family",
    "pitch_1_is_fastball",
    "pitch_2_is_fastball",
    "pitch_1_is_breaking",
    "pitch_2_is_breaking",
    "pitch_1_is_offspeed",
    "pitch_2_is_offspeed",
]
categorical_features = [
    "pitcher_id",
    "batter_id",
    "pitch_1_type",
    "pitcher_handedness",
    "batter_handedness",
    "count_bucket",
    "pitch_1_bucket",
]
action_categgorical_features = [
    "pitch_2_type",
    "pitch_2_bucket",
]
#shared state features
action_numeric_features = [
    "avg_velo",
    "avg_h_mov",
    "avg_v_mov",
    "avg_spin_rate",
    "avg_extension",
    "avg_release_x",
    "avg_release_z",
    "delta_velo_vs_primary_fastball",
    "delta_h_mov_vs_primary_fastball",
    "delta_v_mov_vs_primary_fastball",
]
zone_prior_features = [
    "batter_pitchtype_zone_swing_prior",
    "batter_pitchtype_zone_contact_prior",
    "batter_pitchtype_zone_whiff_prior",
    "batter_pitchtype_zone_in_play_prior",
]
numeric_swing_features  = numeric_features + action_numeric_features
numeric_contact_features = numeric_features + action_numeric_features
numeric_in_play_features = numeric_features + action_numeric_features
numeric_bucket = numeric_features + action_numeric_features
contact_quality = numeric_features + action_numeric_features
head_categorical_features = categorical_features + action_categgorical_features
called_strike_grid_size = 121
called_strike_x_range = (-1.5, 1.5)
cs_z_norm_range = (-0.5, 1.5)
cs_gaussian_sigma = 2.0
cs_min_takes = 25
cs_blend_strength = 300.0
cs_batter_weight_cap = 0.12
cs_hand_weight = 0.25
cs_global_weight = 0.75
default_batter_height_ft = 6.0
pitch_family_labels = ("fastball", "breaking", "offspeed", "other")
batted_ball_types = ("groundball", "line_drive", "fly_ball")
ev_band_labels = ("lt_90", "90_95", "95_100", "100_105", "ge_105")
batted_ball_type_to_idx = {label: idx for idx, label in enumerate(batted_ball_types)}
ev_band_to_idx = {label: idx for idx, label in enumerate(ev_band_labels)}
family_binary_min_rows = 120
family_multiclass_min_rows = 80
targets_per_pool = 3
coarse_contact_bucket_values = np.array([0.0, -0.87, -1.245, -2.05], dtype=float)
zone_prior_blend_strength = 20.0
targets_per_scope = 3

@dataclass
#feature encoding bundle
class EncodedFeatureBundle:
    numeric_features: list[str]
    categorical_features: list[str]
    category_maps: dict[str, dict[str, int]]
    numeric_fill_values: dict[str, float]


@dataclass
#smooth strike surface
class CalledStrikeSurfaceModel:
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    x_centers: np.ndarray
    z_centers: np.ndarray
    global_surface: np.ndarray
    handedness_surfaces: dict[str, np.ndarray]
    batter_surfaces: dict[str, np.ndarray]
    batter_counts: dict[str, int]
    blend_strength: float
    default_height_ft: float

    def _surface_lookup(self, surface: np.ndarray, x: np.ndarray, z_norm: np.ndarray):
        x = np.clip(np.asarray(x, dtype=float), self.x_min, self.x_max)
        z_norm = np.clip(np.asarray(z_norm, dtype=float), self.z_min, self.z_max)

        x_pos = np.interp(x, self.x_centers, np.arange(len(self.x_centers), dtype=float))
        z_pos = np.interp(z_norm, self.z_centers, np.arange(len(self.z_centers), dtype=float))
        x0 = np.floor(x_pos).astype(int)
        z0 = np.floor(z_pos).astype(int)
        x1 = np.clip(x0 + 1, 0, len(self.x_centers) - 1)
        z1 = np.clip(z0 + 1, 0, len(self.z_centers) - 1)
        x0 = np.clip(x0, 0, len(self.x_centers) - 1)
        z0 = np.clip(z0, 0, len(self.z_centers) - 1)
        wx = x_pos - x0
        wz = z_pos - z0

        v00 = surface[z0, x0]
        v10 = surface[z0, x1]
        v01 = surface[z1, x0]
        v11 = surface[z1, x1]
        return (
            (1.0 - wx) * (1.0 - wz) * v00
            + wx * (1.0 - wz) * v10
            + (1.0 - wx) * wz * v01
            + wx * wz * v11
        )

    def predict_proba(self, frame: pd.DataFrame):
        target_x = pd.to_numeric(frame["intended_target_x"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        target_z = pd.to_numeric(frame["intended_target_z"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        heights = (
            pd.to_numeric(frame.get("batter_height_ft", pd.Series(self.default_height_ft, index=frame.index)), errors="coerce")
            .fillna(self.default_height_ft)
            .clip(lower=5.0, upper=7.5)
            .to_numpy(dtype=float)
        )
        z_norm = target_z / np.maximum(heights, 1e-6)
        handedness = frame["batter_handedness"].fillna("Unknown").astype(str).to_numpy()
        batter_ids = frame["batter_id"].fillna(-1).astype(str).to_numpy()

        probs = np.empty(len(frame), dtype=float)
        for hand in np.unique(handedness):
            hand_idx = handedness == hand
            raw_hand_surface = self.handedness_surfaces.get(str(hand), self.global_surface)
            hand_surface = (
                cs_hand_weight * raw_hand_surface
                + cs_global_weight * self.global_surface
            )
            probs[hand_idx] = self._surface_lookup(hand_surface, target_x[hand_idx], z_norm[hand_idx])

        unique_batters = np.unique(batter_ids)
        for batter_id in unique_batters:
            if batter_id not in self.batter_surfaces:
                continue
            batter_idx = batter_ids == batter_id
            batter_surface = self.batter_surfaces[batter_id]
            batter_pred = self._surface_lookup(batter_surface, target_x[batter_idx], z_norm[batter_idx])
            raw_weight = self.batter_counts.get(batter_id, 0) / (self.batter_counts.get(batter_id, 0) + self.blend_strength)
            weight = min(raw_weight, cs_batter_weight_cap)
            probs[batter_idx] = weight * batter_pred + (1.0 - weight) * probs[batter_idx]

        probs = np.clip(probs, 1e-4, 1.0 - 1e-4)
        return np.column_stack([1.0 - probs, probs])


@dataclass
class BatterPitchTypeZonePriorLookup:
    broad_swing_rates: dict[tuple[str, str], float]
    broad_contact_rates: dict[tuple[str, str], float]
    broad_whiff_rates: dict[tuple[str, str], float]
    broad_in_play_rates: dict[tuple[str, str], float]
    specific_swing_rates: dict[tuple[str, str, str], float]
    specific_swing_counts: dict[tuple[str, str, str], int]
    specific_contact_rates: dict[tuple[str, str, str], float]
    specific_contact_counts: dict[tuple[str, str, str], int]
    specific_whiff_rates: dict[tuple[str, str, str], float]
    specific_whiff_counts: dict[tuple[str, str, str], int]
    specific_in_play_rates: dict[tuple[str, str, str], float]
    specific_in_play_counts: dict[tuple[str, str, str], int]
    global_swing_rate: float
    global_contact_rate: float
    global_whiff_rate: float
    global_in_play_rate: float
    blend_strength: float = zone_prior_blend_strength

    def _blended_rate(
        self,
        key: tuple[str, str, str],
        broad_key: tuple[str, str],
        *,
        specific_rates: dict[tuple[str, str, str], float],
        specific_counts: dict[tuple[str, str, str], int],
        broad_rates: dict[tuple[str, str], float],
        global_rate: float):

        broad_rate = float(broad_rates.get(broad_key, global_rate))
        specific_count = int(specific_counts.get(key, 0))
        if specific_count <= 0:
            return broad_rate
        specific_rate = float(specific_rates.get(key, broad_rate))
        return float(
            (specific_count * specific_rate + self.blend_strength * broad_rate)
            / (specific_count + self.blend_strength)
        )

    def annotate(self, frame: pd.DataFrame, *, use_observed_location: bool = False):
        working = frame.copy()
        plate_x_column = "observed_plate_x" if use_observed_location and "observed_plate_x" in working.columns else "intended_target_x"
        plate_z_column = "observed_plate_z" if use_observed_location and "observed_plate_z" in working.columns else "intended_target_z"
        zone_bucket = derive_pitch_zone_bucket(
            working[plate_x_column],
            working[plate_z_column],
            working["batter_handedness"],
            working["batter_height_ft"],
        )
        pitch_types = working["pitch_2_type"].fillna("Unknown").astype(str).str.upper()
        batter_ids = working["batter_id"].fillna(-1).astype(str)
        working["pitch_2_zone_bucket"] = zone_bucket.astype(str)

        swing_priors: list[float] = []
        contact_priors: list[float] = []
        whiff_priors: list[float] = []
        in_play_priors: list[float] = []
        for batter_id, pitch_type, bucket in zip(batter_ids, pitch_types, working["pitch_2_zone_bucket"], strict=False):
            key = (str(batter_id), str(pitch_type), str(bucket))
            broad_key = (str(pitch_type), str(bucket))
            swing_priors.append(
                self._blended_rate(
                    key,
                    broad_key,
                    specific_rates=self.specific_swing_rates,
                    specific_counts=self.specific_swing_counts,
                    broad_rates=self.broad_swing_rates,
                    global_rate=self.global_swing_rate,
                )
            )
            contact_priors.append(
                self._blended_rate(
                    key,
                    broad_key,
                    specific_rates=self.specific_contact_rates,
                    specific_counts=self.specific_contact_counts,
                    broad_rates=self.broad_contact_rates,
                    global_rate=self.global_contact_rate,
                )
            )
            whiff_priors.append(
                self._blended_rate(
                    key,
                    broad_key,
                    specific_rates=self.specific_whiff_rates,
                    specific_counts=self.specific_whiff_counts,
                    broad_rates=self.broad_whiff_rates,
                    global_rate=self.global_whiff_rate,
                )
            )
            in_play_priors.append(
                self._blended_rate(
                    key,
                    broad_key,
                    specific_rates=self.specific_in_play_rates,
                    specific_counts=self.specific_in_play_counts,
                    broad_rates=self.broad_in_play_rates,
                    global_rate=self.global_in_play_rate,
                )
            )

        working["batter_pitchtype_zone_swing_prior"] = np.asarray(swing_priors, dtype=float)
        working["batter_pitchtype_zone_contact_prior"] = np.asarray(contact_priors, dtype=float)
        working["batter_pitchtype_zone_whiff_prior"] = np.asarray(whiff_priors, dtype=float)
        working["batter_pitchtype_zone_in_play_prior"] = np.asarray(in_play_priors, dtype=float)
        return working


@dataclass
class ConstantBinaryModel:
    positive_rate: float

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probability = float(np.clip(self.positive_rate, 1e-4, 1.0 - 1e-4))
        return np.repeat(np.array([[1.0 - probability, probability]], dtype=float), len(features), axis=0)


@dataclass
class MulticlassModelWrapper:
    model: object | None
    classes_: np.ndarray
    class_probabilities: np.ndarray
    n_classes: int

    def predict_proba(self, features: np.ndarray):
        if self.model is None:
            return np.repeat(self.class_probabilities[None, :], len(features), axis=0)
        raw = self.model.predict_proba(features)
        full = np.zeros((len(features), self.n_classes), dtype=float)
        for idx, label in enumerate(self.classes_.astype(int)):
            full[:, label] = raw[:, idx]
        row_sums = full.sum(axis=1, keepdims=True)
        return np.divide(full, np.maximum(row_sums, 1e-12), out=np.full_like(full, 1.0 / self.n_classes), where=row_sums > 0)


@dataclass
class FamilyBinaryHead:
    family_column: str
    models: dict[str, object]
    default_model: object

    def predict_proba(self, frame: pd.DataFrame, features: np.ndarray):
        families = frame[self.family_column].fillna("other").astype(str)
        probs = np.zeros((len(frame), 2), dtype=float)
        for family in pd.unique(families):
            mask = families.eq(family).to_numpy()
            model = self.models.get(str(family), self.default_model)
            probs[mask] = model.predict_proba(features[mask])
        return probs


@dataclass
class FamilyMulticlassHead:
    family_column: str
    models: dict[str, MulticlassModelWrapper]
    default_model: MulticlassModelWrapper

    def predict_proba(self, frame: pd.DataFrame, features: np.ndarray):
        families = frame[self.family_column].fillna("other").astype(str)
        probs = np.zeros((len(frame), self.default_model.n_classes), dtype=float)
        for family in pd.unique(families):
            mask = families.eq(family).to_numpy()
            model = self.models.get(str(family), self.default_model)
            probs[mask] = model.predict_proba(features[mask])
        return probs


@dataclass
class ContactQualityModel:
    joint_model: FamilyMulticlassHead
    value_lookup: np.ndarray

    def expected_value(self, frame: pd.DataFrame, features: np.ndarray):
        joint_probs = self.joint_model.predict_proba(frame, features)
        return joint_probs @ self.value_lookup.reshape(-1)


def _fit_encoder(frame: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]):
    numeric_fill_values: dict[str, float] = {}
    for column in numeric_features:
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        fill_value = float(numeric_values.median()) if numeric_values.notna().any() else 0.0
        numeric_fill_values[column] = fill_value
    category_maps: dict[str, dict[str, int]] = {}
    for column in categorical_features:
        values = frame[column].fillna("Unknown").astype(str)
        uniques = sorted(values.unique().tolist())
        category_maps[column] = {value: idx for idx, value in enumerate(uniques)}
    return EncodedFeatureBundle(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        category_maps=category_maps,
        numeric_fill_values=numeric_fill_values,
    )


def _transform_frame(frame: pd.DataFrame, bundle: EncodedFeatureBundle):
    parts: list[np.ndarray] = []
    for column in bundle.numeric_features:
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        missing_mask = numeric_values.isna().to_numpy(dtype=np.float32)
        filled_values = numeric_values.fillna(bundle.numeric_fill_values[column]).to_numpy(dtype=np.float32)
        parts.append(filled_values[:, None])
        parts.append(missing_mask[:, None])
    for column in bundle.categorical_features:
        mapper = bundle.category_maps[column]
        encoded = frame[column].fillna("Unknown").astype(str).map(mapper).fillna(-1).to_numpy(dtype=np.float32)
        parts.append(encoded[:, None])
    return np.hstack(parts)


def _safe_auc(y_true: pd.Series, y_score: np.ndarray):
    labels = pd.to_numeric(y_true, errors="coerce")
    mask = labels.notna()
    labels = labels.loc[mask].astype(int)
    scores = np.asarray(y_score, dtype=float)[mask.to_numpy()]
    if labels.nunique() < 2 or len(scores) != len(labels):
        return np.nan
    return float(roc_auc_score(labels, scores))


def _safe_log_loss(y_true: pd.Series, y_proba: np.ndarray, *, labels: list[int]):
    targets = pd.to_numeric(y_true, errors="coerce")
    mask = targets.notna()
    valid_targets = targets.loc[mask].astype(int)
    valid_proba = np.asarray(y_proba, dtype=float)[mask.to_numpy()]
    if valid_targets.empty or len(valid_targets) != len(valid_proba):
        return np.nan
    try:
        return float(log_loss(valid_targets, valid_proba, labels=labels))
    except ValueError:
        return np.nan


def _safe_accuracy_from_proba(y_true: pd.Series, y_proba: np.ndarray):
    targets = pd.to_numeric(y_true, errors="coerce")
    mask = targets.notna()
    valid_targets = targets.loc[mask].astype(int).to_numpy()
    valid_proba = np.asarray(y_proba, dtype=float)[mask.to_numpy()]
    if len(valid_targets) == 0 or len(valid_targets) != len(valid_proba):
        return np.nan
    predictions = np.argmax(valid_proba, axis=1)
    return float(np.mean(predictions == valid_targets))


def _safe_mae(y_true: pd.Series, y_pred: np.ndarray):
    targets = pd.to_numeric(y_true, errors="coerce")
    mask = targets.notna()
    valid_targets = targets.loc[mask].to_numpy(dtype=float)
    valid_pred = np.asarray(y_pred, dtype=float)[mask.to_numpy()]
    if len(valid_targets) == 0 or len(valid_targets) != len(valid_pred):
        return np.nan
    return float(np.mean(np.abs(valid_targets - valid_pred)))


def _coarse_bucket_target(frame: pd.DataFrame):
    return pd.Series(
        np.select(
            [
                pd.to_numeric(frame["home_run"], errors="coerce").fillna(0).eq(1),
                pd.to_numeric(frame["double_or_triple"], errors="coerce").fillna(0).eq(1),
                pd.to_numeric(frame["single"], errors="coerce").fillna(0).eq(1),
            ],
            [3, 2, 1],
            default=0,
        ),
        index=frame.index,
        dtype=int,
    )


def _joint_contact_target(frame: pd.DataFrame):
    batted_ball_target = frame["batted_ball_type"].map(batted_ball_type_to_idx)
    ev_band_target = frame["ev_band"].map(ev_band_to_idx)
    joint_target = batted_ball_target * len(ev_band_labels) + ev_band_target
    return pd.to_numeric(joint_target, errors="coerce")


def _binary_branch_metrics(frame: pd.DataFrame, features: np.ndarray, model: object, *, label_column: str):
    if frame.empty or len(features) == 0:
        return {"rows": int(len(frame)), "auc": np.nan}
    probabilities = _predict_binary_head(model, frame, features)[:, 1]
    return {
        "rows": int(len(frame)),
        "auc": _safe_auc(frame[label_column], probabilities),
    }


def _bucket_branch_metrics(frame: pd.DataFrame, features: np.ndarray, model: object):
    if frame.empty or len(features) == 0:
        return {"rows": int(len(frame)), "log_loss": np.nan, "accuracy": np.nan}
    target = _coarse_bucket_target(frame)
    probabilities = _predict_multiclass_head(model, frame, features)
    return {
        "rows": int(len(frame)),
        "log_loss": _safe_log_loss(target, probabilities, labels=list(range(4))),
        "accuracy": _safe_accuracy_from_proba(target, probabilities),
    }


def _contact_quality_branch_metrics(
    frame: pd.DataFrame,
    features: np.ndarray,
    model: ContactQualityModel | None):

    if model is None or frame.empty or len(features) == 0:
        return {
            "rows": int(len(frame)),
            "joint_rows": 0,
            "joint_log_loss": np.nan,
            "joint_accuracy": np.nan,
            "expected_value_mae": np.nan,
        }
    joint_target = _joint_contact_target(frame)
    valid_mask = joint_target.notna().to_numpy()
    joint_rows = int(valid_mask.sum())
    if joint_rows == 0:
        return {
            "rows": int(len(frame)),
            "joint_rows": 0,
            "joint_log_loss": np.nan,
            "joint_accuracy": np.nan,
            "expected_value_mae": np.nan,
        }
    valid_frame = frame.loc[valid_mask].copy()
    valid_features = features[valid_mask]
    joint_probabilities = model.joint_model.predict_proba(valid_frame, valid_features)
    expected_values = model.expected_value(valid_frame, valid_features)

    return {
        "rows": int(len(frame)),
        "joint_rows": joint_rows,
        "joint_log_loss": _safe_log_loss(
            joint_target.loc[valid_frame.index],
            joint_probabilities,
            labels=list(range(len(batted_ball_types) * len(ev_band_labels))),
        ),
        "joint_accuracy": _safe_accuracy_from_proba(joint_target.loc[valid_frame.index], joint_probabilities),
        "expected_value_mae": _safe_mae(valid_frame["pitcher_value_on_contact"], expected_values),
    }


def _called_strike_grid_centers():
    x_edges = np.linspace(called_strike_x_range[0], called_strike_x_range[1], called_strike_grid_size + 1)
    z_edges = np.linspace(cs_z_norm_range[0], cs_z_norm_range[1], called_strike_grid_size + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2.0
    return x_centers, z_centers


def _called_strike_surface_from_group(group: pd.DataFrame):
    x_edges = np.linspace(called_strike_x_range[0], called_strike_x_range[1], called_strike_grid_size + 1)
    z_edges = np.linspace(cs_z_norm_range[0], cs_z_norm_range[1], called_strike_grid_size + 1)

    working = group.copy()
    observed_x_column = "observed_plate_x" if "observed_plate_x" in working.columns else "intended_target_x"
    observed_z_column = "observed_plate_z" if "observed_plate_z" in working.columns else "intended_target_z"
    working["observed_plate_x"] = pd.to_numeric(working[observed_x_column], errors="coerce")
    working["observed_plate_z"] = pd.to_numeric(working[observed_z_column], errors="coerce")
    working["batter_height_ft"] = (
        pd.to_numeric(working["batter_height_ft"], errors="coerce")
        .fillna(default_batter_height_ft)
        .clip(lower=5.0, upper=7.5)
    )
    working["called_strike"] = pd.to_numeric(working["called_strike"], errors="coerce").fillna(0.0)
    working["z_norm"] = working["observed_plate_z"] / working["batter_height_ft"]
    working = working.dropna(subset=["observed_plate_x", "z_norm"])
    if working.empty:
        return np.full((called_strike_grid_size, called_strike_grid_size), 0.5, dtype=float)

    strike_hist, _, _ = np.histogram2d(
        working["z_norm"].to_numpy(dtype=float),
        working["observed_plate_x"].to_numpy(dtype=float),
        bins=[z_edges, x_edges],
        weights=working["called_strike"].to_numpy(dtype=float),
    )
    total_hist, _, _ = np.histogram2d(
        working["z_norm"].to_numpy(dtype=float),
        working["observed_plate_x"].to_numpy(dtype=float),
        bins=[z_edges, x_edges],
    )
    smooth_strike = gaussian_filter(strike_hist, sigma=cs_gaussian_sigma, mode="constant", cval=0.0)
    smooth_total = gaussian_filter(total_hist, sigma=cs_gaussian_sigma, mode="constant", cval=0.0)
    return np.divide(
        smooth_strike,
        np.maximum(smooth_total, 1e-8),
        out=np.full_like(smooth_strike, 0.5, dtype=float),
        where=smooth_total > 1e-8,
    )


def _taken_called_strike_frame(frame: pd.DataFrame):
    working = frame.copy()
    working["take"] = pd.to_numeric(working["take"], errors="coerce").fillna(0).astype(int)
    working["called_strike"] = pd.to_numeric(working["called_strike"], errors="coerce").fillna(0).astype(int)
    working["ball"] = pd.to_numeric(working["ball"], errors="coerce").fillna(0).astype(int)
    working = working.loc[working["take"].eq(1) & working["called_strike"].add(working["ball"]).eq(1)].copy()
    return working


def _called_strike_eval_frame(frame: pd.DataFrame):
    working = _taken_called_strike_frame(frame)
    if working.empty:
        return working
    observed_x = pd.to_numeric(
        working["observed_plate_x"] if "observed_plate_x" in working.columns else working["intended_target_x"],
        errors="coerce",
    )
    observed_z = pd.to_numeric(
        working["observed_plate_z"] if "observed_plate_z" in working.columns else working["intended_target_z"],
        errors="coerce",
    )
    working["intended_target_x"] = observed_x.fillna(pd.to_numeric(working["intended_target_x"], errors="coerce"))
    working["intended_target_z"] = observed_z.fillna(pd.to_numeric(working["intended_target_z"], errors="coerce"))
    return working


def fit_called_strike_surface_model(frame: pd.DataFrame):
    working = _taken_called_strike_frame(frame)
    x_centers, z_centers = _called_strike_grid_centers()
    if working.empty:
        neutral_surface = np.full((called_strike_grid_size, called_strike_grid_size), 0.5, dtype=float)
        return CalledStrikeSurfaceModel(
            x_min=called_strike_x_range[0],
            x_max=called_strike_x_range[1],
            z_min=cs_z_norm_range[0],
            z_max=cs_z_norm_range[1],
            x_centers=x_centers,
            z_centers=z_centers,
            global_surface=neutral_surface,
            handedness_surfaces={},
            batter_surfaces={},
            batter_counts={},
            blend_strength=cs_blend_strength,
            default_height_ft=default_batter_height_ft,
        )

    print("Called strike surface: fitting global and batter-specific contour model")
    working["batter_handedness"] = working["batter_handedness"].fillna("Unknown").astype(str)
    working["batter_id"] = working["batter_id"].fillna(-1).astype(str)
    global_surface = _called_strike_surface_from_group(working)

    handedness_surfaces: dict[str, np.ndarray] = {}
    for handedness, hand_group in working.groupby("batter_handedness", dropna=False):
        handedness_surfaces[str(handedness)] = _called_strike_surface_from_group(hand_group)

    batter_surfaces: dict[str, np.ndarray] = {}
    batter_counts: dict[str, int] = {}
    batter_groups = list(working.groupby("batter_id", dropna=False))
    total_batters = len(batter_groups)
    for idx, (batter_id, batter_group) in enumerate(batter_groups, start=1):
        if idx == 1 or idx == total_batters or idx % 100 == 0:
            _progress("Called strike surfaces", idx, total_batters, f"batter_id={batter_id}")
        if len(batter_group) < cs_min_takes:
            continue
        batter_key = str(batter_id)
        batter_counts[batter_key] = int(len(batter_group))
        batter_surfaces[batter_key] = _called_strike_surface_from_group(batter_group)

    return CalledStrikeSurfaceModel(
        x_min=called_strike_x_range[0],
        x_max=called_strike_x_range[1],
        z_min=cs_z_norm_range[0],
        z_max=cs_z_norm_range[1],
        x_centers=x_centers,
        z_centers=z_centers,
        global_surface=global_surface,
        handedness_surfaces=handedness_surfaces,
        batter_surfaces=batter_surfaces,
        batter_counts=batter_counts,
        blend_strength=cs_blend_strength,
        default_height_ft=default_batter_height_ft,
    )


def _conditional_rate_dict(frame: pd.DataFrame, numerator_column: str, denominator_column: str):
    working = frame.copy()
    working[numerator_column] = pd.to_numeric(working[numerator_column], errors="coerce").fillna(0.0)
    working[denominator_column] = pd.to_numeric(working[denominator_column], errors="coerce").fillna(0.0)
    valid = working.loc[working[denominator_column].gt(0)].copy()
    if valid.empty:
        return {}, {}, 0.0
    rates = (
        valid.groupby(["batter_id", "pitch_2_type", "pitch_2_zone_bucket"], dropna=False)[numerator_column]
        .mean()
        .to_dict()
    )
    counts = (
        valid.groupby(["batter_id", "pitch_2_type", "pitch_2_zone_bucket"], dropna=False)[denominator_column]
        .sum()
        .astype(int)
        .to_dict()
    )
    global_rate = float(valid[numerator_column].mean())
    return (
        {(str(batter_id), str(pitch_type), str(bucket)): float(rate) for (batter_id, pitch_type, bucket), rate in rates.items()},
        {(str(batter_id), str(pitch_type), str(bucket)): int(count) for (batter_id, pitch_type, bucket), count in counts.items()},
        global_rate,
    )


def _broad_conditional_rate_dict(frame: pd.DataFrame, numerator_column: str, denominator_column: str):
    working = frame.copy()
    working[numerator_column] = pd.to_numeric(working[numerator_column], errors="coerce").fillna(0.0)
    working[denominator_column] = pd.to_numeric(working[denominator_column], errors="coerce").fillna(0.0)
    valid = working.loc[working[denominator_column].gt(0)].copy()
    if valid.empty:
        return {}, 0.0
    rates = (
        valid.groupby(["pitch_2_type", "pitch_2_zone_bucket"], dropna=False)[numerator_column]
        .mean()
        .to_dict()
    )
    global_rate = float(valid[numerator_column].mean())
    return ({(str(pitch_type), str(bucket)): float(rate) for (pitch_type, bucket), rate in rates.items()}, global_rate)


def fit_batter_pitchtype_zone_prior_lookup(frame: pd.DataFrame):
    working = frame.copy()
    plate_x_column = "observed_plate_x" if "observed_plate_x" in working.columns else "intended_target_x"
    plate_z_column = "observed_plate_z" if "observed_plate_z" in working.columns else "intended_target_z"
    working["pitch_2_zone_bucket"] = derive_pitch_zone_bucket(
        working[plate_x_column],
        working[plate_z_column],
        working["batter_handedness"],
        working["batter_height_ft"],
    )
    working["pitch_2_type"] = working["pitch_2_type"].fillna("Unknown").astype(str).str.upper()
    working["batter_id"] = working["batter_id"].fillna(-1).astype(str)
    working["pitch_2_zone_bucket"] = working["pitch_2_zone_bucket"].fillna("middle_middle").astype(str)
    working["swing"] = pd.to_numeric(working["swing"], errors="coerce").fillna(0.0)
    working["contact"] = pd.to_numeric(working["contact"], errors="coerce").fillna(0.0)
    working["whiff"] = pd.to_numeric(working["whiff"], errors="coerce").fillna(0.0)
    working["in_play"] = pd.to_numeric(working["in_play"], errors="coerce").fillna(0.0)

    broad_swing_rates = (
        working.groupby(["pitch_2_type", "pitch_2_zone_bucket"], dropna=False)["swing"].mean().to_dict()
    )
    specific_swing_rates = (
        working.groupby(["batter_id", "pitch_2_type", "pitch_2_zone_bucket"], dropna=False)["swing"].mean().to_dict()
    )
    specific_swing_counts = (
        working.groupby(["batter_id", "pitch_2_type", "pitch_2_zone_bucket"], dropna=False)["swing"].count().astype(int).to_dict()
    )
    broad_swing_rates = {(str(pitch_type), str(bucket)): float(rate) for (pitch_type, bucket), rate in broad_swing_rates.items()}
    specific_swing_rates = {
        (str(batter_id), str(pitch_type), str(bucket)): float(rate)
        for (batter_id, pitch_type, bucket), rate in specific_swing_rates.items()
    }
    specific_swing_counts = {
        (str(batter_id), str(pitch_type), str(bucket)): int(count)
        for (batter_id, pitch_type, bucket), count in specific_swing_counts.items()
    }

    broad_contact_rates, global_contact_rate = _broad_conditional_rate_dict(working, "contact", "swing")
    specific_contact_rates, specific_contact_counts, _ = _conditional_rate_dict(working, "contact", "swing")
    broad_whiff_rates, global_whiff_rate = _broad_conditional_rate_dict(working, "whiff", "swing")
    specific_whiff_rates, specific_whiff_counts, _ = _conditional_rate_dict(working, "whiff", "swing")
    broad_in_play_rates, global_in_play_rate = _broad_conditional_rate_dict(working, "in_play", "contact")
    specific_in_play_rates, specific_in_play_counts, _ = _conditional_rate_dict(working, "in_play", "contact")

    return BatterPitchTypeZonePriorLookup(
        broad_swing_rates=broad_swing_rates,
        broad_contact_rates=broad_contact_rates,
        broad_whiff_rates=broad_whiff_rates,
        broad_in_play_rates=broad_in_play_rates,
        specific_swing_rates=specific_swing_rates,
        specific_swing_counts=specific_swing_counts,
        specific_contact_rates=specific_contact_rates,
        specific_contact_counts=specific_contact_counts,
        specific_whiff_rates=specific_whiff_rates,
        specific_whiff_counts=specific_whiff_counts,
        specific_in_play_rates=specific_in_play_rates,
        specific_in_play_counts=specific_in_play_counts,
        global_swing_rate=float(working["swing"].mean()) if len(working) else 0.0,
        global_contact_rate=global_contact_rate,
        global_whiff_rate=global_whiff_rate,
        global_in_play_rate=global_in_play_rate,
    )


def _canonicalize_zone_priors(
    frame: pd.DataFrame,
    zone_prior_model: BatterPitchTypeZonePriorLookup,
    *,
    use_observed_location: bool):
    working = frame.copy()
    stale_columns = [column for column in ["pitch_2_zone_bucket", *zone_prior_features] if column in working.columns]
    if stale_columns:
        working = working.drop(columns=stale_columns)
    return zone_prior_model.annotate(working, use_observed_location=use_observed_location)


def _progress(prefix: str, current: int, total: int, detail: str):
    total = max(int(total), 1)
    current = max(0, min(int(current), total))
    bar_len = 20
    filled = int(round(bar_len * current / total))
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"{prefix} [{bar}] {current}/{total} {detail}")


def _season_split(
    frame: pd.DataFrame,
    *,
    train_end: int = 2023,
    val_season: int = 2024,
    test_season: int = 2025):
    seasons = pd.to_numeric(frame["season"], errors="coerce")
    train = frame.loc[seasons <= train_end].copy()
    val = frame.loc[seasons == val_season].copy()
    test = frame.loc[seasons == test_season].copy()
    return train, val, test


def _derive_count_state_value_lookup(frame: pd.DataFrame):
    season_max = int(pd.to_numeric(frame["season"], errors="coerce").max()) if len(frame) else 0
    pitch_level = load_available_pitch_level_history()
    values = pd.DataFrame()
    if not pitch_level.empty:
        season_column = "Season" if "Season" in pitch_level.columns else "season"
        pitch_level_window = pitch_level.loc[pd.to_numeric(pitch_level[season_column], errors="coerce") <= season_max].copy()
        values = build_true_continuation_count_state_values(pitch_level_window)
    if values.empty:
        values = build_empirical_count_state_values(frame)
    lookup = dict(zip(values["count_bucket"], values["future_pitcher_value"]))
    default_value = float(values["future_pitcher_value"].mean()) if len(values) else 0.0
    lookup["_default"] = default_value
    return lookup


def _derive_pitch2_event_premium_lookup(frame: pd.DataFrame, count_state_lookup: dict[str, float]):
    season_max = int(pd.to_numeric(frame["season"], errors="coerce").max()) if len(frame) else 0
    pitch_level = load_available_pitch_level_history()
    if not pitch_level.empty:
        season_column = "Season" if "Season" in pitch_level.columns else "season"
        pitch_level_window = pitch_level.loc[
            pd.to_numeric(pitch_level[season_column], errors="coerce") <= season_max
        ].copy()
        lookup = build_pitch2_event_premium_lookup(pitch_level_window, count_state_lookup)
        if lookup:
            return lookup
    return {
        "called_strike": {"_default": 0.0},
        "ball": {"_default": 0.0},
        "whiff": {"_default": 0.005},
        "foul": {"_default": 0.0},
    }


def _normalized_event_premium_lookup(event_premium_lookup: dict[str, object] | None = None):

    normalized = {
        "called_strike": {"_default": 0.0},
        "ball": {"_default": 0.0},
        "whiff": {"_default": 0.0},
        "foul": {"_default": 0.0},
    }
    if isinstance(event_premium_lookup, dict):
        if any(isinstance(value, dict) for value in event_premium_lookup.values()):
            for event_kind in normalized:
                event_values = event_premium_lookup.get(event_kind)
                if isinstance(event_values, dict):
                    normalized[event_kind] = {
                        str(key): float(value)
                        for key, value in event_values.items()
                    }
    return normalized


def _map_count_state_values(
    frame: pd.DataFrame,
    lookup: dict[str, float],
    event_premium_lookup: dict[str, object] | None = None):

    mapped = frame.copy()
    default_value = float(lookup.get("_default", 0.0))
    normalized_event_lookup = _normalized_event_premium_lookup(event_premium_lookup)
    for bucket_column, output_column, event_kind in [
        ("called_strike_bucket", "called_strike_state_pitcher_value", "called_strike"),
        ("ball_bucket", "ball_state_pitcher_value", "ball"),
        ("whiff_bucket", "whiff_state_pitcher_value", "whiff"),
        ("foul_bucket", "foul_state_pitcher_value", "foul"),
    ]:
        premium_lookup = normalized_event_lookup.get(event_kind, {"_default": 0.0})
        premium_default = float(premium_lookup.get("_default", 0.0))
        mapped[output_column] = (
            mapped[bucket_column].map(lookup).fillna(default_value)
            + mapped[bucket_column].map(premium_lookup).fillna(premium_default)
        )
    return mapped


def _rolling_origin_windows(frame: pd.DataFrame):
    seasons = pd.to_numeric(frame["season"], errors="coerce")
    windows: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []
    for train_end, eval_season in ((2022, 2023), (2023, 2024), (2024, 2025)):
        train = frame.loc[seasons <= train_end].copy()
        eval_frame = frame.loc[seasons == eval_season].copy()
        if len(train) and len(eval_frame):
            windows.append((train, eval_frame, f"train<={train_end}_eval={eval_season}"))
    return windows


def _sample_rows_per_season(frame: pd.DataFrame, max_rows_per_season: int):
    if max_rows_per_season <= 0 or frame.empty:
        return frame.copy()
    seasons = pd.to_numeric(frame["season"], errors="coerce")
    sampled_parts: list[pd.DataFrame] = []
    for season in sorted(seasons.dropna().astype(int).unique().tolist()):
        season_frame = frame.loc[seasons.eq(season)].copy()
        if len(season_frame) > max_rows_per_season:
            season_frame = season_frame.sample(
                n=max_rows_per_season,
                random_state=random_seed,
            ).copy()
        sampled_parts.append(season_frame)
    if not sampled_parts:
        return frame.head(0).copy()
    combined = pd.concat(sampled_parts, ignore_index=False)
    sort_columns = [column for column in ["season", "state_id"] if column in combined.columns]
    if sort_columns:
        combined = combined.sort_values(sort_columns).copy()
    return combined.reset_index(drop=True)


def _predict_binary_head(model: object, frame: pd.DataFrame, features: np.ndarray):
    if isinstance(model, FamilyBinaryHead):
        return model.predict_proba(frame, features)
    return model.predict_proba(features)


def _predict_multiclass_head(model: object, frame: pd.DataFrame, features: np.ndarray):
    if isinstance(model, FamilyMulticlassHead):
        return model.predict_proba(frame, features)
    return model.predict_proba(features)


def _binary_head(train_x: np.ndarray, train_y: pd.Series):
    labels = pd.to_numeric(train_y, errors="coerce").fillna(0).astype(int)
    positive_rate = float(labels.mean()) if len(labels) else 0.5
    if labels.nunique() < 2:
        return ConstantBinaryModel(positive_rate=positive_rate)
    model = HistGradientBoostingClassifier(
        max_depth=5,
        max_iter=220,
        learning_rate=0.05,
        min_samples_leaf=40,
        random_state=random_seed,
    )
    model.fit(train_x, labels)
    return model


def _fit_multiclass_head(train_x: np.ndarray, train_y: pd.Series, n_classes: int):
    labels = pd.to_numeric(train_y, errors="coerce").fillna(0).astype(int)
    class_probabilities = np.bincount(labels, minlength=n_classes).astype(float)
    class_probabilities = class_probabilities / np.maximum(class_probabilities.sum(), 1.0)
    classes = np.sort(labels.unique())
    if len(classes) < 2:
        return MulticlassModelWrapper(
            model=None,
            classes_=np.asarray(classes, dtype=int),
            class_probabilities=class_probabilities,
            n_classes=n_classes,
        )
    model = HistGradientBoostingClassifier(
        max_depth=5,
        max_iter=240,
        learning_rate=0.05,
        min_samples_leaf=30,
        random_state=random_seed,
    )
    model.fit(train_x, labels)

    return MulticlassModelWrapper(
        model=model,
        classes_=np.asarray(model.classes_, dtype=int),
        class_probabilities=class_probabilities,
        n_classes=n_classes,
    )


def _fit_family_binary_head(
    frame: pd.DataFrame,
    features: np.ndarray,
    *,
    label_column: str,
    family_column: str):

    default_model = _binary_head(features, frame[label_column])
    models: dict[str, object] = {}
    families = frame[family_column].fillna("other").astype(str)
    for family in pitch_family_labels:
        family_mask = families.eq(family)
        if int(family_mask.sum()) < family_binary_min_rows:
            continue
        models[family] = _binary_head(features[family_mask.to_numpy()], frame.loc[family_mask, label_column])
    return FamilyBinaryHead(
        family_column=family_column,
        models=models,
        default_model=default_model,
    )


def _fit_family_multiclass_head(
    frame: pd.DataFrame,
    features: np.ndarray,
    *,
    label_column: str,
    family_column: str,
    n_classes: int):

    default_model = _fit_multiclass_head(features, frame[label_column], n_classes)
    models: dict[str, MulticlassModelWrapper] = {}
    families = frame[family_column].fillna("other").astype(str)
    for family in pitch_family_labels:
        family_mask = families.eq(family)
        if int(family_mask.sum()) < family_multiclass_min_rows:
            continue
        models[family] = _fit_multiclass_head(
            features[family_mask.to_numpy()],
            frame.loc[family_mask, label_column],
            n_classes,
        )

    return FamilyMulticlassHead(
        family_column=family_column,
        models=models,
        default_model=default_model,
    )


def _derive_contact_value_lookup(in_play_frame: pd.DataFrame):
    working = in_play_frame.copy()
    working["batted_ball_idx"] = working["batted_ball_type"].map(batted_ball_type_to_idx)
    working["ev_band_idx"] = working["ev_band"].map(ev_band_to_idx)
    working["pitcher_value_on_contact"] = pd.to_numeric(working["pitcher_value_on_contact"], errors="coerce")
    working = working.dropna(subset=["batted_ball_idx", "ev_band_idx", "pitcher_value_on_contact"])
    if working.empty:
        return np.repeat(coarse_contact_bucket_values[0], len(batted_ball_types) * len(ev_band_labels)).reshape(
            len(batted_ball_types),
            len(ev_band_labels),
        )

    lookup = np.full((len(batted_ball_types), len(ev_band_labels)), np.nan, dtype=float)
    cell_means = (
        working.groupby(["batted_ball_idx", "ev_band_idx"], dropna=False)["pitcher_value_on_contact"]
        .mean()
        .to_dict()
    )
    for (batted_ball_idx, ev_band_idx), value in cell_means.items():
        lookup[int(batted_ball_idx), int(ev_band_idx)] = float(value)

    row_means = (
        working.groupby("batted_ball_idx", dropna=False)["pitcher_value_on_contact"]
        .mean()
        .reindex(range(len(batted_ball_types)))
        .to_numpy(dtype=float)
    )
    col_means = (
        working.groupby("ev_band_idx", dropna=False)["pitcher_value_on_contact"]
        .mean()
        .reindex(range(len(ev_band_labels)))
        .to_numpy(dtype=float)
    )
    global_mean = float(pd.to_numeric(working["pitcher_value_on_contact"], errors="coerce").mean())
    for row_idx in range(lookup.shape[0]):
        for col_idx in range(lookup.shape[1]):
            if not np.isfinite(lookup[row_idx, col_idx]):
                row_mean = row_means[row_idx] if np.isfinite(row_means[row_idx]) else global_mean
                col_mean = col_means[col_idx] if np.isfinite(col_means[col_idx]) else global_mean
                lookup[row_idx, col_idx] = float(np.mean([row_mean, col_mean, global_mean]))
    return lookup


def _fit_contact_quality_model(
    in_play_frame: pd.DataFrame,
    features: np.ndarray,
    *,
    family_column: str):

    if in_play_frame.empty:
        return None
    batted_ball_target = in_play_frame["batted_ball_type"].map(batted_ball_type_to_idx)
    ev_band_target = in_play_frame["ev_band"].map(ev_band_to_idx)
    valid_mask = batted_ball_target.notna() & ev_band_target.notna()
    if not valid_mask.any():
        return None
    valid_frame = in_play_frame.loc[valid_mask].copy()
    valid_features = features[valid_mask.to_numpy()]
    valid_frame["batted_ball_idx"] = batted_ball_target.loc[valid_mask].astype(int).to_numpy()
    valid_frame["ev_band_idx"] = ev_band_target.loc[valid_mask].astype(int).to_numpy()
    valid_frame["joint_contact_idx"] = (
        valid_frame["batted_ball_idx"] * len(ev_band_labels) + valid_frame["ev_band_idx"]
    ).astype(int)
    joint_model = _fit_family_multiclass_head(
        valid_frame,
        valid_features,
        label_column="joint_contact_idx",
        family_column=family_column,
        n_classes=len(batted_ball_types) * len(ev_band_labels),
    )
    return ContactQualityModel(
        joint_model=joint_model,
        value_lookup=_derive_contact_value_lookup(valid_frame),
    )


def _annotate_zone_priors_for_bundle(
    frame: pd.DataFrame,
    event_bundle: dict[str, object],
    *,
    use_observed_location: bool):

    active_feature_names = {
        *numeric_swing_features ,
        *numeric_contact_features,
        *numeric_in_play_features,
        *numeric_bucket,
        *contact_quality,
    }
    if not any(feature_name in active_feature_names for feature_name in zone_prior_features):
        return frame.copy()
    zone_prior_model = event_bundle.get("zone_prior_model")
    if isinstance(zone_prior_model, BatterPitchTypeZonePriorLookup):
        return zone_prior_model.annotate(frame, use_observed_location=use_observed_location)
    return frame.copy()


def _encoder_for_head(event_bundle: dict[str, object], head_name: str):
    encoder_key = f"{head_name}_encoder"
    encoder = event_bundle.get(encoder_key)
    if isinstance(encoder, EncodedFeatureBundle):
        return encoder
    return event_bundle["encoder"]


def _features_for_head(frame: pd.DataFrame, event_bundle: dict[str, object], head_name: str):
    return _transform_frame(frame, _encoder_for_head(event_bundle, head_name))


def prepare_target_lookup_frame(targets: pd.DataFrame):
    prepared = targets.copy()
    if "_state_num" not in prepared.columns:
        prepared["_state_num"] = pd.to_numeric(
            prepared["state_id"].astype(str).str.replace("state_", "", regex=False),
            errors="coerce",
        )
    if "_snapshot_date" not in prepared.columns:
        prepared["_snapshot_date"] = pd.to_datetime(prepared["snapshot_date"], errors="coerce")
    return prepared


def _group_index_map(frame: pd.DataFrame, columns: list[str]):
    if frame.empty:
        return {}
    grouped = frame.groupby(columns, sort=False, dropna=False).groups
    return {
        tuple(str(part) for part in key) if isinstance(key, tuple) else (str(key),): group_index.to_numpy()
        for key, group_index in grouped.items()
    }


def build_target_lookup_indexes(targets: pd.DataFrame):
    prepared = prepare_target_lookup_frame(targets)
    normalized = prepared.assign(
        _pitcher_id_key=prepared["pitcher_id"].fillna("").astype(str),
        _batter_id_key=prepared["batter_id"].fillna("").astype(str) if "batter_id" in prepared.columns else "",
        _pitch_1_type_key=prepared["pitch_1_type"].fillna("").astype(str).str.upper(),
        _pitch_1_bucket_key=prepared["pitch_1_bucket"].fillna("").astype(str) if "pitch_1_bucket" in prepared.columns else "",
        _pitch_2_type_key=prepared["pitch_type"].fillna("").astype(str).str.upper(),
        _batter_hand_key=prepared["batter_handedness"].fillna("Unknown").astype(str).str.upper(),
        _count_bucket_key=prepared["count_bucket"].fillna("").astype(str),
        _scope_key=prepared["target_context_scope"].fillna("").astype(str) if "target_context_scope" in prepared.columns else "",
    )
    has_target_context_scope = "target_context_scope" in prepared.columns
    batter_frame = normalized.loc[normalized["_scope_key"].eq(batter_specific_target_scope)].copy() if has_target_context_scope else normalized.iloc[0:0].copy()
    handedness_frame = normalized.loc[normalized["_scope_key"].eq(handidness_target_scope)].copy() if has_target_context_scope else normalized.copy()
    broad_frame = normalized.loc[normalized["_scope_key"].eq(broad_target_scope)].copy() if has_target_context_scope else normalized.copy()
    default_frame = normalized.loc[normalized["_scope_key"].eq(default_target_scope)].copy() if has_target_context_scope else normalized.copy()
    return {
        "batter_specific": _group_index_map(
            batter_frame,
            ["_batter_id_key", "_pitch_1_type_key", "_pitch_1_bucket_key", "_pitch_2_type_key", "_count_bucket_key"],
        ),
        "batter_handedness": _group_index_map(
            handedness_frame,
            [
                "_pitcher_id_key",
                "_pitch_1_type_key",
                "_pitch_1_bucket_key",
                "_pitch_2_type_key",
                "_batter_hand_key",
                "_count_bucket_key",
            ],
        ),
        "broad": _group_index_map(
            broad_frame,
            ["_pitcher_id_key", "_pitch_2_type_key"],
        ),
        "default": _group_index_map(
            default_frame,
            ["_pitch_2_type_key"],
        ),
    }


def _target_lookup_key_for_scope(row: pd.Series, scope: str):
    normalized_pitch_1_type = str(row.get("pitch_1_type", "")).upper()
    normalized_pitch_1_bucket = _pitch_1_bucket_for_row(row)
    normalized_pitch_2_type = str(row.get("pitch_2_type", "")).upper()
    normalized_count_bucket = str(row.get("count_bucket", ""))
    normalized_batter_handedness = str(row.get("batter_handedness", "")).upper()
    if scope == batter_specific_target_scope:
        return (
            str(row.get("batter_id", "")),
            normalized_pitch_1_type,
            normalized_pitch_1_bucket,
            normalized_pitch_2_type,
            normalized_count_bucket,
        )
    if scope == handidness_target_scope:
        return (
            str(row.get("pitcher_id", "")),
            normalized_pitch_1_type,
            normalized_pitch_1_bucket,
            normalized_pitch_2_type,
            normalized_batter_handedness,
            normalized_count_bucket,
        )
    if scope == broad_target_scope:
        return (str(row.get("pitcher_id", "")), normalized_pitch_2_type)
    if scope == default_target_scope:
        return (normalized_pitch_2_type,)
    return None


def _indexed_target_subset(
    row: pd.Series,
    targets: pd.DataFrame,
    scope: str,
    lookup_indexes: dict[str, dict[tuple[str, ...], np.ndarray]] | None):

    if lookup_indexes is None:
        return pd.DataFrame()
    index_name = {
        batter_specific_target_scope: "batter_specific",
        handidness_target_scope: "batter_handedness",
        broad_target_scope: "broad",
        default_target_scope: "default",
    }.get(scope)
    if index_name is None:
        return pd.DataFrame()
    key = _target_lookup_key_for_scope(row, scope)
    if key is None:
        return pd.DataFrame()
    row_index = lookup_indexes.get(index_name, {}).get(key)
    if row_index is None or len(row_index) == 0:
        return pd.DataFrame()
    return targets.loc[row_index].copy()




def _target_state_numbers(targets: pd.DataFrame):
    if "_state_num" in targets.columns:
        return pd.to_numeric(targets["_state_num"], errors="coerce")
    return pd.to_numeric(
        targets["state_id"].astype(str).str.replace("state_", "", regex=False),
        errors="coerce",
    )


def _row_cutoff_mask(row: pd.Series, targets: pd.DataFrame):
    row_date = pd.to_datetime(row.get("game_date"), errors="coerce")
    target_dates = (
        pd.to_datetime(targets["_snapshot_date"], errors="coerce")
        if "_snapshot_date" in targets.columns
        else pd.to_datetime(targets["snapshot_date"], errors="coerce")
    )
    row_state_num = pd.to_numeric(str(row.get("state_id", "")).replace("state_", ""), errors="coerce")
    target_state_num = _target_state_numbers(targets)
    if pd.notna(row_date):
        base_mask = target_dates.lt(row_date) | (target_dates.eq(row_date) & target_state_num.lt(row_state_num))
    else:
        base_mask = target_state_num.lt(row_state_num)
    return base_mask, target_state_num


def _latest_target_rows(targets: pd.DataFrame, mask: pd.Series, target_state_num: pd.Series):
    target_rows = targets.loc[mask].copy()
    if target_rows.empty:
        return pd.DataFrame()
    target_rows = target_rows.assign(_state_num=target_state_num.loc[target_rows.index].to_numpy())
    latest_state_num = target_rows["_state_num"].max()
    return target_rows.loc[target_rows["_state_num"].eq(latest_state_num)].drop(columns="_state_num").copy()


def _rank_target_rows(target_rows: pd.DataFrame):
    if target_rows.empty:
        return target_rows.copy()
    return target_rows.sort_values(
        by=["weight", "n_obs", "rank", "pitch_2_bucket"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def _default_target_rows():
    rows = default_bucket_prior_rows().copy()
    rows["target_context_scope"] = default_target_scope
    rows["candidate_pool"] = broad_target_scope
    return rows


def _pitch_1_bucket_for_row(row: pd.Series):
    existing_bucket = row.get("pitch_1_bucket")
    if existing_bucket is not None and str(existing_bucket):
        return str(existing_bucket)
    return str(
        locate_bucket_id(
            row.get("pitch_1_plate_x"),
            row.get("pitch_1_plate_z"),
            batter_height_ft=row.get("batter_height_ft"),
        )
    )


def _policy_target_rows_for_scope(
    row: pd.Series,
    targets: pd.DataFrame,
    scope: str,
    lookup_indexes: dict[str, dict[tuple[str, ...], np.ndarray]] | None = None):

    indexed_subset = _indexed_target_subset(row, targets, scope, lookup_indexes)
    if not indexed_subset.empty:
        base_mask, target_state_num = _row_cutoff_mask(row, indexed_subset)
        return _rank_target_rows(_latest_target_rows(indexed_subset, base_mask, target_state_num))
    base_mask, target_state_num = _row_cutoff_mask(row, targets)
    has_target_context_scope = "target_context_scope" in targets.columns
    normalized_pitch_1_type = str(row.get("pitch_1_type", "")).upper()
    normalized_pitch_1_bucket = _pitch_1_bucket_for_row(row)
    normalized_pitch_2_type = str(row.get("pitch_2_type", "")).upper()
    normalized_count_bucket = str(row.get("count_bucket", ""))
    normalized_batter_handedness = str(row.get("batter_handedness", "")).upper()
    pitch_1_bucket_mask = (
        targets["pitch_1_bucket"].astype(str).eq(normalized_pitch_1_bucket)
        if "pitch_1_bucket" in targets.columns
        else pd.Series(True, index=targets.index)
    )

    if not has_target_context_scope:
        if scope == batter_specific_target_scope:
            return pd.DataFrame()
        if scope == handidness_target_scope:
            mask = (
                base_mask
                & targets["pitcher_id"].astype(str).eq(str(row["pitcher_id"]))
                & targets["pitch_1_type"].astype(str).str.upper().eq(normalized_pitch_1_type)
                & targets["pitch_type"].astype(str).str.upper().eq(normalized_pitch_2_type)
                & targets["batter_handedness"].astype(str).str.upper().eq(normalized_batter_handedness)
                & targets["count_bucket"].astype(str).eq(normalized_count_bucket)
            )
            return _rank_target_rows(_latest_target_rows(targets, mask, target_state_num))
        if scope == broad_target_scope:
            mask = (
                base_mask
                & targets["pitcher_id"].astype(str).eq(str(row["pitcher_id"]))
                & targets["pitch_type"].astype(str).str.upper().eq(normalized_pitch_2_type)
            )
            return _rank_target_rows(_latest_target_rows(targets, mask, target_state_num))
        if scope == default_target_scope:
            mask = base_mask & targets["pitch_type"].astype(str).str.upper().eq(normalized_pitch_2_type)
            return _rank_target_rows(_latest_target_rows(targets, mask, target_state_num))
        return pd.DataFrame()

    if scope == batter_specific_target_scope:
        mask = (
            base_mask
            & targets["target_context_scope"].astype(str).eq(batter_specific_target_scope)
            & targets["batter_id"].astype(str).eq(str(row["batter_id"]))
            & targets["pitch_1_type"].astype(str).str.upper().eq(normalized_pitch_1_type)
            & pitch_1_bucket_mask
            & targets["pitch_type"].astype(str).str.upper().eq(normalized_pitch_2_type)
            & targets["count_bucket"].astype(str).eq(normalized_count_bucket)
        )
    elif scope == handidness_target_scope:
        mask = (
            base_mask
            & targets["target_context_scope"].astype(str).eq(handidness_target_scope)
            & targets["pitcher_id"].astype(str).eq(str(row["pitcher_id"]))
            & targets["pitch_1_type"].astype(str).str.upper().eq(normalized_pitch_1_type)
            & pitch_1_bucket_mask
            & targets["pitch_type"].astype(str).str.upper().eq(normalized_pitch_2_type)
            & targets["batter_handedness"].astype(str).str.upper().eq(normalized_batter_handedness)
            & targets["count_bucket"].astype(str).eq(normalized_count_bucket)
        )
    elif scope == broad_target_scope:
        mask = (
            base_mask
            & targets["target_context_scope"].astype(str).eq(broad_target_scope)
            & targets["pitcher_id"].astype(str).eq(str(row["pitcher_id"]))
            & targets["pitch_type"].astype(str).str.upper().eq(normalized_pitch_2_type)
        )
    elif scope == default_target_scope:
        mask = (
            base_mask
            & targets["target_context_scope"].astype(str).eq(default_target_scope)
            & targets["pitch_type"].astype(str).str.upper().eq(normalized_pitch_2_type)
        )
    else:
        return pd.DataFrame()

    return _rank_target_rows(_latest_target_rows(targets, mask, target_state_num))


def _extend_candidate_rows(
    selected: list[dict[str, object]],
    seen_buckets: set[str],
    source_rows: pd.DataFrame,
    *,
    candidate_pool: str,
    limit: int):

    if source_rows.empty or len(selected) >= limit:
        return
    ranked_rows = _rank_target_rows(source_rows)
    for _, source_row in ranked_rows.iterrows():
        bucket_id = str(source_row["pitch_2_bucket"])
        if bucket_id in seen_buckets:
            continue
        record = source_row.to_dict()
        record["candidate_pool"] = candidate_pool
        selected.append(record)
        seen_buckets.add(bucket_id)
        if len(selected) >= limit:
            return


def _select_pool_target_rows(
    row: pd.Series,
    targets: pd.DataFrame,
    *,
    candidate_pool: str,
    scope_order: tuple[str, ...],
    seen_buckets: set[str],
    lookup_indexes: dict[str, dict[tuple[str, ...], np.ndarray]] | None = None):

    selected: list[dict[str, object]] = []
    for scope in scope_order:
        source_rows = _policy_target_rows_for_scope(row, targets, scope, lookup_indexes=lookup_indexes)
        if scope == default_target_scope and source_rows.empty:
            source_rows = _default_target_rows()
        _extend_candidate_rows(
            selected,
            seen_buckets,
            source_rows,
            candidate_pool=candidate_pool,
            limit=targets_per_pool,
        )
        if len(selected) >= targets_per_pool:
            break
    return selected


def _select_policy_candidate_target_rows(
    row: pd.Series,
    targets: pd.DataFrame,
    *,
    lookup_indexes: dict[str, dict[tuple[str, ...], np.ndarray]] | None = None):

    seen_buckets: set[str] = set()
    batter_specific_rows = _select_pool_target_rows(
        row,
        targets,
        candidate_pool=batter_specific_target_scope,
        scope_order=(batter_specific_target_scope,),
        seen_buckets=seen_buckets,
        lookup_indexes=lookup_indexes,
    )
    handedness_rows = _select_pool_target_rows(
        row,
        targets,
        candidate_pool=handidness_target_scope,
        scope_order=(
            handidness_target_scope,
            broad_target_scope,
            default_target_scope,
        ),
        seen_buckets=seen_buckets,
        lookup_indexes=lookup_indexes,
    )
    selected = batter_specific_rows + handedness_rows
    if not selected:
        return pd.DataFrame()
    return pd.DataFrame.from_records(selected)


def _policy_target_rows_for_row(
    row: pd.Series,
    targets: pd.DataFrame,
    *,
    lookup_indexes: dict[str, dict[tuple[str, ...], np.ndarray]] | None = None):

    return _select_policy_candidate_target_rows(row, targets, lookup_indexes=lookup_indexes)


def _generated_candidate_rows_from_target_rows(target_rows: pd.DataFrame):
    if target_rows.empty:
        return pd.DataFrame()
    rows = target_rows.copy().reset_index(drop=True)
    if "rank" not in rows.columns:
        rows["rank"] = np.arange(1, len(rows) + 1)
    rows["candidate_bucket"] = rows["pitch_2_bucket"].astype(str)
    return rows.drop_duplicates(subset=["pitch_2_bucket", "candidate_pool"], keep="first").reset_index(drop=True)


def _generated_candidate_frame_from_row(row: pd.Series, generated_target_rows: pd.DataFrame):
    if generated_target_rows.empty:
        return pd.DataFrame()
    candidates = pd.DataFrame([row.to_dict()] * len(generated_target_rows))
    candidates["pitch_2_bucket"] = generated_target_rows["pitch_2_bucket"].astype(str).to_numpy()
    center_x = []
    center_z = []
    batter_height_ft = row.get("batter_height_ft")
    for bucket_id in generated_target_rows["pitch_2_bucket"].astype(str).tolist():
        bucket_x, bucket_z = bucket_center(bucket_id, batter_height_ft=batter_height_ft)
        center_x.append(bucket_x)
        center_z.append(bucket_z)
    candidates["intended_target_x"] = np.asarray(center_x, dtype=float)
    candidates["intended_target_z"] = np.asarray(center_z, dtype=float)
    candidates["candidate_pool"] = generated_target_rows.get(
        "candidate_pool",
        pd.Series([broad_target_scope] * len(generated_target_rows)),
    ).to_numpy()
    candidates["candidate_kind"] = "generated"
    return candidates.reset_index(drop=True)

def _candidate_frame_for_row(
    row: pd.Series,
    targets: pd.DataFrame,
    event_bundle: dict[str, object],
    *,
    lookup_indexes: dict[str, dict[tuple[str, ...], np.ndarray]] | None = None):

    target_rows = _select_policy_candidate_target_rows(row, targets, lookup_indexes=lookup_indexes)
    return _candidate_frame_from_target_rows(row, target_rows, event_bundle)


def _candidate_frame_from_target_rows(
    row: pd.Series,
    target_rows: pd.DataFrame,
    event_bundle: dict[str, object]):

    if target_rows.empty:
        target_rows = _default_target_rows().assign(
            candidate_pool=broad_target_scope,
        )
    generated_target_rows = _generated_candidate_rows_from_target_rows(target_rows)
    if generated_target_rows.empty:
        generated_target_rows = _generated_candidate_rows_from_target_rows(
            _default_target_rows().assign(candidate_pool=broad_target_scope)
        )
    candidates = _generated_candidate_frame_from_row(row, generated_target_rows)
    observed_candidate = pd.DataFrame([row.to_dict()])
    observed_candidate["candidate_kind"] = "observed"
    observed_candidate["candidate_pool"] = "observed"
    candidates["candidate_kind"] = "generated"
    candidates = pd.concat([observed_candidate, candidates], ignore_index=True)
    return candidates.reset_index(drop=True)


def _planner_window_diagnostics(
    planner_view: pd.DataFrame,
    targets: pd.DataFrame,
    event_bundle: dict[str, object],
    *,
    max_states: int = 750):

    frame = planner_view.copy()
    if max_states > 0 and len(frame) > max_states:
        frame = frame.sample(n=max_states, random_state=random_seed).sort_values("state_id").copy()
    print(f"Planner diagnostics: scoring {len(frame):,} states")
    if frame.empty:
        return {
            "rows": 0,
            "avg_planner_gain": np.nan,
            "avg_observed_score": np.nan,
            "avg_best_candidate_score": np.nan,
            "bucket_spacing": np.nan,
        }
    records: list[dict[str, float | str]] = []
    total = len(frame)
    for idx, (_, row) in enumerate(frame.iterrows(), start=1):
        if idx == 1 or idx == total or idx % 100 == 0:
            _progress("Planner diagnostics", idx, total, f"state_id={row['state_id']}")
        candidates = _candidate_frame_for_row(row, targets, event_bundle)
        candidate_scores = planner_expected_pitcher_value(candidates, event_bundle)
        best_idx = int(np.argmax(candidate_scores))
        observed_idx = int(candidates.index[candidates["candidate_kind"].eq("observed")][0])
        records.append(
            {
                "observed_outcome_bucket": row["observed_outcome_bucket"],
                "observed_score": float(candidate_scores[observed_idx]),
                "best_candidate_score": float(candidate_scores[best_idx]),
                "planner_gain": float(candidate_scores[best_idx] - candidate_scores[observed_idx]),
            }
        )
    report = pd.DataFrame.from_records(records)
    bucket_means = (
        report.groupby("observed_outcome_bucket", dropna=False)["observed_score"]
        .mean()
        .reindex(["out", "single", "double_or_triple", "home_run"])
    )
    bucket_spacing = float(bucket_means.diff().dropna().mean()) if bucket_means.notna().sum() >= 2 else np.nan

    return {
        "rows": int(len(report)),
        "avg_planner_gain": float(report["planner_gain"].mean()),
        "avg_observed_score": float(report["observed_score"].mean()),
        "avg_best_candidate_score": float(report["best_candidate_score"].mean()),
        "bucket_spacing": bucket_spacing,
    }


def _fit_event_tree_model_from_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    split_label: str):

    print(f"Event tree: preparing split {split_label}")
    print(f"Event tree: rows train={len(train):,}, val={len(val):,}, test={len(test):,}")
    count_state_lookup = _derive_count_state_value_lookup(train)
    event_premium_lookup = _derive_pitch2_event_premium_lookup(train, count_state_lookup)
    train = _map_count_state_values(train, count_state_lookup, event_premium_lookup)
    val = _map_count_state_values(val, count_state_lookup, event_premium_lookup)
    test = _map_count_state_values(test, count_state_lookup, event_premium_lookup)

    take_mask_train = train["take"].eq(1)
    swing_mask_train = train["swing"].eq(1)
    contact_mask_train = train["contact"].eq(1)
    in_play_mask_train = train["in_play"].eq(1)
    swing_train = train.loc[swing_mask_train].copy()
    contact_train = train.loc[contact_mask_train].copy()
    in_play_train = train.loc[in_play_mask_train].copy()
    swing_val = val.loc[pd.to_numeric(val["swing"], errors="coerce").fillna(0).eq(1)].copy()
    swing_test = test.loc[pd.to_numeric(test["swing"], errors="coerce").fillna(0).eq(1)].copy()
    contact_val = val.loc[pd.to_numeric(val["contact"], errors="coerce").fillna(0).eq(1)].copy()
    contact_test = test.loc[pd.to_numeric(test["contact"], errors="coerce").fillna(0).eq(1)].copy()
    in_play_val = val.loc[pd.to_numeric(val["in_play"], errors="coerce").fillna(0).eq(1)].copy()
    in_play_test = test.loc[pd.to_numeric(test["in_play"], errors="coerce").fillna(0).eq(1)].copy()
    swing_encoder = _fit_encoder(
        train,
        numeric_swing_features ,
        head_categorical_features,
    )
    contact_encoder = _fit_encoder(
        swing_train if len(swing_train) else train,
        numeric_contact_features,
        head_categorical_features,
    )
    in_play_encoder = _fit_encoder(
        contact_train if len(contact_train) else train,
        numeric_in_play_features,
        head_categorical_features,
    )
    bucket_encoder = _fit_encoder(
        in_play_train if len(in_play_train) else train,
        numeric_bucket,
        head_categorical_features,
    )
    contact_quality_encoder = _fit_encoder(
        in_play_train if len(in_play_train) else train,
        contact_quality,
        head_categorical_features,
    )
    features_train_swing = _transform_frame(train, swing_encoder)
    features_val_swing = _transform_frame(val, swing_encoder)
    features_test_swing = _transform_frame(test, swing_encoder)
    features_train_contact = _transform_frame(swing_train, contact_encoder) if len(swing_train) else np.empty((0, 0))
    features_val_contact = _transform_frame(swing_val, contact_encoder) if len(swing_val) else np.empty((0, 0))
    features_test_contact = _transform_frame(swing_test, contact_encoder) if len(swing_test) else np.empty((0, 0))
    features_train_in_play = _transform_frame(contact_train, in_play_encoder) if len(contact_train) else np.empty((0, 0))
    features_val_in_play = _transform_frame(contact_val, in_play_encoder) if len(contact_val) else np.empty((0, 0))
    features_test_in_play = _transform_frame(contact_test, in_play_encoder) if len(contact_test) else np.empty((0, 0))
    features_train_bucket = _transform_frame(in_play_train, bucket_encoder) if len(in_play_train) else np.empty((0, 0))
    features_val_bucket = _transform_frame(in_play_val, bucket_encoder) if len(in_play_val) else np.empty((0, 0))
    features_test_bucket = _transform_frame(in_play_test, bucket_encoder) if len(in_play_test) else np.empty((0, 0))
    features_train_contact_quality = (
        _transform_frame(in_play_train, contact_quality_encoder) if len(in_play_train) else np.empty((0, 0))
    )
    features_val_contact_quality = (
        _transform_frame(in_play_val, contact_quality_encoder) if len(in_play_val) else np.empty((0, 0))
    )
    features_test_contact_quality = (
        _transform_frame(in_play_test, contact_quality_encoder) if len(in_play_test) else np.empty((0, 0))
    )

    print("Event tree: fitting swing/take/contact/in-play heads")
    swing_model = _fit_family_binary_head(
        train,
        features_train_swing,
        label_column="swing",
        family_column="pitch_2_family",
    )
    called_strike_model = fit_called_strike_surface_model(train)
    contact_model = _fit_family_binary_head(
        swing_train,
        features_train_contact,
        label_column="contact",
        family_column="pitch_2_family",
    )
    in_play_model = _fit_family_binary_head(
        contact_train,
        features_train_in_play,
        label_column="in_play",
        family_column="pitch_2_family",
    )

    in_play_train = in_play_train.copy()
    in_play_train["coarse_bucket"] = _coarse_bucket_target(in_play_train).to_numpy()
    bucket_model = _fit_family_multiclass_head(
        in_play_train,
        features_train_bucket,
        label_column="coarse_bucket",
        family_column="pitch_2_family",
        n_classes=4,
    )
    print("Event tree: fitting in-play bucket model")
    contact_quality_model = _fit_contact_quality_model(
        in_play_train,
        features_train_contact_quality,
        family_column="pitch_2_family",
    )

    val_take = _called_strike_eval_frame(val)
    test_take = _called_strike_eval_frame(test)
    metrics = {
        "swing_val_auc": (
            _safe_auc(val["swing"], _predict_binary_head(swing_model, val, features_val_swing)[:, 1]) if len(val) else np.nan
        ),
        "swing_test_auc": (
            _safe_auc(test["swing"], _predict_binary_head(swing_model, test, features_test_swing)[:, 1]) if len(test) else np.nan
        ),
        "called_strike_val_auc": (
            _safe_auc(val_take["called_strike"], called_strike_model.predict_proba(val_take)[:, 1])
            if len(val_take)
            else np.nan
        ),
        "called_strike_test_auc": (
            _safe_auc(test_take["called_strike"], called_strike_model.predict_proba(test_take)[:, 1])
            if len(test_take)
            else np.nan
        ),
        "lower_tree_heldout": {
            "contact_branch": {
                "val": _binary_branch_metrics(swing_val, features_val_contact, contact_model, label_column="contact"),
                "test": _binary_branch_metrics(swing_test, features_test_contact, contact_model, label_column="contact"),
            },
            "in_play_branch": {
                "val": _binary_branch_metrics(contact_val, features_val_in_play, in_play_model, label_column="in_play"),
                "test": _binary_branch_metrics(contact_test, features_test_in_play, in_play_model, label_column="in_play"),
            },
            "bucket_branch": {
                "val": _bucket_branch_metrics(in_play_val, features_val_bucket, bucket_model),
                "test": _bucket_branch_metrics(in_play_test, features_test_bucket, bucket_model),
            },
            "contact_quality_branch": {
                "val": _contact_quality_branch_metrics(in_play_val, features_val_contact_quality, contact_quality_model),
                "test": _contact_quality_branch_metrics(in_play_test, features_test_contact_quality, contact_quality_model),
            },
        },
    }
    return {
        "encoder": swing_encoder,
        "zone_prior_model": None,
        "swing_encoder": swing_encoder,
        "contact_encoder": contact_encoder,
        "in_play_encoder": in_play_encoder,
        "bucket_encoder": bucket_encoder,
        "contact_quality_encoder": contact_quality_encoder,
        "swing_model": swing_model,
        "called_strike_model": called_strike_model,
        "contact_model": contact_model,
        "in_play_model": in_play_model,
        "bucket_model": bucket_model,
        "contact_quality_model": contact_quality_model,
        "metrics": metrics,
        "count_state_lookup": count_state_lookup,
        "event_premium_lookup": event_premium_lookup,
    }


def fit_event_tree_model(
    event_tree_view: pd.DataFrame,
    *,
    train_end: int = 2023,
    val_season: int = 2024,
    test_season: int = 2025):

    train, val, test = _season_split(
        event_tree_view,
        train_end=train_end,
        val_season=val_season,
        test_season=test_season,
    )
    return _fit_event_tree_model_from_splits(
        train,
        val,
        test,
        split_label=f"train<={train_end}, val={val_season}, test={test_season}",
    )


def _pickle_dump(path, payload: object):
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def save_bundles(
    event_bundle: dict[str, object],
    metadata: dict[str, object],
    *,
    event_bundle_path: Path = event_tree_model_path,
    metadata_path: Path = planner_metadata_path):

    _pickle_dump(event_bundle_path, event_bundle)
    _pickle_dump(metadata_path, metadata)


def _bundle_lookup_keys(lookup: object):
    if not isinstance(lookup, dict):
        return []
    return sorted(str(key) for key in lookup.keys())


def _bundle_event_premium_summary(lookup: object):
    if not isinstance(lookup, dict):
        return {}
    summary: dict[str, list[str]] = {}
    for event_kind, event_lookup in lookup.items():
        if isinstance(event_lookup, dict):
            summary[str(event_kind)] = sorted(str(key) for key in event_lookup.keys())
    return summary


def _bundle_component_class_name(component: object):
    return type(component).__name__ if component is not None else "None"


def _research_traceability_metadata(event_bundle: dict[str, object]):
    count_state_lookup = event_bundle.get("count_state_lookup", {})
    event_premium_lookup = event_bundle.get("event_premium_lookup", {})
    return {
        "contract_version": "v2_traceability_v2",
        "research_objective_components": {
            "evaluation_bundle_source": "saved_research_event_bundle",
            "called_strike_runtime_source": "saved_research_called_strike_model",
            "called_strike_model_class": _bundle_component_class_name(event_bundle.get("called_strike_model")),
            "continuation_value_source": "saved_training_split_count_state_lookup",
            "continuation_lookup_keys": _bundle_lookup_keys(count_state_lookup),
            "event_premium_source": "saved_training_split_pitch2_event_premium_lookup",
            "event_premium_lookup_keys": _bundle_event_premium_summary(event_premium_lookup),
            "planner_value_path": "planner_expected_pitcher_value_using_saved_event_bundle",
            "offline_evaluation_contract": (
                "models/v2/evaluate.py scores the saved research event bundle directly and does not apply "
                "deployment-only runtime overrides."
            ),
        },
        "deployment_objective_overrides": {
            "called_strike_model": (
                "Deployment replaces the saved research called-strike model with the shared "
                "StrikeZoneSurfaceStore used by the UI strike-zone contour and the live pitch-1/pitch-2 take branches."
            ),
            "count_state_lookup": (
                "Deployment replaces the saved research continuation lookup with the RE288-style relative "
                "pitcher count-state lookup centered on the 1-1 state."
            ),
            "event_premium_lookup": (
                "Deployment preserves the saved pitch-2 event premium lookup when present and only rebuilds it from "
                "pitch-level history as a runtime fallback if the artifact is missing that lookup."
            ),
        },
    }

def train_from_tables(
    tables: V2Tables,
    *,
    source_path: str,
    max_rows_per_season: int = 0):

    event_tree_view = tables.pitch2_event_tree_view
    if max_rows_per_season > 0:
        print(
            f"Training v2 planner: using deterministic quick sample with up to {max_rows_per_season:,} rows per season"
        )
        event_tree_view = _sample_rows_per_season(event_tree_view, max_rows_per_season)
    seasons = pd.to_numeric(event_tree_view["season"], errors="coerce")
    fold_2025 = assign_crossfit_fold(event_tree_view)
    recommender_train = event_tree_view.loc[seasons.le(2024)].copy()
    recommender_val = event_tree_view.loc[seasons.eq(2025) & fold_2025.ne(4)].copy()
    recommender_test = event_tree_view.loc[seasons.eq(2025) & fold_2025.eq(4)].copy()
    evaluator_train = event_tree_view.loc[seasons.eq(2025) & fold_2025.ne(4)].copy()
    evaluator_val = event_tree_view.loc[seasons.eq(2025) & fold_2025.eq(4)].copy()
    evaluator_test = evaluator_val.copy()
    if recommender_train.empty or evaluator_train.empty or recommender_test.empty:
        raise ValueError("Requested v2 time split failed: one of the required train/eval partitions is empty.")
    print("Training v2 planner: fitting direct event/value model")
    event_bundle = _fit_event_tree_model_from_splits(
        recommender_train,
        recommender_val,
        recommender_test,
        split_label="recommender train=2021-2024, val=2025 fold!=4, test=2025 fold==4",
    )
    evaluator_bundle = _fit_event_tree_model_from_splits(
        evaluator_train,
        evaluator_val,
        evaluator_test,
        split_label="evaluator train=2025 fold!=4, val=2025 fold==4, test=2025 fold==4",
    )
    cross_fit_contract = {
        "bundle_role": "recommender",
        "recommendation_bundle_path": event_tree_model_path.name,
        "evaluation_bundle_path": event_tree_evaluator.name,
        "split_strategy": "time_split_with_2025_hash_holdout",
        "recommender_train_window": "2021-2024",
        "recommender_validation_window": "2025 fold!=4",
        "recommender_test_window": "2025 fold==4",
        "evaluator_train_window": "2025 fold!=4",
        "evaluator_validation_window": "2025 fold==4",
        "evaluator_test_window": "2025 fold==4",
        "evaluation_fold_rule": "2025 fold == 4",
        "evaluation_fraction_of_2025": 0.2,
        "evaluator_training_fraction_of_2025": 0.8,
        "evaluation_window": "2025 fold==4",
    }
    metadata = {
        "source_path": source_path,
        "modeling_mode": "direct_target_response_no_command",
        "event_metrics": event_bundle["metrics"],
        "objective_adjustments": {
            "event_premium_lookup": event_bundle.get("event_premium_lookup", {}),
        },
        "cross_fit_contract": cross_fit_contract,
        **_research_traceability_metadata(event_bundle),
        "feature_spec": {
            "direct_numeric_state_features": numeric_features,
            "categorical_state_features": categorical_features,
            "action_categorical_features": action_categgorical_features,
            "action_numeric_features": action_numeric_features,
            "zone_prior_features": [],
            "head_numeric_features": {
                "swing": numeric_swing_features ,
                "contact": numeric_contact_features,
                "in_play": numeric_in_play_features,
                "bucket": numeric_bucket,
                "contact_quality": contact_quality,
            },
            "head_categorical_features": {
                "swing": head_categorical_features,
                "contact": head_categorical_features,
                "in_play": head_categorical_features,
                "bucket": head_categorical_features,
                "contact_quality": head_categorical_features,
            },
        },
        "debug_run_config": {
            "max_rows_per_season": max_rows_per_season,
        },
    }
    save_bundles(event_bundle, metadata)
    evaluator_metadata = {
        "source_path": source_path,
        "modeling_mode": "direct_target_response_no_command",
        "event_metrics": evaluator_bundle["metrics"],
        "objective_adjustments": {
            "event_premium_lookup": evaluator_bundle.get("event_premium_lookup", {}),
        },
        "cross_fit_contract": {
            **cross_fit_contract,
            "bundle_role": "evaluator",
            "paired_recommendation_bundle_path": event_tree_model_path.name,
        },
        **_research_traceability_metadata(evaluator_bundle),
        "feature_spec": metadata["feature_spec"],
        "debug_run_config": metadata["debug_run_config"],
    }
    save_bundles(
        evaluator_bundle,
        evaluator_metadata,
        event_bundle_path=event_tree_evaluator,
        metadata_path=planner_eval_metadata,
    )
    print("Training v2 planner: saved bundles and metadata")
    return {
        "event_bundle": event_bundle,
        "evaluator_bundle": evaluator_bundle,
        "metadata": metadata,
        "evaluator_metadata": evaluator_metadata,
        "tables": tables,
    }


def planner_expected_pitcher_value(
    frame: pd.DataFrame,
    event_bundle: dict[str, object]):
    
    scored_frame = _map_count_state_values(
        frame,
        event_bundle["count_state_lookup"],
        event_bundle.get("event_premium_lookup"),
    )
    swing_features = _features_for_head(scored_frame, event_bundle, "swing")
    p_swing = _predict_binary_head(event_bundle["swing_model"], scored_frame, swing_features)[:, 1]
    p_take = 1.0 - p_swing
    p_called_strike = event_bundle["called_strike_model"].predict_proba(scored_frame)[:, 1]
    p_ball = 1.0 - p_called_strike
    contact_features = _features_for_head(scored_frame, event_bundle, "contact")
    p_contact = _predict_binary_head(event_bundle["contact_model"], scored_frame, contact_features)[:, 1]
    p_whiff = 1.0 - p_contact
    in_play_features = _features_for_head(scored_frame, event_bundle, "in_play")
    p_in_play = _predict_binary_head(event_bundle["in_play_model"], scored_frame, in_play_features)[:, 1]
    p_foul = 1.0 - p_in_play
    bucket_features = _features_for_head(scored_frame, event_bundle, "bucket")
    bucket_probs = _predict_multiclass_head(event_bundle["bucket_model"], scored_frame, bucket_features)
    contact_quality_model = event_bundle.get("contact_quality_model")
    if isinstance(contact_quality_model, ContactQualityModel):
        contact_quality_features = _features_for_head(scored_frame, event_bundle, "contact_quality")
        expected_contact_pitcher_value = contact_quality_model.expected_value(scored_frame, contact_quality_features)
    else:
        expected_contact_pitcher_value = bucket_probs @ coarse_contact_bucket_values
    return (
        p_take * (
            p_called_strike * scored_frame["called_strike_state_pitcher_value"].to_numpy(dtype=float)
            + p_ball * scored_frame["ball_state_pitcher_value"].to_numpy(dtype=float)
        )
        + p_swing * (
            p_whiff * scored_frame["whiff_state_pitcher_value"].to_numpy(dtype=float)
            + p_contact * (
                p_foul * scored_frame["foul_state_pitcher_value"].to_numpy(dtype=float)
                + p_in_play * expected_contact_pitcher_value
            )
        )
    )
