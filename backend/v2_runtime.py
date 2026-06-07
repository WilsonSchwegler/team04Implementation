from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import pickle
import re
import sys
from typing import Any
from uuid import uuid4
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.v2.config import (
    batter_top3_cache_path,
    event_tree_model_path,
    planner_metadata_path,
    pitch2_plavver_eval_path,
    pitch_target_distributions_path,
    pitcher_top3_cache_path,
)

from models.v2.cache_helpers import (
    append_cached_bucket_records,
    batter_top3_cache_path_for_row,
    load_top3_cache,
    nested_bucket_lookup,
    pitcher_top3_cache_path_for_row,
    top_level_lookup_ids,
)

from models.v2.data import (
    build_pitch2_event_premium_lookup,
    build_pitch_type_shape_reference_lookup,
    derive_relative_pitch_shape_bucket,
    load_available_pitch_level_history,
    pitch_shape_deltas_from_reference_lookup,
)

from models.v2.modeling import (
    batter_specific_target_scope,
    batted_ball_types,
    broad_target_scope,
    default_target_scope,
    ev_band_labels,
    handidness_target_scope,
    ContactQualityModel,
    _default_target_rows,
    _features_for_head,
    _generated_candidate_frame_from_row,
    _map_count_state_values,
    _pitch_1_bucket_for_row,
    _predict_binary_head,
    _predict_multiclass_head,
    _generated_candidate_rows_from_target_rows,
    _select_pool_target_rows,
    _select_policy_candidate_target_rows,
    build_target_lookup_indexes,
    prepare_target_lookup_frame,
)

from models.v2.location_grid import bucket_center, locate_bucket_id, grid_vertical_bounds, strike_zone_bounds

from models.p1.config import (
    event_tree_model_path as p1_event_tree_model_path,
    p1_planner_eval_view_path,
)

from models.p1.modeling import predict_event_probabilities as predict_pitch1_event_probabilities


pitch_type_names = {
    "FF": "4-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Slurve",
    "CH": "Changeup",
    "FS": "Split-Finger",
    "FO": "Forkball",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CS": "Slow Curve",
    "KN": "Knuckleball",
    "EP": "Eephus",
    "FA": "Fastball",
}
fastball_types = {"FF", "FT", "SI", "FC", "FA"}
breaking_types = {"SL", "CU", "KC", "SV", "CS", "ST"}
offspeed_types = {"CH", "FS", "FO", "SC", "KN", "EP"}
min_pitch_count = 20
default_batter_height_ft = 6.0
min_batter_height_ft = 5.0
max_batter_height_ft = 7.5
height_top_ratio = 0.535
height_bot_ratio = 0.27
surface_grid_size = 121
surface_x_range = (-1.5, 1.5)
surface_z_norm_range = (-0.5, 1.5)
surface_gaussian_sigma = 2.0
surface_min_batter_takes = 25
surface_blend_strength = 300.0
surface_batter_weight_cap = 0.12
surface_display_hand_weight = 0.25
surface_display_global_weight = 0.75
synthetic_state_id = "state_9999999"
synthetic_furture_date = pd.Timestamp("2026-01-01")
p2_re288_vals = {
    "0-2": 0.41,
    "1-1": 0.50,
    "2-0": 0.62,
}

data_dir = project_root / "data"
pitch_level_2025 = data_dir / "pitch_level_2025.csv"
prepared_pitch_level_2025 = data_dir / "pitch_level_runtime_2025.parquet"
strike_zone_runtime_store_path = data_dir / "strike_zone_runtime_store.json"
pitch_averages_2025 = data_dir / "pitch_type_averages_2025.csv"
strike_zone_cache_path = project_root / "models" / "v2" / "artifacts" / "strike_zone_contour_cache.json"
outcome_labels = ("out", "single", "double_or_triple", "home_run")
generic_left_batter_id = -1001
generic_right_batter_id = -1002


def _runtime_csv_variant(path: Path):
    return path.with_suffix(".csv")


def _runtime_table_exists(path: Path):
    return path.exists() or _runtime_csv_variant(path).exists()


def _read_runtime_table(path: Path):
    csv_path = _runtime_csv_variant(path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(f"Missing runtime table. Expected {path} or {csv_path}.")


def _component_class_name(component: Any):
    return type(component).__name__ if component is not None else "None"


def _lookup_keys(lookup: Any):
    if not isinstance(lookup, dict):
        return []
    return sorted(str(key) for key in lookup.keys())


def _event_premium_lookup_keys(lookup: Any):
    if not isinstance(lookup, dict):
        return {}
    summary: dict[str, list[str]] = {}
    for event_kind, event_lookup in lookup.items():
        if isinstance(event_lookup, dict):
            summary[str(event_kind)] = sorted(str(key) for key in event_lookup.keys())
    return summary


#exact batter cache key
def _runtime_batter_cache_path(row: pd.Series):
    return batter_top3_cache_path_for_row(row, pitch_1_bucket_getter=_pitch_1_bucket_for_row)


def _generic_batter_cache_id_for_row(row: pd.Series):
    batter_hand = str(row.get("batter_handedness", "")).upper()
    if batter_hand == "L":
        return generic_left_batter_id
    if batter_hand == "R":
        return generic_right_batter_id
    return None


def _runtime_batter_cache_paths(row: pd.Series):
    return [_runtime_batter_cache_path(row)]


def _runtime_pitcher_cache_path(row: pd.Series):
    return pitcher_top3_cache_path_for_row(row, pitch_1_bucket_getter=_pitch_1_bucket_for_row)


def _runtime_pitcher_cache_paths(row: pd.Series):
    return [_runtime_pitcher_cache_path(row)]


#build candidate buckets from caches
def _target_rows_from_top3_caches(
    row: pd.Series,
    *,
    batter_top3_cache: dict[str, object],
    pitcher_top3_cache: dict[str, object],
    targets: pd.DataFrame,
    lookup_indexes: dict[str, dict[tuple[str, ...], np.ndarray]] | None = None):

    selected: list[dict[str, object]] = []
    seen_buckets: set[str] = set()
    stats = {
        "batter_cache_hits": 0,
        "pitcher_cache_hits": 0,
        "batter_fallbacks": 0,
        "pitcher_fallbacks": 0,
    }

    batter_hit = False
    batter_buckets: list[str] = []
    for batter_cache_path in _runtime_batter_cache_paths(row):
        batter_hit, batter_buckets = nested_bucket_lookup(batter_top3_cache, batter_cache_path)
        if batter_hit:
            break

    if batter_hit:
        stats["batter_cache_hits"] = 1
        append_cached_bucket_records(selected, seen_buckets, batter_buckets, candidate_pool=batter_specific_target_scope, target_context_scope=batter_specific_target_scope)
    else:
        stats["batter_fallbacks"] = 1
        selected.extend(
            _select_pool_target_rows(row, targets, candidate_pool=batter_specific_target_scope, scope_order=(batter_specific_target_scope,), seen_buckets=seen_buckets, lookup_indexes=lookup_indexes)
        )

    pitcher_hit = False
    pitcher_buckets: list[str] = []
    for pitcher_cache_path in _runtime_pitcher_cache_paths(row):
        pitcher_hit, pitcher_buckets = nested_bucket_lookup(pitcher_top3_cache, pitcher_cache_path)
        if pitcher_hit:
            break

    if pitcher_hit:
        stats["pitcher_cache_hits"] = 1
        append_cached_bucket_records(selected, seen_buckets, pitcher_buckets, candidate_pool=handidness_target_scope, target_context_scope=handidness_target_scope)
    
    else:
        stats["pitcher_fallbacks"] = 1
        selected.extend(
            _select_pool_target_rows(row, targets, candidate_pool=handidness_target_scope, scope_order=(handidness_target_scope, broad_target_scope, default_target_scope), seen_buckets=seen_buckets, lookup_indexes=lookup_indexes)
        )

    if not selected:
        return pd.DataFrame(), stats
    return pd.DataFrame.from_records(selected), stats


def _load_planner_metadata():
    if not planner_metadata_path.exists():
        return {}
    with planner_metadata_path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload if isinstance(payload, dict) else {}


def _load_strike_zone_source_frame():
    prepared_csv_path = _runtime_csv_variant(prepared_pitch_level_2025)
    if prepared_csv_path.exists():
        return pd.read_csv(prepared_csv_path)
    if prepared_pitch_level_2025.exists():
        return pd.read_parquet(prepared_pitch_level_2025)
    if pitch_level_2025.exists():
        return pd.read_csv(pitch_level_2025)
    raise FileNotFoundError(
        "Missing strike-zone runtime source. Expected either "
        f"{prepared_pitch_level_2025} or {pitch_level_2025}."
    )


def _load_strike_zone_runtime_store_payload():
    if not strike_zone_runtime_store_path.exists():
        return None
    try:
        with strike_zone_runtime_store_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _runtime_traceability_summary(
    *,
    planner_metadata: dict[str, Any],
    saved_called_strike_model_class: str,
    saved_pitch1_called_strike_model_class: str,
    saved_count_lookup_keys: list[str],
    saved_event_premium_lookup_keys: dict[str, list[str]],
    runtime_event_bundle: dict[str, Any],
    runtime_pitch1_event_bundle: dict[str, Any],
    contour_store: Any,
    event_premium_source: str):

    research_components = planner_metadata.get("research_objective_components") or {
        "evaluation_bundle_source": "saved_research_event_bundle",
        "called_strike_runtime_source": "saved_research_called_strike_model",
        "called_strike_model_class": saved_called_strike_model_class,
        "continuation_value_source": "saved_training_split_count_state_lookup",
        "continuation_lookup_keys": saved_count_lookup_keys,
        "event_premium_source": "saved_training_split_pitch2_event_premium_lookup",
        "event_premium_lookup_keys": saved_event_premium_lookup_keys,
        "planner_value_path": "planner_expected_pitcher_value_using_saved_event_bundle",
        "offline_evaluation_contract": (
            "models/v2/evaluate.py scores the saved research event bundle directly and does not apply "
            "deployment-only runtime overrides."
        ),
    }
    deployment_overrides = planner_metadata.get("deployment_objective_overrides") or {
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
    }
    return {
        "contract_version": planner_metadata.get("contract_version", "v2_traceability_v2"),
        "research_objective_components": research_components,
        "deployment_objective_overrides": deployment_overrides,
        "deployment_runtime": {
            "called_strike_runtime_source": "shared_deployed_strike_zone_surface_store",
            "called_strike_runtime_class": _component_class_name(contour_store),
            "pitch1_called_strike_runtime_source": "shared_deployed_strike_zone_surface_store",
            "pitch1_called_strike_runtime_class": _component_class_name(runtime_pitch1_event_bundle.get("called_strike_model")),
            "saved_research_called_strike_model_class": saved_called_strike_model_class,
            "saved_pitch1_called_strike_model_class": saved_pitch1_called_strike_model_class,
            "continuation_value_source": "deployment_re288_relative_pitcher_count_state_lookup",
            "saved_research_continuation_lookup_keys": saved_count_lookup_keys,
            "runtime_continuation_lookup_keys": _lookup_keys(runtime_event_bundle.get("count_state_lookup")),
            "event_premium_source": event_premium_source,
            "runtime_event_premium_lookup_keys": _event_premium_lookup_keys(runtime_event_bundle.get("event_premium_lookup")),
            "applied_overrides": [
                "replaced_v2_called_strike_model_with_shared_strike_zone_surface_store",
                "replaced_p1_called_strike_model_with_shared_strike_zone_surface_store",
                "replaced_v2_count_state_lookup_with_re288_relative_pitcher_count_lookup",
            ]
            + (
                ["rebuilt_v2_event_premium_lookup_from_pitch_level_history"]
                if event_premium_source == "runtime_fallback_from_pitch_level_history"
                else []
            ),
        },
    }


def _safe_float(value: Any, default: float = np.nan):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float(default)


def _safe_int(value: Any, default: int = 0):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return int(numeric) if pd.notna(numeric) else int(default)


def _zone_distance(plate_x: float, plate_z: float):
    horizontal_gap = max(abs(float(plate_x)) - 0.83, 0.0)
    lower_gap = max(1.5 - float(plate_z), 0.0)
    upper_gap = max(float(plate_z) - 3.5, 0.0)
    vertical_gap = lower_gap if lower_gap > 0.0 else upper_gap
    return float(np.sqrt(horizontal_gap**2 + vertical_gap**2))


def _perceived_velocity(velo: float, extension: float):
    extension = float(np.clip(extension, 0.0, 15.0))
    denom = max(60.5 - extension, 1e-6)
    return float(velo * (60.5 / denom))


def _pitch_family_flags(pitch_type: str):
    normalized = str(pitch_type or "").upper()
    return (
        int(normalized in fastball_types),
        int(normalized in breaking_types),
        int(normalized in offspeed_types),
    )


def _format_person_name(name: Any):
    raw = str(name or "").strip()
    if not raw:
        return ""

    def _cap_token(token: str):
        if not token:
            return token
        upper_token = token.upper()
        if upper_token in {"II", "III", "IV", "V", "JR.", "SR."}:
            return upper_token
        if "." in token:
            dotted_parts = token.split(".")
            if all(not part or len(part) <= 2 for part in dotted_parts):
                return ".".join(part.upper() if part else part for part in dotted_parts)
        parts = token.split("-")
        capped_parts: list[str] = []
        for part in parts:
            apostrophe_parts = part.split("'")
            capped_apostrophe = [
                sub[:1].upper() + sub[1:].lower() if sub else sub
                for sub in apostrophe_parts
            ]
            capped = "'".join(capped_apostrophe)
            capped = re.sub(r"\bMc([a-z])", lambda m: f"Mc{m.group(1).upper()}", capped)
            capped_parts.append(capped)
        return "-".join(capped_parts)

    return " ".join(_cap_token(token) for token in raw.split())


def _resolve_matchup_batter_hand(batter_hand: Any, pitcher_hand: Any):
    batter = str(batter_hand or "").upper()
    pitcher = str(pitcher_hand or "").upper()
    if batter in {"L", "R"}:
        return batter
    if batter == "S":
        if pitcher == "R":
            return "L"
        if pitcher == "L":
            return "R"
    return "Unknown"


def _listing_batter_hand(values: pd.Series):
    hands = {
        str(value).upper()
        for value in values.tolist()
        if str(value).upper() in {"L", "R", "S"}
    }
    if {"L", "R"}.issubset(hands):
        return "S"
    if "L" in hands:
        return "L"
    if "R" in hands:
        return "R"
    if "S" in hands:
        return "S"
    return "Unknown"


def _re288_relative_pitcher_count_state_lookup():
    reference_re = float(p2_re288_vals["1-1"])
    lookup = {
        bucket: float(reference_re - float(run_expectancy))
        for bucket, run_expectancy in p2_re288_vals.items()
    }
    lookup["_default"] = float(np.mean(list(lookup.values())))
    return lookup


def _planner_event_breakdown(
    frame: pd.DataFrame,
    event_bundle: dict[str, Any]):

    scored_frame = _map_count_state_values(
        frame,
        event_bundle["count_state_lookup"],
        event_bundle.get("event_premium_lookup"),
    )
    swing_features = _features_for_head(scored_frame, event_bundle, "swing")
    p_swing = _predict_binary_head(event_bundle["swing_model"], scored_frame, swing_features)[:, 1]
    p_take = 1.0 - p_swing
    p_called_strike_given_take = event_bundle["called_strike_model"].predict_proba(scored_frame)[:, 1]
    p_ball_given_take = 1.0 - p_called_strike_given_take
    contact_features = _features_for_head(scored_frame, event_bundle, "contact")
    p_contact_given_swing = _predict_binary_head(event_bundle["contact_model"], scored_frame, contact_features)[:, 1]
    p_whiff_given_swing = 1.0 - p_contact_given_swing
    in_play_features = _features_for_head(scored_frame, event_bundle, "in_play")
    p_in_play_given_contact = _predict_binary_head(event_bundle["in_play_model"], scored_frame, in_play_features)[:, 1]
    p_foul_given_contact = 1.0 - p_in_play_given_contact
    bucket_features = _features_for_head(scored_frame, event_bundle, "bucket")
    outcome_given_in_play_probs = _predict_multiclass_head(event_bundle["bucket_model"], scored_frame, bucket_features)

    contact_quality_model = event_bundle.get("contact_quality_model")
    batted_ball_probs = None
    ev_band_probs = None
    if isinstance(contact_quality_model, ContactQualityModel):
        contact_quality_features = _features_for_head(scored_frame, event_bundle, "contact_quality")
        joint_probs = contact_quality_model.joint_model.predict_proba(scored_frame, contact_quality_features)
        joint_grid = joint_probs.reshape(len(frame), len(batted_ball_types), len(ev_band_labels))
        batted_ball_probs = joint_grid.sum(axis=2)
        ev_band_probs = joint_grid.sum(axis=1)
        expected_contact_pitcher_value = joint_probs @ contact_quality_model.value_lookup.reshape(-1)
    else:
        expected_contact_pitcher_value = outcome_given_in_play_probs @ np.array([0.0, -0.87, -1.245, -2.05], dtype=float)

    p_called_strike = p_take * p_called_strike_given_take
    p_ball = p_take * p_ball_given_take
    p_swinging_strike = p_swing * p_whiff_given_swing
    p_contact = p_swing * p_contact_given_swing
    p_foul_ball = p_contact * p_foul_given_contact
    p_ball_in_play = p_contact * p_in_play_given_contact
    outcome_overall_probs = p_ball_in_play[:, None] * outcome_given_in_play_probs

    expected_pitcher_value = (
        p_take * (
            p_called_strike_given_take * scored_frame["called_strike_state_pitcher_value"].to_numpy(dtype=float)
            + p_ball_given_take * scored_frame["ball_state_pitcher_value"].to_numpy(dtype=float)
        )
        + p_swing * (
            p_whiff_given_swing * scored_frame["whiff_state_pitcher_value"].to_numpy(dtype=float)
            + p_contact_given_swing * (
                p_foul_given_contact * scored_frame["foul_state_pitcher_value"].to_numpy(dtype=float)
                + p_in_play_given_contact * expected_contact_pitcher_value
            )
        )
    )

    return {
        "expected_pitcher_value": expected_pitcher_value,
        "swing_probability": p_swing,
        "take_probability": p_take,
        "called_strike_given_take_probability": p_called_strike_given_take,
        "ball_given_take_probability": p_ball_given_take,
        "contact_given_swing_probability": p_contact_given_swing,
        "whiff_given_swing_probability": p_whiff_given_swing,
        "in_play_given_contact_probability": p_in_play_given_contact,
        "foul_given_contact_probability": p_foul_given_contact,
        "called_strike_probability": p_called_strike,
        "ball_probability": p_ball,
        "swinging_strike_probability": p_swinging_strike,
        "contact_probability": p_contact,
        "foul_ball_probability": p_foul_ball,
        "ball_in_play_probability": p_ball_in_play,
        "outcome_given_in_play_probabilities": outcome_given_in_play_probs,
        "outcome_probabilities": outcome_overall_probs,
        "batted_ball_type_probabilities": batted_ball_probs,
        "exit_velocity_band_probabilities": ev_band_probs,
        "expected_contact_pitcher_value": expected_contact_pitcher_value,
    }


def _planning_spatial_region_value(
    plate_x: float | int | None,
    plate_z: float | int | None,
    batter_hand: str | None):

    px = _safe_float(plate_x)
    pz = _safe_float(plate_z)
    hand = str(batter_hand or "").upper()
    if not np.isfinite(px) or not np.isfinite(pz) or hand not in {"L", "R"}:
        return "unknown"

    relative_x = px if hand == "L" else -px
    horizontal_cut = 0.83 / 3.0
    if relative_x >= horizontal_cut:
        horizontal = "inside"
    elif relative_x <= -horizontal_cut:
        horizontal = "outside"
    else:
        horizontal = "middle"

    zone_height = 3.5 - 1.5
    low_cut = 1.5 + zone_height / 3.0
    high_cut = 1.5 + (2.0 * zone_height / 3.0)
    if pz < 1.5:
        vertical = "down_chase"
    elif pz < low_cut:
        vertical = "low"
    elif pz < high_cut:
        vertical = "middle"
    elif pz <= 3.5:
        vertical = "high"
    else:
        vertical = "up_chase"

    return f"{horizontal}_{vertical}"


def _numeric_band_value(value: float | int | None, *, minimum: float, maximum: float, step: float):
    numeric = _safe_float(value)
    if not np.isfinite(numeric):
        return "unknown"
    clipped = max(minimum, min(numeric, maximum - 1e-9))
    start = minimum + np.floor((clipped - minimum) / step) * step
    end = start + step
    return f"{int(round(start))}_{int(round(end))}"


def _pitch_bucket_key_from_values(
    *,
    pitch_type: str,
    velo: float | int | None,
    spin_rate: float | int | None,
    plate_x: float | int | None,
    plate_z: float | int | None,
    batter_hand: str):
    location_region = _planning_spatial_region_value(plate_x, plate_z, batter_hand)
    velo_band = _numeric_band_value(velo, minimum=70.0, maximum=104.0, step=2.0)
    spin_band = _numeric_band_value(spin_rate, minimum=1000.0, maximum=3600.0, step=200.0)
    return f"{str(pitch_type).upper()}|v={velo_band}|s={spin_band}|loc={location_region}"


def _surface_grid_centers():
    x_edges = np.linspace(surface_x_range[0], surface_x_range[1], surface_grid_size + 1)
    z_edges = np.linspace(surface_z_norm_range[0], surface_z_norm_range[1], surface_grid_size + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2.0
    return x_centers, z_centers


def _surface_from_frame(frame: pd.DataFrame):
    x_edges = np.linspace(surface_x_range[0], surface_x_range[1], surface_grid_size + 1)
    z_edges = np.linspace(surface_z_norm_range[0], surface_z_norm_range[1], surface_grid_size + 1)
    working = frame.copy()
    working["plate_x"] = pd.to_numeric(working["plate_x"], errors="coerce")
    working["plate_z"] = pd.to_numeric(working["plate_z"], errors="coerce")
    working["height_ft"] = (
        pd.to_numeric(working["height_ft"], errors="coerce")
        .fillna(default_batter_height_ft)
        .clip(lower=min_batter_height_ft, upper=max_batter_height_ft)
    )
    working["z_norm"] = working["plate_z"] / working["height_ft"]
    working["is_called_strike"] = pd.to_numeric(working["is_called_strike"], errors="coerce").fillna(0.0)
    working = working.dropna(subset=["plate_x", "z_norm"])
    if working.empty:
        return np.full((surface_grid_size, surface_grid_size), 0.5, dtype=float)

    strike_hist, _, _ = np.histogram2d(
        working["z_norm"].to_numpy(dtype=float),
        working["plate_x"].to_numpy(dtype=float),
        bins=[z_edges, x_edges],
        weights=working["is_called_strike"].to_numpy(dtype=float),
    )
    total_hist, _, _ = np.histogram2d(
        working["z_norm"].to_numpy(dtype=float),
        working["plate_x"].to_numpy(dtype=float),
        bins=[z_edges, x_edges],
    )
    smooth_strike = gaussian_filter(strike_hist, sigma=surface_gaussian_sigma, mode="constant", cval=0.0)
    smooth_total = gaussian_filter(total_hist, sigma=surface_gaussian_sigma, mode="constant", cval=0.0)
    return np.divide(
        smooth_strike,
        np.maximum(smooth_total, 1e-8),
        out=np.full_like(smooth_strike, 0.5, dtype=float),
        where=smooth_total > 1e-8,
    )


def _bilinear_lookup(
    surface: np.ndarray,
    x_centers: np.ndarray,
    z_centers: np.ndarray,
    x: np.ndarray,
    z_norm: np.ndarray):

    x = np.clip(np.asarray(x, dtype=float), x_centers.min(), x_centers.max())
    z_norm = np.clip(np.asarray(z_norm, dtype=float), z_centers.min(), z_centers.max())
    x_pos = np.interp(x, x_centers, np.arange(len(x_centers), dtype=float))
    z_pos = np.interp(z_norm, z_centers, np.arange(len(z_centers), dtype=float))
    x0 = np.clip(np.floor(x_pos).astype(int), 0, len(x_centers) - 1)
    z0 = np.clip(np.floor(z_pos).astype(int), 0, len(z_centers) - 1)
    x1 = np.clip(x0 + 1, 0, len(x_centers) - 1)
    z1 = np.clip(z0 + 1, 0, len(z_centers) - 1)
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


def _interpolate_threshold(z_values: np.ndarray, probs: np.ndarray, idx: int, direction: str):
    if direction == "bottom":
        if idx <= 0:
            return float(z_values[idx])
        p0, p1 = probs[idx - 1], probs[idx]
        z0, z1 = z_values[idx - 1], z_values[idx]
    else:
        if idx >= len(z_values) - 1:
            return float(z_values[idx])
        p0, p1 = probs[idx], probs[idx + 1]
        z0, z1 = z_values[idx], z_values[idx + 1]
    if np.isclose(p0, p1):
        return float(z0)
    alpha = (0.5 - p0) / (p1 - p0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return float(z0 + alpha * (z1 - z0))


def _longest_contiguous_run(indices: list[int]):
    if not indices:
        return []
    best_start = 0
    best_len = 1
    run_start = 0
    run_len = 1
    for pos in range(1, len(indices)):
        if indices[pos] == indices[pos - 1] + 1:
            run_len += 1
            continue
        if run_len > best_len:
            best_start = run_start
            best_len = run_len
        run_start = pos
        run_len = 1
    if run_len > best_len:
        best_start = run_start
        best_len = run_len
    return indices[best_start:best_start + best_len]


def _smooth_series(values: list[float], window: int = 5):
    if len(values) < max(window, 3):
        return values
    kernel = np.ones(int(window), dtype=float)
    kernel = kernel / kernel.sum()
    padded = np.pad(np.asarray(values, dtype=float), (window // 2, window // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(float).tolist()


def _resample_curve(points: list[dict[str, float]], *, target_points: int = 161):
    if len(points) < 3:
        return points
    xs = np.asarray([point["x"] for point in points], dtype=float)
    zs = np.asarray([point["z"] for point in points], dtype=float)
    new_xs = np.linspace(xs[0], xs[-1], max(int(target_points), len(points)))
    new_zs = np.interp(new_xs, xs, zs)
    return [{"x": float(x), "z": float(z)} for x, z in zip(new_xs, new_zs)]


def _build_latest_index_lookup(frame: pd.DataFrame, key_columns: list[str]):
    deduped = frame.drop_duplicates(key_columns, keep="last")
    lookup: dict[tuple[Any, ...], Any] = {}
    for index, row in deduped.iterrows():
        lookup[tuple(row[column] for column in key_columns)] = index
    return lookup


def _build_latest_target_lookup(frame: pd.DataFrame, key_columns: list[str]):
    working = frame.copy()
    state_num = pd.to_numeric(working["state_id"].astype(str).str.replace("state_", "", regex=False), errors="coerce")
    working = working.assign(_state_num=state_num)
    lookup: dict[tuple[Any, ...], pd.DataFrame] = {}
    for key, group in working.groupby(key_columns, dropna=False, sort=False):
        latest_state_num = group["_state_num"].max()
        latest_rows = group.loc[group["_state_num"].eq(latest_state_num)].drop(columns="_state_num").copy()
        normalized_key = key if isinstance(key, tuple) else (key,)
        lookup[normalized_key] = latest_rows.reset_index(drop=True)
    return lookup


@dataclass
class PitchProfile:
    pitch_type: str
    pitch_count: int
    velo: float
    extension: float
    h_mov: float
    v_mov: float
    plate_x: float
    plate_z: float
    release_x: float
    release_y: float
    release_z: float
    spin_axis: float
    spin_rate: float
    vx0: float
    vy0: float
    vz0: float
    ax: float
    ay: float
    az: float


def _pitch_profile_from_averages(
    pitcher_profiles: pd.DataFrame,
    pitcher_id: int,
    pitch_type: str):

    pitch_type = str(pitch_type or "").upper()
    if int(pitcher_id) not in pitcher_profiles.index:
        return None
    
    row = pitcher_profiles.loc[int(pitcher_id)]
    count = _safe_float(row.get(f"{pitch_type}_pitch_count"), 0.0)
    if not np.isfinite(count) or count < min_pitch_count:
        return None
    
    return PitchProfile(
        pitch_type=pitch_type,
        pitch_count=int(count),
        velo=_safe_float(row.get(f"{pitch_type}_velo")),
        extension=_safe_float(row.get(f"{pitch_type}_extension")),
        h_mov=_safe_float(row.get(f"{pitch_type}_h_mov")),
        v_mov=_safe_float(row.get(f"{pitch_type}_v_mov")),
        plate_x=_safe_float(row.get(f"{pitch_type}_plate_x"), 0.0),
        plate_z=_safe_float(row.get(f"{pitch_type}_plate_z"), 2.5),
        release_x=_safe_float(row.get(f"{pitch_type}_release_x")),
        release_y=_safe_float(row.get(f"{pitch_type}_release_y")),
        release_z=_safe_float(row.get(f"{pitch_type}_release_z")),
        spin_axis=_safe_float(row.get(f"{pitch_type}_spin_axis")),
        spin_rate=_safe_float(row.get(f"{pitch_type}_spin_rate")),
        vx0=_safe_float(row.get(f"{pitch_type}_vx0")),
        vy0=_safe_float(row.get(f"{pitch_type}_vy0")),
        vz0=_safe_float(row.get(f"{pitch_type}_vz0")),
        ax=_safe_float(row.get(f"{pitch_type}_ax")),
        ay=_safe_float(row.get(f"{pitch_type}_ay")),
        az=_safe_float(row.get(f"{pitch_type}_az")),
    )


def _pitch_family_label(pitch_type: str):
    pitch_type = str(pitch_type or "").upper()
    if pitch_type in fastball_types:
        return "fastball"
    if pitch_type in breaking_types:
        return "breaking"
    if pitch_type in offspeed_types:
        return "offspeed"
    return "other"


def _simple_line_trajectory(pitch_profile: PitchProfile, *, steps: int = 28):
    steps = max(int(steps), 2)
    x0 = _safe_float(pitch_profile.release_x, 0.0)
    y0 = _safe_float(pitch_profile.release_y, 54.0)
    z0 = _safe_float(pitch_profile.release_z, 6.0)
    x1 = _safe_float(pitch_profile.plate_x, 0.0)
    z1 = _safe_float(pitch_profile.plate_z, 2.5)
    points: list[dict[str, float]] = []
    for idx in range(steps):
        alpha = idx / (steps - 1)
        points.append(
            {
                "x": float(x0 + alpha * (x1 - x0)),
                "y": float(y0 + alpha * (0.0 - y0)),
                "z": float(z0 + alpha * (z1 - z0)),
            }
        )
    return points


def _plate_crossing_time(pitch_profile: PitchProfile):
    release_y = _safe_float(pitch_profile.release_y)
    vy0 = _safe_float(pitch_profile.vy0)
    ay = _safe_float(pitch_profile.ay)
    if not np.isfinite(release_y) or not np.isfinite(vy0) or not np.isfinite(ay):
        return None
    if abs(ay) < 1e-8:
        if abs(vy0) < 1e-8:
            return None
        t = -release_y / vy0
        return float(t) if t > 0 else None
    roots = np.roots([0.5 * ay, vy0, release_y])
    real_positive = [float(root.real) for root in roots if abs(root.imag) < 1e-8 and root.real > 0]
    if not real_positive:
        return None
    return min(real_positive)


def _trajectory_from_profile(pitch_profile: PitchProfile, *, steps: int = 28):
    raw_values = [
        pitch_profile.release_x,
        pitch_profile.release_y,
        pitch_profile.release_z,
        pitch_profile.vx0,
        pitch_profile.vy0,
        pitch_profile.vz0,
        pitch_profile.ax,
        pitch_profile.ay,
        pitch_profile.az,
        pitch_profile.plate_x,
        pitch_profile.plate_z,
    ]
    if not all(np.isfinite(_safe_float(value)) for value in raw_values):
        return _simple_line_trajectory(pitch_profile, steps=steps)

    t_cross = _plate_crossing_time(pitch_profile)
    if t_cross is None or t_cross <= 0:
        return _simple_line_trajectory(pitch_profile, steps=steps)

    times = np.linspace(0.0, t_cross, max(int(steps), 2))
    points: list[dict[str, float]] = []
    for t in times:
        x = pitch_profile.release_x + pitch_profile.vx0 * t + 0.5 * pitch_profile.ax * t * t
        y = pitch_profile.release_y + pitch_profile.vy0 * t + 0.5 * pitch_profile.ay * t * t
        z = pitch_profile.release_z + pitch_profile.vz0 * t + 0.5 * pitch_profile.az * t * t
        points.append({"x": float(x), "y": float(y), "z": float(z)})

    dx = _safe_float(pitch_profile.plate_x, 0.0) - points[-1]["x"]
    dz = _safe_float(pitch_profile.plate_z, 2.5) - points[-1]["z"]
    denom = max(len(points) - 1, 1)
    anchored: list[dict[str, float]] = []
    for idx, point in enumerate(points):
        alpha = idx / denom
        anchored.append(
            {
                "x": float(point["x"] + alpha * dx),
                "y": float(point["y"]),
                "z": float(point["z"] + alpha * dz),
            }
        )
    anchored[-1]["x"] = _safe_float(pitch_profile.plate_x, 0.0)
    anchored[-1]["y"] = 0.0
    anchored[-1]["z"] = _safe_float(pitch_profile.plate_z, 2.5)
    return anchored


def _primary_fastball_profile(pitcher_profiles: pd.DataFrame, pitcher_id: int):
    fastball_profiles = [
        _pitch_profile_from_averages(pitcher_profiles, pitcher_id, pitch_type)
        for pitch_type in fastball_types
    ]
    fastball_profiles = [profile for profile in fastball_profiles if profile is not None]
    if not fastball_profiles:
        return None
    return max(fastball_profiles, key=lambda profile: profile.pitch_count)


def _fastball_relative_deltas(
    pitcher_profiles: pd.DataFrame,
    pitcher_id: int,
    pitch_profile: PitchProfile):

    primary_fastball = _primary_fastball_profile(pitcher_profiles, pitcher_id)
    if primary_fastball is None:
        return 0.0, 0.0, 0.0
    return (
        float(pitch_profile.velo - primary_fastball.velo),
        float(pitch_profile.h_mov - primary_fastball.h_mov),
        float(pitch_profile.v_mov - primary_fastball.v_mov),
    )


@dataclass
class PitchRecord:
    pitch_number: int
    pitch_type: str
    target_x: float
    target_z: float


@dataclass
class AtBatSession:
    at_bat_id: str
    pitcher_id: int
    pitcher_name: str
    pitcher_team: str | None
    pitcher_handedness: str
    batter_id: int
    batter_name: str
    batter_team: str | None
    batter_handedness: str
    balls: int = 0
    strikes: int = 0
    next_pitch_number: int = 1
    terminal: bool = False
    pitch_history: list[PitchRecord] = field(default_factory=list)

    def to_dict(self):
        payload = asdict(self)
        payload["pitch_history"] = [asdict(record) for record in self.pitch_history]
        return payload


class StrikeZoneSurfaceStore:
    def __init__(self, pitch_level_2025: pd.DataFrame):
        self.x_centers, self.z_centers = _surface_grid_centers()
        self.frame = self._prepare_frame(pitch_level_2025)
        self._finalize_from_prepared_frame()

    @classmethod
    def from_prepared_frame(cls, prepared_frame: pd.DataFrame):
        instance = cls.__new__(cls)
        instance.x_centers, instance.z_centers = _surface_grid_centers()
        instance.frame = prepared_frame.copy()
        instance._finalize_from_prepared_frame()
        return instance

    @classmethod
    def from_serialized_payload(cls, payload: dict[str, Any]):
        instance = cls.__new__(cls)
        instance.x_centers = np.asarray(payload.get("x_centers", _surface_grid_centers()[0]), dtype=float)
        instance.z_centers = np.asarray(payload.get("z_centers", _surface_grid_centers()[1]), dtype=float)
        height_lookup_payload = payload.get("height_lookup") or {}
        batter_counts_payload = payload.get("batter_counts") or {}
        batter_surfaces_payload = payload.get("batter_surfaces") or {}
        handedness_surfaces_payload = payload.get("handedness_surfaces") or {}

        instance.height_lookup = pd.Series(
            {
                int(batter_id): float(height_ft)
                for batter_id, height_ft in height_lookup_payload.items()
            },
            dtype=float,
        )
        instance.global_surface = np.asarray(payload.get("global_surface", []), dtype=float)
        instance.handedness_surfaces = {
            str(hand): np.asarray(surface, dtype=float)
            for hand, surface in handedness_surfaces_payload.items()
        }
        instance.batter_surfaces = {
            int(batter_id): np.asarray(surface, dtype=float)
            for batter_id, surface in batter_surfaces_payload.items()
        }
        instance.batter_counts = {
            int(batter_id): int(count)
            for batter_id, count in batter_counts_payload.items()
        }
        instance.frame = pd.DataFrame(
            {
                "batter_id": list(instance.height_lookup.index.astype(int)),
                "batter_hand": ["Unknown"] * len(instance.height_lookup),
                "plate_x": [0.0] * len(instance.height_lookup),
                "plate_z": [0.0] * len(instance.height_lookup),
                "height_ft": instance.height_lookup.to_numpy(dtype=float),
                "is_called_strike": [0] * len(instance.height_lookup),
            }
        )
        instance.contour_cache = instance._load_or_build_contour_cache()
        return instance

    def _finalize_from_prepared_frame(self):
        self.height_lookup = self.frame.groupby("batter_id", dropna=False)["height_ft"].median()
        self.global_surface = _surface_from_frame(self.frame)
        self.handedness_surfaces: dict[str, np.ndarray] = {}
        for hand, group in self.frame.groupby("batter_hand", dropna=False):
            self.handedness_surfaces[str(hand)] = _surface_from_frame(group)
        self.batter_surfaces: dict[int, np.ndarray] = {}
        self.batter_counts = self.frame.groupby("batter_id", dropna=False).size().to_dict()
        self.contour_cache = self._load_or_build_contour_cache()

    def _prepare_frame(self, pitch_level_2025: pd.DataFrame):
        expected_columns = {"batter_id", "batter_hand", "plate_x", "plate_z", "height_ft", "is_called_strike"}
        working = pitch_level_2025.copy()
        if expected_columns.issubset(set(working.columns)):
            working["batter_id"] = pd.to_numeric(working["batter_id"], errors="coerce").fillna(-1).astype(int)
            working["batter_hand"] = working["batter_hand"].fillna("Unknown").astype(str).str.upper()
            working["height_ft"] = (
                pd.to_numeric(working["height_ft"], errors="coerce")
                .fillna(default_batter_height_ft)
                .clip(lower=min_batter_height_ft, upper=max_batter_height_ft)
            )
            working["is_called_strike"] = pd.to_numeric(working["is_called_strike"], errors="coerce").fillna(0).astype(int)
            return working[["batter_id", "batter_hand", "plate_x", "plate_z", "height_ft", "is_called_strike"]].copy()
        working = working.loc[
            pd.to_numeric(working["is_called_strike"], errors="coerce").fillna(0).add(
                pd.to_numeric(working["is_ball"], errors="coerce").fillna(0)
            ).eq(1)
        ].copy()
        top_height = pd.to_numeric(working["sz_top"], errors="coerce") / height_top_ratio
        bot_height = pd.to_numeric(working["sz_bot"], errors="coerce") / height_bot_ratio
        working["height_ft"] = (
            pd.concat([top_height, bot_height], axis=1)
            .mean(axis=1, skipna=True)
            .fillna(default_batter_height_ft)
            .clip(lower=min_batter_height_ft, upper=max_batter_height_ft)
        )
        working["batter_id"] = pd.to_numeric(working["batter"], errors="coerce").fillna(-1).astype(int)
        working["batter_hand"] = working["batter_hand"].fillna("Unknown").astype(str).str.upper()
        return working[["batter_id", "batter_hand", "plate_x", "plate_z", "height_ft", "is_called_strike"]].copy()

    def _batter_surface(self, batter_id: int, batter_hand: str):
        batter_id = int(batter_id)
        batter_hand = str(batter_hand or "Unknown").upper()
        count = int(self.batter_counts.get(batter_id, 0))
        if count >= surface_min_batter_takes:
            if batter_id not in self.batter_surfaces:
                batter_rows = self.frame.loc[self.frame["batter_id"].eq(batter_id)].copy()
                self.batter_surfaces[batter_id] = _surface_from_frame(batter_rows)
            return self.batter_surfaces[batter_id], count, "batter"
        if batter_hand in self.handedness_surfaces:
            return self.handedness_surfaces[batter_hand], count, "handedness"
        return self.global_surface, count, "global"

    def _blended_surface(self, batter_id: int, batter_hand: str, *, include_global: bool):
        surface, count, source_level = self._batter_surface(batter_id, batter_hand)
        hand_surface = self.handedness_surfaces.get(str(batter_hand or "Unknown").upper(), self.global_surface)
        if include_global:
            base_surface = (
                surface_display_hand_weight * hand_surface
                + surface_display_global_weight * self.global_surface
            )
        else:
            base_surface = hand_surface

        if source_level == "batter":
            weight = min(count / (count + surface_blend_strength), surface_batter_weight_cap)
            blended = weight * surface + (1.0 - weight) * base_surface
        elif source_level == "handedness" and include_global:
            blended = base_surface
        else:
            blended = surface
        return blended, count, source_level

    def _contour_cache_key(self, batter_id: int, batter_hand: str) -> str:
        return f"{int(batter_id)}|{str(batter_hand or 'Unknown').upper()}"

    def _clone_contour_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            **payload,
            "contour_points": [dict(point) for point in payload.get("contour_points", [])],
        }

    def _load_or_build_contour_cache(self) -> dict[str, dict[str, Any]]:
        if strike_zone_cache_path.exists():
            try:
                with strike_zone_cache_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                pass
        cache: dict[str, dict[str, Any]] = {}
        for batter_id in sorted(int(bid) for bid in self.height_lookup.index.tolist()):
            for batter_hand in ("L", "R", "Unknown"):
                key = self._contour_cache_key(batter_id, batter_hand)
                cache[key] = self._build_strike_zone_contour_payload(batter_id, batter_hand)
        try:
            strike_zone_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with strike_zone_cache_path.open("w", encoding="utf-8") as handle:
                json.dump(cache, handle)
        except Exception:
            pass
        return cache

    def _build_strike_zone_contour_payload(self, batter_id: int, batter_hand: str):
        blended, count, source_level = self._blended_surface(batter_id, batter_hand, include_global=True)

        batter_height = float(np.clip(_safe_float(self.height_lookup.get(int(batter_id), default_batter_height_ft), default_batter_height_ft), min_batter_height_ft, max_batter_height_ft))
        zone_bottom_ft, zone_top_ft = strike_zone_bounds(batter_height_ft=batter_height)
        grid_bottom_ft, grid_top_ft = grid_vertical_bounds(batter_height_ft=batter_height)
        candidate_rows: list[tuple[int, float, float, float]] = []
        for xi, x in enumerate(self.x_centers):
            column = blended[:, xi]
            inside = column > 0.5
            if not inside.any():
                continue
            bottom_idx = int(np.argmax(inside))
            top_idx = int(len(inside) - 1 - np.argmax(inside[::-1]))
            bottom_z_norm = _interpolate_threshold(self.z_centers, column, bottom_idx, "bottom")
            top_z_norm = _interpolate_threshold(self.z_centers, column, top_idx, "top")
            candidate_rows.append((xi, float(x), float(bottom_z_norm * batter_height), float(top_z_norm * batter_height)))

        run_indices = _longest_contiguous_run([row[0] for row in candidate_rows])
        run_index_set = set(run_indices)
        filtered_rows = [row for row in candidate_rows if row[0] in run_index_set]

        if filtered_rows:
            bottom_smoothed = _smooth_series([row[2] for row in filtered_rows], window=13)
            top_smoothed = _smooth_series([row[3] for row in filtered_rows], window=13)
            bottom_points = [
                {"x": float(row[1]), "z": float(bottom_smoothed[idx])}
                for idx, row in enumerate(filtered_rows)
            ]
            top_points = [
                {"x": float(row[1]), "z": float(top_smoothed[idx])}
                for idx, row in enumerate(filtered_rows)
            ]
            bottom_points = _resample_curve(bottom_points, target_points=181)
            top_points = _resample_curve(top_points, target_points=181)
        else:
            bottom_points = []
            top_points = []

        contour_points = bottom_points + top_points[::-1]
        if contour_points:
            contour_points.append(contour_points[0])
        return {
            "batter_id": int(batter_id),
            "season": 2025,
            "source_level": source_level,
            "n_taken_pitches": count,
            "batter_height_ft": batter_height,
            "zone_bottom_ft": float(zone_bottom_ft),
            "zone_top_ft": float(zone_top_ft),
            "grid_bottom_ft": float(grid_bottom_ft),
            "grid_top_ft": float(grid_top_ft),
            "contour_points": contour_points,
        }

    def predict_called_strike_probability(
        self,
        *,
        batter_id: int,
        batter_hand: str,
        plate_x: float,
        plate_z: float,
        height_ft: float | None = None):
        blended, count, source_level = self._blended_surface(batter_id, batter_hand, include_global=True)

        batter_height = height_ft
        if batter_height is None or not np.isfinite(float(batter_height)):
            batter_height = _safe_float(self.height_lookup.get(int(batter_id), default_batter_height_ft), default_batter_height_ft)
        batter_height = float(np.clip(batter_height, min_batter_height_ft, max_batter_height_ft))
        z_norm = float(plate_z) / max(batter_height, 1e-6)
        probability = float(
            _bilinear_lookup(
                blended,
                self.x_centers,
                self.z_centers,
                np.array([plate_x], dtype=float),
                np.array([z_norm], dtype=float),
            )[0]
        )
        probability = float(np.clip(probability, 1e-4, 1.0 - 1e-4))
        return {
            "probability": probability,
            "batter_height_ft": batter_height,
            "n_taken_pitches": count,
            "source_level": source_level,
        }

    def predict_proba(self, frame: pd.DataFrame):

        if frame.empty:
            return np.zeros((0, 2), dtype=float)

        batter_ids = pd.to_numeric(frame.get("batter_id", -1), errors="coerce").fillna(-1).astype(int).to_numpy()
        batter_hands = frame.get("batter_handedness", pd.Series("Unknown", index=frame.index)).fillna("Unknown").astype(str).str.upper().to_numpy()
        target_x = pd.to_numeric(frame.get("intended_target_x", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        target_z = pd.to_numeric(frame.get("intended_target_z", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        heights = (
            pd.to_numeric(frame.get("batter_height_ft", pd.Series(np.nan, index=frame.index)), errors="coerce")
            .to_numpy(dtype=float)
        )

        probs = np.empty(len(frame), dtype=float)

        for batter_id, batter_hand in {(int(bid), str(hand)) for bid, hand in zip(batter_ids, batter_hands)}:
            mask = (batter_ids == batter_id) & (batter_hands == batter_hand)
            blended, _, _ = self._blended_surface(batter_id, batter_hand, include_global=True)
            row_heights = heights[mask].copy()
            if row_heights.size:
                default_height = float(
                    np.clip(
                        _safe_float(self.height_lookup.get(int(batter_id), default_batter_height_ft), default_batter_height_ft),
                        min_batter_height_ft,
                        max_batter_height_ft,
                    )
                )

                row_heights = np.where(np.isfinite(row_heights), row_heights, default_height)
                row_heights = np.clip(row_heights, min_batter_height_ft, max_batter_height_ft)
                z_norm = target_z[mask] / np.maximum(row_heights, 1e-6)
                probs[mask] = _bilinear_lookup(
                    blended,
                    self.x_centers,
                    self.z_centers,
                    target_x[mask],
                    z_norm,
                )

        probs = np.clip(probs, 1e-4, 1.0 - 1e-4)
        return np.column_stack([1.0 - probs, probs])

    def strike_zone_contour(self, batter_id: int, batter_hand: str):
        key = self._contour_cache_key(batter_id, batter_hand)
        cached = self.contour_cache.get(key)
        if cached is not None:
            return self._clone_contour_payload(cached)
        payload = self._build_strike_zone_contour_payload(batter_id, batter_hand)
        self.contour_cache[key] = payload
        return self._clone_contour_payload(payload)


class PitchOneResearchService:
    def __init__(
        self,
        planner_view: pd.DataFrame,
        pitch_averages_2025: pd.DataFrame,
        event_bundle: dict[str, Any],
        contour_store: StrikeZoneSurfaceStore):

        self.planner_view = planner_view.copy()
        self.planner_view["game_date"] = pd.to_datetime(self.planner_view["game_date"], errors="coerce")
        self.planner_view["pitch_1_type"] = self.planner_view["pitch_1_type"].fillna("").astype(str).str.upper()
        self.planner_view["batter_handedness"] = self.planner_view["batter_handedness"].fillna("Unknown").astype(str).str.upper()
        self.template_rows = self.planner_view.sort_values(["game_date", "state_id"]).copy()
        self.pitcher_profiles = pitch_averages_2025.set_index("mlbam_id", drop=False)
        self.event_bundle = event_bundle
        self.contour_store = contour_store
        self.max_source_date = self.template_rows["game_date"].max()
        if pd.isna(self.max_source_date):
            self.max_source_date = synthetic_furture_date
        self.deployable_pitch_types_by_pitcher = self._build_pitch_type_lookup()
        self.template_exact_lookup: dict[tuple[Any, ...], Any] = {}
        self.template_hand_lookup: dict[tuple[Any, ...], Any] = {}
        self.template_pitcher_lookup: dict[tuple[Any, ...], Any] = {}
        self.template_pitch_type_lookup: dict[tuple[Any, ...], Any] = {}
        self.pitch_profile_cache: dict[tuple[int, str], PitchProfile | None] = {}
        self.primary_fastball_cache: dict[int, PitchProfile | None] = {}
        self.batter_rows = (
            self.template_rows
            .drop_duplicates("batter_id", keep="last")[["batter_id", "batter_name", "batter_team", "batter_handedness"]]
        )
        self.pitcher_rows = (
            self.template_rows
            .drop_duplicates("pitcher_id", keep="last")[["pitcher_id", "pitcher_name", "pitcher_team", "pitcher_handedness"]]
        )

    def available_pitch_types(self, pitcher_id: int):
        return self.deployable_pitch_types_by_pitcher.get(int(pitcher_id), [])

    def _build_pitch_type_lookup(self):
        lookup: dict[int, list[str]] = {}
        for pitcher_id, row in self.pitcher_profiles.iterrows():
            pitch_types: list[str] = []
            for pitch_type in pitch_type_names:
                count = _safe_float(row.get(f"{pitch_type}_pitch_count"), 0.0)
                if np.isfinite(count) and count >= min_pitch_count:
                    pitch_types.append(pitch_type)
            if pitch_types:
                lookup[int(pitcher_id)] = sorted(set(pitch_types))
        return lookup

    def get_pitch_profile(self, pitcher_id: int, pitch_type: str):
        key = (int(pitcher_id), str(pitch_type).upper())
        if key not in self.pitch_profile_cache:
            self.pitch_profile_cache[key] = _pitch_profile_from_averages(self.pitcher_profiles, pitcher_id, pitch_type)
        return self.pitch_profile_cache[key]

    def _primary_fastball_profile(self, pitcher_id: int):
        pitcher_id = int(pitcher_id)
        if pitcher_id not in self.primary_fastball_cache:
            fastball_profiles = [
                self.get_pitch_profile(pitcher_id, pitch_type)
                for pitch_type in fastball_types
            ]
            fastball_profiles = [profile for profile in fastball_profiles if profile is not None]
            self.primary_fastball_cache[pitcher_id] = (
                max(fastball_profiles, key=lambda profile: profile.pitch_count)
                if fastball_profiles
                else None
            )
        return self.primary_fastball_cache[pitcher_id]

    def _fastball_relative_deltas(self, pitcher_id: int, pitch_profile: PitchProfile):
        primary_fastball = self._primary_fastball_profile(pitcher_id)

        if primary_fastball is None:
            return 0.0, 0.0, 0.0
        
        return (
            float(pitch_profile.velo - primary_fastball.velo),
            float(pitch_profile.h_mov - primary_fastball.h_mov),
            float(pitch_profile.v_mov - primary_fastball.v_mov),
        )

    def _find_template_row(
        self,
        *,
        pitcher_id: int,
        batter_id: int,
        batter_handedness: str,
        pitch_1_type: str):

        pitch_1_type = str(pitch_1_type).upper()
        batter_handedness = str(batter_handedness).upper()
        exact_key = (pitcher_id, batter_id, pitch_1_type)
        if exact_key not in self.template_exact_lookup:
            subset = self.template_rows.loc[
                self.template_rows["pitcher_id"].eq(pitcher_id)
                & self.template_rows["batter_id"].eq(batter_id)
                & self.template_rows["pitch_1_type"].eq(pitch_1_type)
            ]
            self.template_exact_lookup[exact_key] = subset.index[-1] if len(subset) else None

        hand_key = (pitcher_id, pitch_1_type, batter_handedness)
        if hand_key not in self.template_hand_lookup:
            subset = self.template_rows.loc[
                self.template_rows["pitcher_id"].eq(pitcher_id)
                & self.template_rows["pitch_1_type"].eq(pitch_1_type)
                & self.template_rows["batter_handedness"].eq(batter_handedness)
            ]
            self.template_hand_lookup[hand_key] = subset.index[-1] if len(subset) else None

        pitcher_key = (pitcher_id, pitch_1_type)
        if pitcher_key not in self.template_pitcher_lookup:
            subset = self.template_rows.loc[
                self.template_rows["pitcher_id"].eq(pitcher_id)
                & self.template_rows["pitch_1_type"].eq(pitch_1_type)
            ]
            self.template_pitcher_lookup[pitcher_key] = subset.index[-1] if len(subset) else None

        pitchtype_key = (pitch_1_type,)
        if pitchtype_key not in self.template_pitch_type_lookup:
            subset = self.template_rows.loc[self.template_rows["pitch_1_type"].eq(pitch_1_type)]
            self.template_pitch_type_lookup[pitchtype_key] = subset.index[-1] if len(subset) else None

        candidate_specs = [
            ("exact_pitcher_batter_pitchtype", self.template_exact_lookup.get(exact_key)),
            ("pitcher_hand_pitchtype", self.template_hand_lookup.get(hand_key)),
            ("pitcher_pitchtype", self.template_pitcher_lookup.get(pitcher_key)),
            ("pitchtype_only", self.template_pitch_type_lookup.get(pitchtype_key)),
        ]
        for level, index in candidate_specs:
            if index is not None:
                return self.template_rows.loc[index].copy(), level
        raise ValueError(
            f"No historical pitch1 template row found for pitcher {pitcher_id} and pitch type {pitch_1_type}."
        )

    def _apply_lookup_overrides(self, row: pd.Series, pitcher_id: int, batter_id: int, pitch_1_type: str):
        return row.copy()

    def _build_live_row(
        self,
        *,
        pitcher_id: int,
        batter_id: int,
        batter_hand: str,
        pitch_type: str,
        target_x: float,
        target_z: float):
        pitch_profile = self.get_pitch_profile(pitcher_id, pitch_type)
        if pitch_profile is None:
            raise ValueError(f"Pitch type {str(pitch_type).upper()} is not available for pitcher {pitcher_id}.")

        template, match_level = self._find_template_row(
            pitcher_id=pitcher_id,
            batter_id=batter_id,
            batter_handedness=batter_hand,
            pitch_1_type=str(pitch_type).upper(),
        )

        batter_height_ft = self.contour_store.predict_called_strike_probability(
            batter_id=batter_id,
            batter_hand=batter_hand,
            plate_x=float(target_x),
            plate_z=float(target_z),
        )["batter_height_ft"]

        updated = self._apply_lookup_overrides(template, pitcher_id, batter_id, str(pitch_type).upper())
        p1_fastball, p1_breaking, p1_offspeed = _pitch_family_flags(str(pitch_type).upper())
        pitch_delta_velo, pitch_delta_h_mov, pitch_delta_v_mov = self._fastball_relative_deltas(pitcher_id, pitch_profile)
        updated["state_id"] = synthetic_state_id
        updated["season"] = 2025
        updated["game_date"] = self.max_source_date + pd.Timedelta(days=1)
        updated["pitcher_id"] = pitcher_id
        updated["batter_id"] = batter_id
        updated["batter_handedness"] = str(batter_hand).upper()
        updated["batter_height_ft"] = batter_height_ft
        updated["balls_before_p1"] = 0
        updated["strikes_before_p1"] = 0
        updated["count_bucket"] = "0-0"
        updated["outs_when_up"] = 0
        updated["pitch_1_type"] = str(pitch_type).upper()
        updated["pitch_1_family"] = _pitch_family_label(str(pitch_type).upper())
        updated["pitch_1_is_fastball"] = p1_fastball
        updated["pitch_1_is_breaking"] = p1_breaking
        updated["pitch_1_is_offspeed"] = p1_offspeed
        updated["avg_velo"] = pitch_profile.velo
        updated["avg_h_mov"] = pitch_profile.h_mov
        updated["avg_v_mov"] = pitch_profile.v_mov
        updated["avg_spin_rate"] = pitch_profile.spin_rate
        updated["avg_extension"] = pitch_profile.extension
        updated["avg_release_x"] = pitch_profile.release_x
        updated["avg_release_z"] = pitch_profile.release_z
        updated["avg_plate_x"] = pitch_profile.plate_x
        updated["avg_plate_z"] = pitch_profile.plate_z
        updated["delta_velo_vs_primary_fastball"] = pitch_delta_velo
        updated["delta_h_mov_vs_primary_fastball"] = pitch_delta_h_mov
        updated["delta_v_mov_vs_primary_fastball"] = pitch_delta_v_mov
        updated["n_pitches"] = pitch_profile.pitch_count
        updated["intended_target_x"] = float(target_x)
        updated["intended_target_z"] = float(target_z)

        return pd.DataFrame([updated.to_dict()]), pitch_profile, match_level

    def score_pitch(
        self,
        *,
        pitcher_id: int,
        batter_id: int,
        batter_hand: str,
        pitch_type: str,
        target_x: float,
        target_z: float):

        live_frame, pitch_profile, match_level = self._build_live_row(
            pitcher_id=int(pitcher_id),
            batter_id=int(batter_id),
            batter_hand=str(batter_hand).upper(),
            pitch_type=str(pitch_type).upper(),
            target_x=float(target_x),
            target_z=float(target_z),
        )

        probabilities = predict_pitch1_event_probabilities(live_frame, self.event_bundle)
        called_strike_given_take = float(probabilities["p_called_strike"][0])
        ball_given_take = float(probabilities["p_ball"][0])
        if called_strike_given_take > 0.5:
            predicted_pitch2_count_bucket = "0-1"
            predicted_pitch1_mode = "take_called_strike"
        else:
            predicted_pitch2_count_bucket = "1-0"
            predicted_pitch1_mode = "take_ball"
        return {
            "pitch_type": pitch_profile.pitch_type,
            "target_x": float(target_x),
            "target_z": float(target_z),
            "pitch_profile": asdict(pitch_profile),
            "swing_probability": 0.0,
            "take_probability": 1.0,
            "called_strike_given_take_probability": round(called_strike_given_take, 4),
            "ball_given_take_probability": round(ball_given_take, 4),
            "take_strike_probability": round(called_strike_given_take, 4),
            "take_ball_probability": round(ball_given_take, 4),
            "contact_given_swing_probability": 0.0,
            "whiff_given_swing_probability": 0.0,
            "in_play_given_contact_probability": 0.0,
            "foul_given_contact_probability": 0.0,
            "in_play_bucket_probabilities": {
                "out": 1.0,
                "single": 0.0,
                "double_or_triple": 0.0,
                "home_run": 0.0,
            },
            "template_match_level": match_level,
            "called_strike_probability_source": "shared_deployed_batter_hand_league_surface",
            "batter_height_ft": round(float(live_frame.iloc[0]["batter_height_ft"]), 3),
            "predicted_pitch1_mode": predicted_pitch1_mode,
            "predicted_pitch2_count_bucket": predicted_pitch2_count_bucket,
        }


class V2PlannerRuntime:
    def __init__(
        self,
        planner_view: pd.DataFrame,
        targets: pd.DataFrame,
        event_bundle: dict[str, Any],
        pitch_averages_2025: pd.DataFrame,
        pitch_one_service: PitchOneResearchService):

        self.planner_view = planner_view.copy()
        self.planner_view["pitch_1_type"] = self.planner_view["pitch_1_type"].fillna("").astype(str).str.upper()
        self.planner_view["pitch_2_type"] = self.planner_view["pitch_2_type"].fillna("").astype(str).str.upper()
        self.planner_view["batter_handedness"] = self.planner_view["batter_handedness"].fillna("Unknown").astype(str).str.upper()
        self.planner_view["count_bucket"] = self.planner_view["count_bucket"].fillna("").astype(str)
        self.targets = prepare_target_lookup_frame(targets)
        self.targets["pitch_type"] = self.targets["pitch_type"].fillna("").astype(str).str.upper()
        self.targets["pitch_1_type"] = self.targets["pitch_1_type"].fillna("").astype(str).str.upper()
        if "pitch_1_bucket" in self.targets.columns:
            self.targets["pitch_1_bucket"] = self.targets["pitch_1_bucket"].fillna("").astype(str)
        if "pitch_2_bucket" in self.targets.columns:
            self.targets["pitch_2_bucket"] = self.targets["pitch_2_bucket"].fillna("").astype(str)
        if "batter_id" in self.targets.columns:
            self.targets["batter_id"] = self.targets["batter_id"].fillna("").astype(str)
        self.targets["batter_handedness"] = self.targets["batter_handedness"].fillna("Unknown").astype(str).str.upper()
        self.targets["count_bucket"] = self.targets["count_bucket"].fillna("").astype(str)
        if "target_context_scope" in self.targets.columns:
            self.targets["target_context_scope"] = self.targets["target_context_scope"].fillna("").astype(str)
        self.target_lookup_indexes = build_target_lookup_indexes(self.targets)
        self.batter_top3_cache_path = Path(batter_top3_cache_path)
        self.pitcher_top3_cache_path = Path(pitcher_top3_cache_path)
        self.batter_top3_cache = load_top3_cache(self.batter_top3_cache_path)
        self.pitcher_top3_cache = load_top3_cache(self.pitcher_top3_cache_path)
        self.batter_cache_ids = top_level_lookup_ids(self.batter_top3_cache)
        self.pitcher_cache_ids = top_level_lookup_ids(self.pitcher_top3_cache)
        self.event_bundle = event_bundle
        self.pitch_one_service = pitch_one_service
        self.pitcher_profiles = pitch_averages_2025.set_index("mlbam_id", drop=False)
        self.runtime_view = self.planner_view.copy()
        seasons = pd.to_numeric(self.runtime_view["season"], errors="coerce")
        self.runtime_view_2025 = self.runtime_view.loc[seasons.eq(2025)].copy()
        self.max_source_date = pd.to_datetime(self.runtime_view["game_date"], errors="coerce").max()

        if pd.isna(self.max_source_date):
            self.max_source_date = synthetic_furture_date
        valid_contour_batter_ids = {
            int(batter_id)
            for batter_id in pd.Index(self.pitch_one_service.contour_store.height_lookup.index)
            if pd.notna(batter_id)
        }
        batter_role_counts = (
            pd.to_numeric(self.runtime_view_2025["batter_id"], errors="coerce")
            .dropna()
            .astype(int)
            .value_counts()
        )
        pitcher_role_counts = (
            pd.to_numeric(self.runtime_view_2025["pitcher_id"], errors="coerce")
            .dropna()
            .astype(int)
            .value_counts()
        )
        eligible_batter_ids = {
            batter_id
            for batter_id in valid_contour_batter_ids
            if (
                int(batter_role_counts.get(batter_id, 0)) > 0
                and (
                    int(pitcher_role_counts.get(batter_id, 0)) == 0
                    or int(batter_role_counts.get(batter_id, 0)) >= 20
                    or int(batter_role_counts.get(batter_id, 0)) >= 0.5 * int(pitcher_role_counts.get(batter_id, 0))
                )
            )
        }
        if self.batter_cache_ids:
            eligible_batter_ids &= self.batter_cache_ids
        eligible_batter_frame = self.runtime_view_2025.loc[
            pd.to_numeric(self.runtime_view_2025["batter_id"], errors="coerce").fillna(-1).astype(int).isin(eligible_batter_ids)
        ].copy()

        batter_hand_summary = (
            eligible_batter_frame.groupby("batter_id", dropna=False)["batter_handedness"]
            .apply(_listing_batter_hand)
            .reset_index(name="batter_handedness")
        )
        batter_hand_summary["batter_is_switch_hitter"] = (
            batter_hand_summary["batter_handedness"].astype(str).str.upper().eq("S")
        )
        self.batter_rows = (
            eligible_batter_frame
            .sort_values(["game_date"])
            .drop_duplicates("batter_id", keep="last")[["batter_id", "batter_name", "batter_team"]]
            .merge(batter_hand_summary, on="batter_id", how="left")
            .copy()
        )
        self.pitcher_rows = (
            self.runtime_view_2025.loc[
                pd.to_numeric(self.runtime_view_2025["pitcher_id"], errors="coerce").fillna(-1).astype(int).isin(
                    self.pitcher_cache_ids if self.pitcher_cache_ids else set()
                )
            ].sort_values(["game_date"])
            .drop_duplicates("pitcher_id", keep="last")[["pitcher_id", "pitcher_name", "pitcher_team", "pitcher_handedness"]]
            .copy()
        )
        if not self.pitcher_cache_ids:
            self.pitcher_rows = (
                self.runtime_view_2025.sort_values(["game_date"])
                .drop_duplicates("pitcher_id", keep="last")[["pitcher_id", "pitcher_name", "pitcher_team", "pitcher_handedness"]]
                .copy()
            )
        self.batter_rows["batter_name"] = self.batter_rows["batter_name"].map(_format_person_name)
        self.pitcher_rows["pitcher_name"] = self.pitcher_rows["pitcher_name"].map(_format_person_name)
        generic_batters = pd.DataFrame([
            {"batter_id": -1001, "batter_name": "Generic LHB", "batter_team": "", "batter_handedness": "L", "batter_is_switch_hitter": False},
            {"batter_id": -1002, "batter_name": "Generic RHB", "batter_team": "", "batter_handedness": "R", "batter_is_switch_hitter": False},
        ])
        self.batter_rows = pd.concat([generic_batters, self.batter_rows], ignore_index=True)
        self.batter_lookup = self.batter_rows.set_index("batter_id", drop=False)
        self.pitcher_lookup = self.pitcher_rows.set_index("pitcher_id", drop=False)
        self.template_rows = self.planner_view.sort_values(["game_date", "state_id"]).copy()
        self.template_exact_lookup: dict[tuple[Any, ...], Any] = {}
        self.template_hand_lookup: dict[tuple[Any, ...], Any] = {}
        self.template_pitchtype_count_lookup: dict[tuple[Any, ...], Any] = {}
        self.template_pitchtype_lookup: dict[tuple[Any, ...], Any] = {}
        self.target_candidate_lookup: dict[tuple[Any, ...], pd.DataFrame] = {}
        self.pitch_profile_cache: dict[tuple[int, str], PitchProfile | None] = {}
        self.primary_fastball_cache: dict[int, PitchProfile | None] = {}
        self.pitch_1_shape_reference_lookup = build_pitch_type_shape_reference_lookup(
            self.runtime_view,
            prefix="pitch_1",
        )
        self.pitch_2_shape_reference_lookup = build_pitch_type_shape_reference_lookup(
            self.runtime_view,
            prefix="pitch_2",
        )
        self.deployable_pitch_types_by_pitcher = {
            int(pitcher_id): self.pitch_one_service.available_pitch_types(int(pitcher_id))
            for pitcher_id in self.pitcher_rows["pitcher_id"].dropna().unique().tolist()
        }

    def pitcher_list(self):
        frame = self.pitcher_rows.drop_duplicates("pitcher_id").copy()
        frame["available_pitch_types"] = frame["pitcher_id"].map(
            lambda pitcher_id: self.deployable_pitch_types_by_pitcher.get(int(pitcher_id), [])
        )
        frame = frame.loc[frame["available_pitch_types"].map(bool)].sort_values("pitcher_name").copy()
        return frame.to_dict(orient="records")

    def batter_list(self):
        frame = self.batter_rows.drop_duplicates("batter_id").sort_values("batter_name").copy()
        return frame.to_dict(orient="records")

    def batter_info(self, batter_id: int):
        if int(batter_id) not in self.batter_lookup.index:
            raise ValueError(f"Batter {batter_id} not found in v2 source data.")
        return self.batter_lookup.loc[int(batter_id)].to_dict()

    def matchup_batter_info(
        self,
        batter_id: int,
        *,
        pitcher_id: int | None = None,
        pitcher_handedness: str | None = None):

        batter = self.batter_info(batter_id).copy()
        if pitcher_handedness is None and pitcher_id is not None:
            pitcher_handedness = str(self.pitcher_info(pitcher_id)["pitcher_handedness"]).upper()
        resolved_handedness = _resolve_matchup_batter_hand(
            batter.get("batter_handedness"),
            pitcher_handedness,
        )
        batter["resolved_batter_handedness"] = resolved_handedness
        if resolved_handedness in {"L", "R"}:
            batter["batter_handedness"] = resolved_handedness
        return batter

    def pitcher_info(self, pitcher_id: int):

        if int(pitcher_id) not in self.pitcher_lookup.index:
            raise ValueError(f"Pitcher {pitcher_id} not found in v2 source data.")
        return self.pitcher_lookup.loc[int(pitcher_id)].to_dict()

    def available_pitch_types(self, pitcher_id: int):
        return self.deployable_pitch_types_by_pitcher.get(int(pitcher_id), [])

    def get_pitch_profile(self, pitcher_id: int, pitch_type: str):
        key = (int(pitcher_id), str(pitch_type).upper())
        if key not in self.pitch_profile_cache:
            self.pitch_profile_cache[key] = _pitch_profile_from_averages(self.pitcher_profiles, pitcher_id, pitch_type)
        return self.pitch_profile_cache[key]

    def _primary_fastball_profile(self, pitcher_id: int):
        pitcher_id = int(pitcher_id)
        if pitcher_id not in self.primary_fastball_cache:
            fastball_profiles = [
                self.get_pitch_profile(pitcher_id, pitch_type)
                for pitch_type in fastball_types
            ]
            fastball_profiles = [profile for profile in fastball_profiles if profile is not None]
            self.primary_fastball_cache[pitcher_id] = (
                max(fastball_profiles, key=lambda profile: profile.pitch_count)
                if fastball_profiles
                else None
            )
        return self.primary_fastball_cache[pitcher_id]

    def _fastball_relative_deltas(self, pitcher_id: int, pitch_profile: PitchProfile):
        primary_fastball = self._primary_fastball_profile(pitcher_id)
        if primary_fastball is None:
            return 0.0, 0.0, 0.0
        return (
            float(pitch_profile.velo - primary_fastball.velo),
            float(pitch_profile.h_mov - primary_fastball.h_mov),
            float(pitch_profile.v_mov - primary_fastball.v_mov),
        )

    def _shape_bucket_values_for_profile(
        self,
        *,
        prefix: str,
        pitcher_id: int,
        pitch_type: str,
        pitch_profile: PitchProfile):

        reference_lookup = (
            self.pitch_1_shape_reference_lookup
            if prefix == "pitch_1"
            else self.pitch_2_shape_reference_lookup
        )
        deltas, reference_source = pitch_shape_deltas_from_reference_lookup(
            reference_lookup,
            pitcher_id=pitcher_id,
            pitch_type=pitch_type,
            velo=pitch_profile.velo,
            spin_rate=pitch_profile.spin_rate,
            h_mov=pitch_profile.h_mov,
            v_mov=pitch_profile.v_mov,
        )
        delta_velo, delta_spin_rate, delta_h_mov, delta_v_mov = deltas
        tight_bucket = str(
            derive_relative_pitch_shape_bucket(
                pd.Series([delta_velo]),
                pd.Series([delta_spin_rate]),
                pd.Series([delta_h_mov]),
                pd.Series([delta_v_mov]),
                level="tight",
            ).iloc[0]
        )
        medium_bucket = str(
            derive_relative_pitch_shape_bucket(
                pd.Series([delta_velo]),
                pd.Series([delta_spin_rate]),
                pd.Series([delta_h_mov]),
                pd.Series([delta_v_mov]),
                level="medium",
            ).iloc[0]
        )
        return {
            "delta_velo_vs_type_mean": float(delta_velo),
            "delta_spin_rate_vs_type_mean": float(delta_spin_rate),
            "delta_h_mov_vs_type_mean": float(delta_h_mov),
            "delta_v_mov_vs_type_mean": float(delta_v_mov),
            "shape_bucket_tight": tight_bucket,
            "shape_bucket_medium": medium_bucket,
            "reference_source": reference_source,
        }

    def pitcher_trajectories(self, pitcher_id: int):
        pitcher = self.pitcher_info(pitcher_id)
        trajectories: list[dict[str, Any]] = []
        for pitch_type in self.available_pitch_types(pitcher_id):
            profile = self.get_pitch_profile(pitcher_id, pitch_type)
            if profile is None:
                continue
            trajectories.append(
                {
                    "pitch_type": pitch_type,
                    "pitch_name": pitch_type_names.get(pitch_type, pitch_type),
                    "trajectory": _trajectory_from_profile(profile),
                    "plate_x": round(float(profile.plate_x), 4),
                    "plate_z": round(float(profile.plate_z), 4),
                    "velo": round(float(profile.velo), 1),
                    "spin_rate": round(float(profile.spin_rate), 0),
                    "h_mov": round(float(profile.h_mov), 1),
                    "v_mov": round(float(profile.v_mov), 1),
                    "extension": round(float(profile.extension), 2),
                }
            )
        return {
            "pitcher_id": int(pitcher_id),
            "pitcher_name": pitcher["pitcher_name"],
            "trajectories": trajectories,
        }

    def _find_template_row(
        self,
        *,
        pitcher_id: int,
        batter_id: int,
        batter_handedness: str,
        pitch_1_type: str,
        pitch_2_type: str,
        count_bucket: str):

        pitch_1_type = str(pitch_1_type).upper()
        pitch_2_type = str(pitch_2_type).upper()
        batter_handedness = str(batter_handedness).upper()
        count_bucket = str(count_bucket)
        exact_key = (pitcher_id, batter_id, pitch_1_type, pitch_2_type, count_bucket)
        if exact_key not in self.template_exact_lookup:
            subset = self.template_rows.loc[
                self.template_rows["pitcher_id"].eq(pitcher_id)
                & self.template_rows["batter_id"].eq(batter_id)
                & self.template_rows["pitch_1_type"].eq(pitch_1_type)
                & self.template_rows["pitch_2_type"].eq(pitch_2_type)
                & self.template_rows["count_bucket"].eq(count_bucket)
            ]
            self.template_exact_lookup[exact_key] = subset.index[-1] if len(subset) else None

        hand_key = (pitcher_id, pitch_1_type, pitch_2_type, batter_handedness, count_bucket)
        if hand_key not in self.template_hand_lookup:
            subset = self.template_rows.loc[
                self.template_rows["pitcher_id"].eq(pitcher_id)
                & self.template_rows["pitch_1_type"].eq(pitch_1_type)
                & self.template_rows["pitch_2_type"].eq(pitch_2_type)
                & self.template_rows["batter_handedness"].eq(batter_handedness)
                & self.template_rows["count_bucket"].eq(count_bucket)
            ]
            self.template_hand_lookup[hand_key] = subset.index[-1] if len(subset) else None

        pitchtype_count_key = (pitcher_id, pitch_2_type, count_bucket)
        if pitchtype_count_key not in self.template_pitchtype_count_lookup:
            subset = self.template_rows.loc[
                self.template_rows["pitcher_id"].eq(pitcher_id)
                & self.template_rows["pitch_2_type"].eq(pitch_2_type)
                & self.template_rows["count_bucket"].eq(count_bucket)
            ]
            self.template_pitchtype_count_lookup[pitchtype_count_key] = subset.index[-1] if len(subset) else None

        pitchtype_key = (pitcher_id, pitch_2_type)
        if pitchtype_key not in self.template_pitchtype_lookup:
            subset = self.template_rows.loc[
                self.template_rows["pitcher_id"].eq(pitcher_id)
                & self.template_rows["pitch_2_type"].eq(pitch_2_type)
            ]
            self.template_pitchtype_lookup[pitchtype_key] = subset.index[-1] if len(subset) else None

        candidate_specs = [
            ("exact_pitcher_batter_sequence_count", self.template_exact_lookup.get(exact_key)),
            ("pitcher_hand_sequence_count", self.template_hand_lookup.get(hand_key)),
            ("pitcher_pitchtype_count", self.template_pitchtype_count_lookup.get(pitchtype_count_key)),
            ("pitcher_pitchtype", self.template_pitchtype_lookup.get(pitchtype_key)),
        ]
        for level, index in candidate_specs:
            if index is not None:
                return self.template_rows.loc[index].copy(), level
        raise ValueError(
            f"No historical v2 template row found for pitcher {pitcher_id} and pitch type {pitch_2_type}."
        )

    def _candidate_frame_for_runtime(self, row: pd.Series):

        candidate_key = (
            row["pitcher_id"],
            str(row["batter_id"]),
            str(row["pitch_1_type"]).upper(),
            _pitch_1_bucket_for_row(row),
            str(row["pitch_2_type"]).upper(),
            str(row["batter_handedness"]).upper(),
            str(row["count_bucket"]),
        )
        if candidate_key not in self.target_candidate_lookup:
            target_rows, _cache_stats = _target_rows_from_top3_caches(
                row,
                batter_top3_cache=self.batter_top3_cache,
                pitcher_top3_cache=self.pitcher_top3_cache,
                targets=self.targets,
                lookup_indexes=self.target_lookup_indexes,
            )
            if target_rows.empty:
                target_rows = _default_target_rows().assign(
                    candidate_pool=broad_target_scope,
                )
            generated_target_rows = _generated_candidate_rows_from_target_rows(target_rows)
            if generated_target_rows.empty:
                generated_target_rows = _generated_candidate_rows_from_target_rows(
                    _default_target_rows().assign(candidate_pool=broad_target_scope)
                )
            self.target_candidate_lookup[candidate_key] = generated_target_rows.reset_index(drop=True)
        generated_target_rows = self.target_candidate_lookup[candidate_key]
        candidates = _generated_candidate_frame_from_row(row, generated_target_rows)
        return candidates.reset_index(drop=True)

    def _bucket_map_target_rows_for_runtime(self, row: pd.Series):

        selected: list[dict[str, object]] = []

        def add_cached_rows(bucket_ids: list[str], *, source: str, candidate_pool: str, target_context_scope: str):
            for rank, bucket_id in enumerate(bucket_ids[:3], start=1):
                normalized = str(bucket_id)
                if not normalized:
                    continue
                selected.append(
                    {
                        "pitch_2_bucket": normalized,
                        "candidate_pool": candidate_pool,
                        "target_context_scope": target_context_scope,
                        "rank": rank,
                        "bucket_map_source": source,
                    }
                )

        def add_fallback_rows(rows: object, *, source: str):
            records = rows.to_dict(orient="records") if isinstance(rows, pd.DataFrame) else list(rows)
            for record in records[:3]:
                if not isinstance(record, dict):
                    continue
                record = record.copy()
                record["bucket_map_source"] = source
                selected.append(record)

        batter_hit = False
        batter_buckets: list[str] = []
        for batter_cache_path in _runtime_batter_cache_paths(row):
            batter_hit, batter_buckets = nested_bucket_lookup(self.batter_top3_cache, batter_cache_path)
            if batter_hit:
                break

        if batter_hit:
            add_cached_rows(
                batter_buckets,
                source="hitter",
                candidate_pool=batter_specific_target_scope,
                target_context_scope=batter_specific_target_scope,
            )

        else:
            add_fallback_rows(
                _select_pool_target_rows(
                    row,
                    self.targets,
                    candidate_pool=batter_specific_target_scope,
                    scope_order=(batter_specific_target_scope,),
                    seen_buckets=set(),
                    lookup_indexes=self.target_lookup_indexes,
                ),
                source="hitter",
            )

        pitcher_hit = False
        pitcher_buckets: list[str] = []
        for pitcher_cache_path in _runtime_pitcher_cache_paths(row):
            pitcher_hit, pitcher_buckets = nested_bucket_lookup(self.pitcher_top3_cache, pitcher_cache_path)
            if pitcher_hit:
                break

        if pitcher_hit:
            add_cached_rows(
                pitcher_buckets,
                source="pitcher",
                candidate_pool=handidness_target_scope,
                target_context_scope=handidness_target_scope,
            )

        else:
            add_fallback_rows(
                _select_pool_target_rows(
                    row,
                    self.targets,
                    candidate_pool=handidness_target_scope,
                    scope_order=(handidness_target_scope, broad_target_scope, default_target_scope),
                    seen_buckets=set(),
                    lookup_indexes=self.target_lookup_indexes,
                ),
                source="pitcher",
            )

        return pd.DataFrame.from_records(selected) if selected else pd.DataFrame()

    def _bucket_map_for_runtime(self, row: pd.Series):
        target_rows = self._bucket_map_target_rows_for_runtime(row)
        generated_target_rows = target_rows.copy().reset_index(drop=True)
        if generated_target_rows.empty:
            return {"hitter": {}, "pitcher": {}}
        if "rank" not in generated_target_rows.columns:
            generated_target_rows["rank"] = np.arange(1, len(generated_target_rows) + 1)
        if "candidate_pool" not in generated_target_rows.columns:
            generated_target_rows["candidate_pool"] = broad_target_scope

        candidates = _generated_candidate_frame_from_row(row, generated_target_rows)
        breakdown = _planner_event_breakdown(candidates, self.event_bundle)
        scores = breakdown["expected_pitcher_value"]
        bucket_map: dict[str, dict[str, float]] = {"hitter": {}, "pitcher": {}}
        for idx, generated_row in generated_target_rows.reset_index(drop=True).iterrows():
            source = str(generated_row.get("bucket_map_source", ""))
            if source not in bucket_map:
                continue
            bucket_id = str(generated_row.get("pitch_2_bucket", ""))
            if not bucket_id:
                continue
            bucket_map[source][bucket_id] = round(float(scores[idx]), 6)
        return bucket_map

    def _bucket_map_from_scored_candidates(self, candidates: pd.DataFrame, scores: np.ndarray):
        bucket_map: dict[str, dict[str, float]] = {"hitter": {}, "pitcher": {}}
        source_by_pool = {
            batter_specific_target_scope: "hitter",
            handidness_target_scope: "pitcher",
        }
        for idx, (_, candidate) in enumerate(candidates.reset_index(drop=True).iterrows()):
            candidate_pool = str(candidate.get("candidate_pool", ""))
            source = source_by_pool.get(candidate_pool)
            if source is None:
                continue
            bucket_id = str(candidate.get("pitch_2_bucket", ""))
            if not bucket_id:
                continue
            bucket_map[source][bucket_id] = round(float(scores[idx]), 6)
        return bucket_map

    def _apply_lookup_overrides(self, row: pd.Series, pitcher_id: int, batter_id: int, pitch_2_type: str):
        return row.copy()

    def _build_synthetic_row(self, session: AtBatSession, pitch_2_type: str):
        count_bucket = f"{session.balls}-{session.strikes}"
        template, match_level = self._find_template_row(
            pitcher_id=session.pitcher_id,
            batter_id=session.batter_id,
            batter_handedness=session.batter_handedness,
            pitch_1_type=session.pitch_history[0].pitch_type,
            pitch_2_type=pitch_2_type,
            count_bucket=count_bucket,
        )
        pitch1_record = session.pitch_history[0]
        pitch1_profile = self.get_pitch_profile(session.pitcher_id, pitch1_record.pitch_type)
        pitch2_profile = self.get_pitch_profile(session.pitcher_id, pitch_2_type)
        if pitch1_profile is None or pitch2_profile is None:
            raise ValueError(f"Pitch profile unavailable for synthetic row build: {pitch1_record.pitch_type} -> {pitch_2_type}")

        batter_height_ft = self.pitch_one_service.contour_store.predict_called_strike_probability(
            batter_id=session.batter_id,
            batter_hand=session.batter_handedness,
            plate_x=pitch1_record.target_x,
            plate_z=pitch1_record.target_z,
        )["batter_height_ft"]

        updated = template.copy()
        updated = self._apply_lookup_overrides(updated, session.pitcher_id, session.batter_id, pitch_2_type)

        pitch1_x = pitch1_record.target_x
        pitch1_z = pitch1_record.target_z
        p1_fastball, p1_breaking, p1_offspeed = _pitch_family_flags(pitch1_record.pitch_type)
        p2_fastball, p2_breaking, p2_offspeed = _pitch_family_flags(pitch_2_type)
        pitch1_delta_velo, pitch1_delta_h_mov, pitch1_delta_v_mov = self._fastball_relative_deltas(
            session.pitcher_id,
            pitch1_profile,
        )
        pitch2_delta_velo, pitch2_delta_h_mov, pitch2_delta_v_mov = self._fastball_relative_deltas(
            session.pitcher_id,
            pitch2_profile,
        )
        pitch1_shape_values = self._shape_bucket_values_for_profile(
            prefix="pitch_1",
            pitcher_id=session.pitcher_id,
            pitch_type=pitch1_record.pitch_type,
            pitch_profile=pitch1_profile,
        )
        pitch2_shape_values = self._shape_bucket_values_for_profile(
            prefix="pitch_2",
            pitcher_id=session.pitcher_id,
            pitch_type=pitch_2_type,
            pitch_profile=pitch2_profile,
        )

        updated["state_id"] = synthetic_state_id
        updated["season"] = 2025
        updated["game_date"] = self.max_source_date + pd.Timedelta(days=1)
        updated["pitcher_id"] = session.pitcher_id
        updated["pitcher_name"] = session.pitcher_name
        updated["pitcher_team"] = session.pitcher_team
        updated["pitcher_handedness"] = session.pitcher_handedness
        updated["batter_id"] = session.batter_id
        updated["batter_name"] = session.batter_name
        updated["batter_team"] = session.batter_team
        updated["batter_handedness"] = session.batter_handedness
        updated["batter_height_ft"] = batter_height_ft
        updated["balls_before_p2"] = session.balls
        updated["strikes_before_p2"] = session.strikes
        updated["count_bucket"] = count_bucket
        updated["pitch_1_type"] = pitch1_record.pitch_type
        updated["pitch_1_family"] = _pitch_family_label(pitch1_record.pitch_type)
        updated["pitch_1_velo"] = pitch1_profile.velo
        updated["pitch_1_h_mov"] = pitch1_profile.h_mov
        updated["pitch_1_v_mov"] = pitch1_profile.v_mov
        updated["pitch_1_spin_rate"] = pitch1_profile.spin_rate
        updated["pitch_1_extension"] = pitch1_profile.extension
        updated["pitch_1_spin_axis"] = pitch1_profile.spin_axis
        updated["pitch_1_release_x"] = pitch1_profile.release_x
        updated["pitch_1_release_y"] = pitch1_profile.release_y
        updated["pitch_1_release_z"] = pitch1_profile.release_z
        updated["pitch_1_plate_x"] = pitch1_x
        updated["pitch_1_plate_z"] = pitch1_z
        updated["pitch_1_strike"] = int(session.strikes == 1 and session.balls == 0)
        updated["pitch_1_perceived_velo"] = _perceived_velocity(pitch1_profile.velo, pitch1_profile.extension)
        updated["pitch_1_zone_distance"] = _zone_distance(pitch1_x, pitch1_z)
        updated["pitch_1_delta_velo_vs_primary_fastball"] = pitch1_delta_velo
        updated["pitch_1_delta_h_mov_vs_primary_fastball"] = pitch1_delta_h_mov
        updated["pitch_1_delta_v_mov_vs_primary_fastball"] = pitch1_delta_v_mov
        updated["pitch_2_type"] = pitch_2_type
        updated["pitch_2_family"] = _pitch_family_label(pitch_2_type)
        updated["same_pitch_type"] = int(str(pitch1_record.pitch_type).upper() == str(pitch_2_type).upper())
        updated["same_pitch_family"] = int(
            (p1_fastball and p2_fastball)
            or (p1_breaking and p2_breaking)
            or (p1_offspeed and p2_offspeed)
        )
        updated["pitch_1_is_fastball"] = p1_fastball
        updated["pitch_2_is_fastball"] = p2_fastball
        updated["pitch_1_is_breaking"] = p1_breaking
        updated["pitch_2_is_breaking"] = p2_breaking
        updated["pitch_1_is_offspeed"] = p1_offspeed
        updated["pitch_2_is_offspeed"] = p2_offspeed
        updated["avg_velo"] = pitch2_profile.velo
        updated["avg_h_mov"] = pitch2_profile.h_mov
        updated["avg_v_mov"] = pitch2_profile.v_mov
        updated["avg_spin_rate"] = pitch2_profile.spin_rate
        updated["avg_extension"] = pitch2_profile.extension
        updated["avg_release_x"] = pitch2_profile.release_x
        updated["avg_release_z"] = pitch2_profile.release_z
        updated["avg_plate_x"] = pitch2_profile.plate_x
        updated["avg_plate_z"] = pitch2_profile.plate_z
        updated["delta_velo_vs_primary_fastball"] = pitch2_delta_velo
        updated["delta_h_mov_vs_primary_fastball"] = pitch2_delta_h_mov
        updated["delta_v_mov_vs_primary_fastball"] = pitch2_delta_v_mov
        updated["n_pitches"] = pitch2_profile.pitch_count
        updated["pitch_1_bucket"] = str(locate_bucket_id(pitch1_x, pitch1_z, batter_height_ft=batter_height_ft))
        updated["pitch_2_bucket"] = str(
            locate_bucket_id(
                pitch2_profile.plate_x,
                pitch2_profile.plate_z,
                batter_height_ft=batter_height_ft,
            )
        )
        pitch2_center_x, pitch2_center_z = bucket_center(
            updated["pitch_2_bucket"],
            batter_height_ft=batter_height_ft,
        )
        updated["intended_target_x"] = pitch2_center_x
        updated["intended_target_z"] = pitch2_center_z
        updated["pitch_1_shape_bucket_tight"] = pitch1_shape_values["shape_bucket_tight"]
        updated["pitch_1_shape_bucket_medium"] = pitch1_shape_values["shape_bucket_medium"]
        updated["pitch_2_shape_bucket_tight"] = pitch2_shape_values["shape_bucket_tight"]
        updated["pitch_2_shape_bucket_medium"] = pitch2_shape_values["shape_bucket_medium"]
        return updated, match_level

    #score pitch-2 options
    def recommend_next(
        self,
        session: AtBatSession,
        pitch_types: list[str] | None = None):

        if session.terminal:
            raise ValueError("The at-bat is terminal; there is no next pitch to recommend.")
        if session.next_pitch_number != 2:
            raise ValueError(
                "The current v2 planner only supports the recommendation immediately before pitch 2."
            )
        if not session.pitch_history:
            raise ValueError("Pitch 1 must be selected before calling the pitch-2 planner.")

        candidate_pitch_types = pitch_types or self.available_pitch_types(session.pitcher_id)
        prepared_candidates: list[tuple[str, str, pd.DataFrame]] = []
        batched_frames: list[pd.DataFrame] = []
        for pitch_type in candidate_pitch_types:
            synthetic_row, match_level = self._build_synthetic_row(session, pitch_type)
            candidates = self._candidate_frame_for_runtime(synthetic_row)
            candidates = candidates.copy()
            candidates["recommendation_group"] = str(pitch_type).upper()
            prepared_candidates.append((str(pitch_type).upper(), match_level, candidates))
            batched_frames.append(candidates)

        if not batched_frames:
            return {
                "pitch_number": session.next_pitch_number,
                "count": f"{session.balls}-{session.strikes}",
                "recommendations": [],
                "assumptions": [
                    "Pitch 1 is treated as a take, so the next count is estimated as 0-1 or 1-0.",
                    "Pitch 2 targets come from common hitter and pitcher bucket locations.",
                    "Pitch 1 shape numbers come from the pitcher's 2025 averages.",
                    "This only plans pitch 2 right now.",
                ],
            }

        batched_candidate_frame = pd.concat(batched_frames, ignore_index=True)
        breakdown = _planner_event_breakdown(batched_candidate_frame, self.event_bundle)
        batched_scores = breakdown["expected_pitcher_value"]

        recommendations: list[dict[str, Any]] = []
        start_idx = 0
        for pitch_type, match_level, candidates in prepared_candidates:
            end_idx = start_idx + len(candidates)
            scores = batched_scores[start_idx:end_idx]
            start_idx = end_idx
            ranking = np.argsort(-scores)
            best_idx = int(ranking[0])
            best_global_idx = start_idx - len(candidates) + best_idx
            best_row = candidates.iloc[best_idx]
            bucket_map = self._bucket_map_from_scored_candidates(candidates, scores)
            outcome_given_in_play = breakdown["outcome_given_in_play_probabilities"][best_global_idx]
            outcome_overall = breakdown["outcome_probabilities"][best_global_idx]
            batted_ball_probs = breakdown["batted_ball_type_probabilities"]
            exit_velocity_probs = breakdown["exit_velocity_band_probabilities"]
            recommendations.append(
                {
                    "pitch_type": str(pitch_type).upper(),
                    "pitch_type_name": pitch_type_names.get(str(pitch_type).upper(), str(pitch_type).upper()),
                    "recommended_target_x": round(float(best_row["intended_target_x"]), 4),
                    "recommended_target_z": round(float(best_row["intended_target_z"]), 4),
                    "recommended_bucket": str(best_row.get("pitch_2_bucket", "")),
                    "expected_pitcher_value": round(float(scores[best_idx]), 6),
                    "template_match_level": match_level,
                    "candidate_pool": str(best_row.get("candidate_pool", "")),
                    "bucket_map": bucket_map,
                    "pitch_outlook": {
                        "swing_probability": round(float(breakdown["swing_probability"][best_global_idx]), 6),
                        "take_probability": round(float(breakdown["take_probability"][best_global_idx]), 6),
                        "called_strike_given_take_probability": round(
                            float(breakdown["called_strike_given_take_probability"][best_global_idx]), 6
                        ),
                        "ball_given_take_probability": round(
                            float(breakdown["ball_given_take_probability"][best_global_idx]), 6
                        ),
                        "contact_given_swing_probability": round(
                            float(breakdown["contact_given_swing_probability"][best_global_idx]), 6
                        ),
                        "whiff_given_swing_probability": round(
                            float(breakdown["whiff_given_swing_probability"][best_global_idx]), 6
                        ),
                        "in_play_given_contact_probability": round(
                            float(breakdown["in_play_given_contact_probability"][best_global_idx]), 6
                        ),
                        "foul_given_contact_probability": round(
                            float(breakdown["foul_given_contact_probability"][best_global_idx]), 6
                        ),
                        "called_strike_probability": round(
                            float(breakdown["called_strike_probability"][best_global_idx]), 6
                        ),
                        "ball_probability": round(float(breakdown["ball_probability"][best_global_idx]), 6),
                        "swinging_strike_probability": round(
                            float(breakdown["swinging_strike_probability"][best_global_idx]), 6
                        ),
                        "contact_probability": round(float(breakdown["contact_probability"][best_global_idx]), 6),
                        "foul_ball_probability": round(
                            float(breakdown["foul_ball_probability"][best_global_idx]), 6
                        ),
                        "ball_in_play_probability": round(
                            float(breakdown["ball_in_play_probability"][best_global_idx]), 6
                        ),
                        "expected_contact_pitcher_value": round(
                            float(breakdown["expected_contact_pitcher_value"][best_global_idx]), 6
                        ),
                        "outcome_given_in_play_probabilities": {
                            label: round(float(outcome_given_in_play[idx]), 6)
                            for idx, label in enumerate(outcome_labels)
                        },
                        "outcome_probabilities": {
                            label: round(float(outcome_overall[idx]), 6)
                            for idx, label in enumerate(outcome_labels)
                        },
                        "batted_ball_type_probabilities": (
                            {
                                label: round(float(batted_ball_probs[best_global_idx][idx]), 6)
                                for idx, label in enumerate(batted_ball_types)
                            }
                            if batted_ball_probs is not None
                            else None
                        ),
                        "exit_velocity_band_probabilities": (
                            {
                                label: round(float(exit_velocity_probs[best_global_idx][idx]), 6)
                                for idx, label in enumerate(ev_band_labels)
                            }
                            if exit_velocity_probs is not None
                            else None
                        ),
                    },
                }
            )
        recommendations = sorted(recommendations, key=lambda row: row["expected_pitcher_value"], reverse=True)
        return {
            "pitch_number": session.next_pitch_number,
            "count": f"{session.balls}-{session.strikes}",
            "recommendations": recommendations,
            "assumptions": [
                "Pitch 1 is treated as a take, so the next count is estimated as 0-1 or 1-0.",
                "Pitch 2 target spots come from a fixed bucket grid.",
                "Candidate buckets come from hitter history and pitcher-vs-hand history.",
                "Pitch 1 shape numbers come from the pitcher's 2025 averages.",
                "This only plans pitch 2 right now.",
            ],
        }


class AtBatManager:
    def __init__(self):
        self.sessions: dict[str, AtBatSession] = {}

    def create_session(
        self,
        pitcher_info: dict[str, Any],
        batter_info: dict[str, Any]):

        session = AtBatSession(
            at_bat_id=f"ab_{uuid4().hex[:12]}",
            pitcher_id=int(pitcher_info["pitcher_id"]),
            pitcher_name=str(pitcher_info["pitcher_name"]),
            pitcher_team=None if pd.isna(pitcher_info.get("pitcher_team")) else str(pitcher_info.get("pitcher_team")),
            pitcher_handedness=str(pitcher_info["pitcher_handedness"]).upper(),
            batter_id=int(batter_info["batter_id"]),
            batter_name=str(batter_info["batter_name"]),
            batter_team=None if pd.isna(batter_info.get("batter_team")) else str(batter_info.get("batter_team")),
            batter_handedness=str(batter_info["batter_handedness"]).upper(),
        )
        self.sessions[session.at_bat_id] = session
        return session

    def get(self, at_bat_id: str):

        session = self.sessions.get(str(at_bat_id))
        if session is None:
            raise ValueError(f"At-bat session {at_bat_id} not found.")
        return session

    def close(self, at_bat_id: str):
        self.sessions.pop(str(at_bat_id), None)

    def score_pitch_one(
        self,
        at_bat_id: str,
        *,
        pitch_type: str,
        target_x: float,
        target_z: float,
        predicted_count_bucket: str):

        session = self.get(at_bat_id)
        if session.terminal:
            raise ValueError("The at-bat is already terminal.")
        if session.next_pitch_number != 1:
            raise ValueError("Pitch 1 has already been selected for this at-bat.")
        bucket = str(predicted_count_bucket)
        if bucket not in {"0-1", "1-0"}:
            raise ValueError(f"Unsupported predicted count bucket: {predicted_count_bucket}")
        session.pitch_history = [
            PitchRecord(
                pitch_number=1,
                pitch_type=str(pitch_type).upper(),
                target_x=float(target_x),
                target_z=float(target_z),
            )
        ]
        balls, strikes = [int(part) for part in bucket.split("-", maxsplit=1)]
        session.balls = balls
        session.strikes = strikes
        session.next_pitch_number = 2
        return session

def load_v2_backend_runtime():

    if not event_tree_model_path.exists():
        raise FileNotFoundError(
            f"Missing v2 event bundle at {event_tree_model_path}. Run models/v2/train.py first."
        )
    if not p1_event_tree_model_path.exists():
        raise FileNotFoundError(
            f"Missing pitch1 event bundle at {p1_event_tree_model_path}. Run models/p1/train.py first."
        )
    if not _runtime_table_exists(pitch2_plavver_eval_path) or not _runtime_table_exists(pitch_target_distributions_path):
        raise FileNotFoundError(
            "Missing v2 tables. Run models/v2/data.py first so the backend can load planner tables."
        )
    if not _runtime_table_exists(p1_planner_eval_view_path):
        raise FileNotFoundError(
            "Missing pitch1 tables. Run models/p1/train.py first so the backend can load pitch1 planner tables."
        )
    
    if (
        _load_strike_zone_runtime_store_payload() is None
        and not _runtime_csv_variant(prepared_pitch_level_2025).exists()
        and not prepared_pitch_level_2025.exists()
        and not pitch_level_2025.exists()):

        raise FileNotFoundError(
            "Missing strike-zone runtime source. Expected one of "
            f"{strike_zone_runtime_store_path}, {_runtime_csv_variant(prepared_pitch_level_2025)}, "
            f"{prepared_pitch_level_2025}, or {pitch_level_2025}."
        )
    
    if not pitch_averages_2025.exists():
        raise FileNotFoundError(f"Missing pitch-type averages at {pitch_averages_2025}.")
    with event_tree_model_path.open("rb") as handle:
        event_bundle = pickle.load(handle)
    with p1_event_tree_model_path.open("rb") as handle:
        pitch1_event_bundle = pickle.load(handle)
    planner_metadata = _load_planner_metadata()
    planner_view = _read_runtime_table(pitch2_plavver_eval_path)
    targets = _read_runtime_table(pitch_target_distributions_path)
    pitch1_planner_view = _read_runtime_table(p1_planner_eval_view_path)
    strike_zone_runtime_store = _load_strike_zone_runtime_store_payload()
    pitch_averages_2025_frame = pd.read_csv(pitch_averages_2025)

    if strike_zone_runtime_store is not None:
        contour_store = StrikeZoneSurfaceStore.from_serialized_payload(strike_zone_runtime_store)
    else:
        pitch_level_2025_frame = _load_strike_zone_source_frame()
        contour_store = (
            StrikeZoneSurfaceStore.from_prepared_frame(pitch_level_2025_frame)
            if {"batter_id", "batter_hand", "plate_x", "plate_z", "height_ft", "is_called_strike"}.issubset(set(pitch_level_2025_frame.columns))
            else StrikeZoneSurfaceStore(pitch_level_2025_frame)
        )

    saved_called_strike_model_class = _component_class_name(event_bundle.get("called_strike_model"))
    saved_pitch1_called_strike_model_class = _component_class_name(pitch1_event_bundle.get("called_strike_model"))
    saved_count_lookup_keys = _lookup_keys(event_bundle.get("count_state_lookup"))
    saved_event_premium_lookup_keys = _event_premium_lookup_keys(event_bundle.get("event_premium_lookup"))
    event_bundle["called_strike_model"] = contour_store
    pitch1_event_bundle["called_strike_model"] = contour_store
    event_bundle["count_state_lookup"] = _re288_relative_pitcher_count_state_lookup()
    event_premium_source = "saved_research_event_bundle"

    if "event_premium_lookup" not in event_bundle:
        pitch_level_history = load_available_pitch_level_history()
        event_bundle["event_premium_lookup"] = build_pitch2_event_premium_lookup(
            pitch_level_history,
            event_bundle["count_state_lookup"],
        )

        event_premium_source = "runtime_fallback_from_pitch_level_history"

    pitch_one_service = PitchOneResearchService(
        planner_view=pitch1_planner_view,
        pitch_averages_2025=pitch_averages_2025_frame,
        event_bundle=pitch1_event_bundle,
        contour_store=contour_store,
    )
    planner_runtime = V2PlannerRuntime(
        planner_view=planner_view,
        targets=targets,
        event_bundle=event_bundle,
        pitch_averages_2025=pitch_averages_2025_frame,
        pitch_one_service=pitch_one_service,
    )
    session_manager = AtBatManager()
    traceability = _runtime_traceability_summary(
        planner_metadata=planner_metadata,
        saved_called_strike_model_class=saved_called_strike_model_class,
        saved_pitch1_called_strike_model_class=saved_pitch1_called_strike_model_class,
        saved_count_lookup_keys=saved_count_lookup_keys,
        saved_event_premium_lookup_keys=saved_event_premium_lookup_keys,
        runtime_event_bundle=event_bundle,
        runtime_pitch1_event_bundle=pitch1_event_bundle,
        contour_store=contour_store,
        event_premium_source=event_premium_source,
    )

    return {
        "contour_store": contour_store,
        "pitch_one_service": pitch_one_service,
        "planner_runtime": planner_runtime,
        "session_manager": session_manager,
        "event_bundle": event_bundle,
        "pitch1_event_bundle": pitch1_event_bundle,
        "planner_metadata": planner_metadata,
        "traceability": traceability,
    }


def prewarm_backend_runtime(runtime: dict[str, Any]):
    planner: V2PlannerRuntime = runtime["planner_runtime"]
    pitch_one_service: PitchOneResearchService = runtime["pitch_one_service"]
    sessions: AtBatManager = runtime["session_manager"]

    warmed_pitch_profiles = 0
    warmed_pitchers = 0
    for pitcher_id, pitch_types in planner.deployable_pitch_types_by_pitcher.items():
        planner._primary_fastball_profile(int(pitcher_id))
        pitch_one_service._primary_fastball_profile(int(pitcher_id))
        warmed_pitchers += 1
        for pitch_type in pitch_types:
            if planner.get_pitch_profile(int(pitcher_id), pitch_type) is not None:
                warmed_pitch_profiles += 1
            pitch_one_service.get_pitch_profile(int(pitcher_id), pitch_type)

    sample_batters: list[dict[str, Any]] = []
    seen_hands: set[str] = set()
    for batter in planner.batter_list():
        hand = str(batter.get("batter_handedness", "")).upper()
        if hand in {"L", "R"} and hand not in seen_hands:
            sample_batters.append(batter)
            seen_hands.add(hand)
        if len(sample_batters) >= 2:
            break
    if not sample_batters:
        sample_batters = planner.batter_list()[:1]

    dry_run_count = 0
    for pitcher in planner.pitcher_list()[:1]:
        candidate_pitch_types = planner.available_pitch_types(int(pitcher["pitcher_id"]))
        if not candidate_pitch_types:
            continue
        seed_pitch_type = candidate_pitch_types[0]
        seed_profile = planner.get_pitch_profile(int(pitcher["pitcher_id"]), seed_pitch_type)
        if seed_profile is None:
            continue
        for batter in sample_batters:
            session = sessions.create_session(pitcher, batter)
            try:
                assessment = pitch_one_service.score_pitch(
                    pitcher_id=int(session.pitcher_id),
                    batter_id=int(session.batter_id),
                    batter_hand=str(session.batter_handedness).upper(),
                    pitch_type=seed_pitch_type,
                    target_x=float(seed_profile.plate_x),
                    target_z=float(seed_profile.plate_z),
                )
                session = sessions.score_pitch_one(
                    session.at_bat_id,
                    pitch_type=seed_pitch_type,
                    target_x=float(seed_profile.plate_x),
                    target_z=float(seed_profile.plate_z),
                    predicted_count_bucket=str(assessment["predicted_pitch2_count_bucket"]),
                )
                planner.recommend_next(session, candidate_pitch_types)
                planner.recommend_next(session, candidate_pitch_types)
                dry_run_count += 1
            finally:
                sessions.close(session.at_bat_id)

    return {
        "status": "ok",
        "warmed_pitchers": warmed_pitchers,
        "warmed_pitch_profiles": warmed_pitch_profiles,
        "dry_run_predictions": dry_run_count,
        "sampled_batter_hands": sorted(seen_hands),
    }
