from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    package = Path(__file__).resolve().parents[2]
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    from models.v2.config import (
        batter_bucket,
        batter_top3_cache_path,
        data,
        pitch2_event_tree,
        pitch2_observed_action,
        pitch2_outcome,
        pitch2_plavver_eval_path,
        pitch2_states,
        pitch_target_distributions_path,
        pitcher_bucket,
        pitcher_top3_cache_path,
        pitcher_arsenal_profile,
        source_data,
        table,
        glob_pitch_level,
    )
    from models.v2.schema import (
        arsenal,
        observed_action,
        outcomes,
        state,
        target_distribution,
    )
    from models.v2.location_grid import (
        bucket_ids,
        bucket_center,
        bucket_center_frame,
        locate_bucket_series,
    )
else:
    from .config import (
        batter_bucket,
        batter_top3_cache_path,
        data,
        pitch2_event_tree,
        pitch2_observed_action,
        pitch2_outcome,
        pitch2_plavver_eval_path,
        pitch2_states,
        pitch_target_distributions_path,
        pitcher_bucket,
        pitcher_top3_cache_path,
        pitcher_arsenal_profile,
        source_data,
        table,
        glob_pitch_level,
    )
    from .schema import (
        arsenal,
        observed_action,
        outcomes,
        state,
        target_distribution,
    )
    from .location_grid import (
        bucket_ids,
        bucket_center,
        bucket_center_frame,
        locate_bucket_series,
    )


hit_value = {
    "out": 0.0,
    "single": 0.87,
    "double_or_triple": 1.245,
    "home_run": 2.05,
}
FB_types = {"FF", "FT", "SI", "FC", "FA"}
BB_types = {"SL", "CU", "KC", "SV", "CS"}
OFF_types = {"CH", "FS", "FO", "SC", "KN", "EP"}
PITCH2_CONTINUATION_STATES = (
    (0, 2),
    (1, 1),
    (2, 0),
)
called_strike_height_top_ratio = 0.535
called_strike_height_bot_ratio = 0.27
default_batter_height_ft = 6.0
min_batter_height_ft = 5.0
max_batter_height_ft = 7.5
zone_horiz_third_ft = 0.28
zone_vert_third_low = 1.0 / 3.0
zone_vert_third_high = 2.0 / 3.0
batter_specific_target_scope = "batter_specific"
handidness_target_scope = "batter_handedness"
broad_target_scope = "broad_pitchtype"
default_target_scope = "default_pitchtype"
num_buckets = 3

@dataclass(frozen=True)
#saved v2 tables
class V2Tables:
    pitch2_states: pd.DataFrame
    pitcher_arsenal_profiles: pd.DataFrame
    pitch_target_distributions: pd.DataFrame
    pitch2_observed_actions: pd.DataFrame
    pitch2_outcomes: pd.DataFrame
    pitch2_event_tree_view: pd.DataFrame
    pitch2_planner_eval_view: pd.DataFrame


#cli entry args
def build_argument_parser():
    parser = argparse.ArgumentParser(description="Build normalized v2 planner tables.")
    parser.add_argument(
        "--source",
        default=str(source_data),
        help="Source first-two-pitch dataset CSV.",
    )
    return parser


def _safe_numeric(series: pd.Series):
    return pd.to_numeric(series, errors="coerce")


def _perceived_velocity(velo: pd.Series, extension: pd.Series):
    extension = _safe_numeric(extension).clip(lower=0.0, upper=15.0)
    velo = _safe_numeric(velo)
    denom = (60.5 - extension).replace(0.0, np.nan)
    return velo * (60.5 / denom)


def _zone_distance(plate_x: pd.Series, plate_z: pd.Series):
    plate_x = _safe_numeric(plate_x)
    plate_z = _safe_numeric(plate_z)
    horizontal_gap = (plate_x.abs() - 0.83).clip(lower=0.0)
    lower_gap = (1.5 - plate_z).clip(lower=0.0)
    upper_gap = (plate_z - 3.5).clip(lower=0.0)
    vertical_gap = lower_gap.where(lower_gap > 0.0, upper_gap)
    return np.sqrt(horizontal_gap ** 2 + vertical_gap ** 2)


#online prior feature
def _group_shifted_rate(
    frame: pd.DataFrame,
    group_columns: list[str],
    target_column: str,
    prior_mean: float,
    prior_strength: float = 25.0):

    working = frame[group_columns].copy()
    working[target_column] = _safe_numeric(frame[target_column]).fillna(0.0)
    grouped = working.groupby(group_columns, sort=False, dropna=False)
    prior_sum = grouped[target_column].cumsum() - working[target_column]
    prior_count = grouped.cumcount()
    return (prior_sum + prior_strength * prior_mean) / (prior_count + prior_strength)


def _group_shifted_conditional_rate(
    frame: pd.DataFrame,
    group_columns: list[str],
    numerator_column: str,
    denominator_column: str,
    prior_mean: float,
    prior_strength: float = 25.0):

    working = frame[group_columns].copy()
    working["_num"] = _safe_numeric(frame[numerator_column]).fillna(0.0)
    working["_den"] = _safe_numeric(frame[denominator_column]).fillna(0.0)
    grouped = working.groupby(group_columns, sort=False, dropna=False)
    prior_num = grouped["_num"].cumsum() - working["_num"]
    prior_den = grouped["_den"].cumsum() - working["_den"]
    return (prior_num + prior_strength * prior_mean) / (prior_den + prior_strength)


def _group_shifted_rate_with_prior_series(
    frame: pd.DataFrame,
    group_columns: list[str],
    target_column: str,
    prior_mean_series: pd.Series,
    prior_strength: float = 25.0):

    working = frame[group_columns].copy()
    working[target_column] = _safe_numeric(frame[target_column]).fillna(0.0)
    grouped = working.groupby(group_columns, sort=False, dropna=False)
    prior_sum = grouped[target_column].cumsum() - working[target_column]
    prior_count = grouped.cumcount()
    prior_mean = _safe_numeric(prior_mean_series).reindex(frame.index).fillna(_safe_numeric(prior_mean_series).mean())
    return (prior_sum + prior_strength * prior_mean) / (prior_count + prior_strength)


def _group_shifted_conditional_rate_with_prior_series(
    frame: pd.DataFrame,
    group_columns: list[str],
    numerator_column: str,
    denominator_column: str,
    prior_mean_series: pd.Series,
    prior_strength: float = 25.0):

    working = frame[group_columns].copy()
    working["_num"] = _safe_numeric(frame[numerator_column]).fillna(0.0)
    working["_den"] = _safe_numeric(frame[denominator_column]).fillna(0.0)
    grouped = working.groupby(group_columns, sort=False, dropna=False)
    prior_num = grouped["_num"].cumsum() - working["_num"]
    prior_den = grouped["_den"].cumsum() - working["_den"]
    prior_mean = _safe_numeric(prior_mean_series).reindex(frame.index).fillna(_safe_numeric(prior_mean_series).mean())
    return (prior_num + prior_strength * prior_mean) / (prior_den + prior_strength)


def _group_shifted_rolling_rate(
    frame: pd.DataFrame,
    group_columns: list[str],
    target_column: str,
    window: int,
    prior_mean: float,
    prior_strength: float = 10.0):

    pieces: list[pd.Series] = []
    for _, group in frame.groupby(group_columns, sort=False, dropna=False):
        series = _safe_numeric(group[target_column]).fillna(0.0)
        shifted = series.shift(1)
        rolling_sum = shifted.rolling(window=window, min_periods=1).sum()
        rolling_count = shifted.notna().rolling(window=window, min_periods=1).sum()
        result = (rolling_sum + prior_strength * prior_mean) / (rolling_count + prior_strength)
        pieces.append(result)
    return pd.concat(pieces).sort_index()


def _group_shifted_rolling_conditional_rate(
    frame: pd.DataFrame,
    group_columns: list[str],
    numerator_column: str,
    denominator_column: str,
    window: int,
    prior_mean: float,
    prior_strength: float = 10.0):

    pieces: list[pd.Series] = []
    for _, group in frame.groupby(group_columns, sort=False, dropna=False):
        numerator = _safe_numeric(group[numerator_column]).fillna(0.0)
        denominator = _safe_numeric(group[denominator_column]).fillna(0.0)
        shifted_num = numerator.shift(1)
        shifted_den = denominator.shift(1)
        rolling_num = shifted_num.rolling(window=window, min_periods=1).sum()
        rolling_den = shifted_den.rolling(window=window, min_periods=1).sum()
        result = (rolling_num + prior_strength * prior_mean) / (rolling_den + prior_strength)
        pieces.append(result)
    return pd.concat(pieces).sort_index()


def _group_shifted_rolling_mean(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    window: int,
    default_value: float):

    pieces: list[pd.Series] = []
    for _, group in frame.groupby(group_columns, sort=False, dropna=False):
        series = _safe_numeric(group[value_column])
        shifted = series.shift(1)
        rolling_mean = shifted.rolling(window=window, min_periods=1).mean()
        pieces.append(rolling_mean.fillna(default_value))
    return pd.concat(pieces).sort_index()


def _group_shifted_rolling_masked_mean(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    mask_column: str,
    window: int,
    default_value: float,
    prior_strength: float = 10.0):

    pieces: list[pd.Series] = []
    for _, group in frame.groupby(group_columns, sort=False, dropna=False):
        values = _safe_numeric(group[value_column]).fillna(0.0)
        mask = _safe_numeric(group[mask_column]).fillna(0.0)
        shifted_weighted = (values * mask).shift(1)
        shifted_mask = mask.shift(1)
        rolling_weighted = shifted_weighted.rolling(window=window, min_periods=1).sum()
        rolling_mask = shifted_mask.rolling(window=window, min_periods=1).sum()
        result = (rolling_weighted + prior_strength * default_value) / (rolling_mask + prior_strength)
        pieces.append(result)
    return pd.concat(pieces).sort_index()


def _group_shifted_rolling_share(
    frame: pd.DataFrame,
    numerator_group_columns: list[str],
    denominator_group_columns: list[str],
    window: int,
    prior_mean: float,
    prior_strength: float = 10.0):

    working = frame.copy()
    working["_row_one"] = 1.0
    numerator_parts: list[pd.Series] = []
    for _, group in working.groupby(numerator_group_columns, sort=False, dropna=False):
        shifted = _safe_numeric(group["_row_one"]).shift(1)
        numerator_parts.append(shifted.rolling(window=window, min_periods=1).sum())
    numerator = pd.concat(numerator_parts).sort_index()

    denominator_parts: list[pd.Series] = []
    for _, group in working.groupby(denominator_group_columns, sort=False, dropna=False):
        shifted = _safe_numeric(group["_row_one"]).shift(1)
        denominator_parts.append(shifted.rolling(window=window, min_periods=1).sum())
    denominator = pd.concat(denominator_parts).sort_index()
    return (numerator.fillna(0.0) + prior_strength * prior_mean) / (denominator.fillna(0.0) + prior_strength)


def _group_shifted_mean(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    default_value: float):

    working = frame[group_columns].copy()
    values = _safe_numeric(frame[value_column]).fillna(0.0)
    working["_value"] = values
    grouped = working.groupby(group_columns, sort=False, dropna=False)
    prior_sum = grouped["_value"].cumsum() - values
    prior_count = grouped.cumcount()
    return (prior_sum / prior_count.replace(0, np.nan)).fillna(default_value)


def _group_shifted_mean_with_prior_series(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    prior_mean_series: pd.Series):

    working = frame[group_columns].copy()
    values = _safe_numeric(frame[value_column]).fillna(0.0)
    working["_value"] = values
    grouped = working.groupby(group_columns, sort=False, dropna=False)
    prior_sum = grouped["_value"].cumsum() - values
    prior_count = grouped.cumcount()
    prior_mean = _safe_numeric(prior_mean_series).reindex(frame.index).fillna(_safe_numeric(prior_mean_series).mean())
    return (prior_sum / prior_count.replace(0, np.nan)).fillna(prior_mean)


def _group_shifted_std(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    default_value: float):

    working = frame[group_columns].copy()
    values = _safe_numeric(frame[value_column]).fillna(0.0)
    working["_value"] = values
    working["_value_sq"] = values ** 2
    grouped = working.groupby(group_columns, sort=False, dropna=False)
    prior_sum = grouped["_value"].cumsum() - values
    prior_sum_sq = grouped["_value_sq"].cumsum() - working["_value_sq"]
    prior_count = grouped.cumcount()
    prior_mean = prior_sum / prior_count.replace(0, np.nan)
    prior_var = (prior_sum_sq / prior_count.replace(0, np.nan)) - (prior_mean ** 2)
    return np.sqrt(prior_var.clip(lower=0.0)).fillna(default_value)


def _batter_view_plate_x(plate_x: pd.Series, batter_handedness: pd.Series):
    x = _safe_numeric(plate_x)
    handedness = batter_handedness.fillna("R").astype(str).str.upper()
    multiplier = np.where(handedness.eq("R"), -1.0, 1.0)
    return x * multiplier


def _batter_zone_height_bounds(batter_height_ft: pd.Series):
    height = (
        _safe_numeric(batter_height_ft)
        .fillna(default_batter_height_ft)
        .clip(lower=min_batter_height_ft, upper=max_batter_height_ft)
    )
    zone_top = height * called_strike_height_top_ratio
    zone_bottom = height * called_strike_height_bot_ratio
    return zone_bottom, zone_top


def derive_pitch_zone_bucket(
    plate_x: pd.Series,
    plate_z: pd.Series,
    batter_handedness: pd.Series,
    batter_height_ft: pd.Series):

    batter_view_x = _batter_view_plate_x(plate_x, batter_handedness)
    horizontal_bucket = np.select(
        [
            batter_view_x >= zone_horiz_third_ft,
            batter_view_x <= -zone_horiz_third_ft,
        ],
        ["inside", "outside"],
        default="middle",
    )
    zone_bottom, zone_top = _batter_zone_height_bounds(batter_height_ft)
    zone_height = (zone_top - zone_bottom).replace(0.0, np.nan)
    z_unit = ((_safe_numeric(plate_z) - zone_bottom) / zone_height).fillna(0.5)
    vertical_bucket = np.select(
        [
            z_unit < zone_vert_third_low,
            z_unit < zone_vert_third_high,
        ],
        ["low", "middle"],
        default="high",
    )
    return pd.Series(
        np.char.add(np.char.add(vertical_bucket.astype(str), "_"), horizontal_bucket.astype(str)),
        index=plate_x.index,
    )


def derive_pitch_location_context_bucket(
    plate_x: pd.Series,
    plate_z: pd.Series,
    batter_handedness: pd.Series,
    batter_height_ft: pd.Series):

    batter_view_x = _batter_view_plate_x(plate_x, batter_handedness)
    zone_bottom, zone_top = _batter_zone_height_bounds(batter_height_ft)
    plate_z_numeric = _safe_numeric(plate_z)
    in_zone_mask = (
        batter_view_x.ge(-0.83)
        & batter_view_x.le(0.83)
        & plate_z_numeric.ge(zone_bottom)
        & plate_z_numeric.le(zone_top)
    )
    in_zone_bucket = derive_pitch_zone_bucket(plate_x, plate_z, batter_handedness, batter_height_ft)
    zone_mid = zone_bottom + ((zone_top - zone_bottom) / 2.0)
    broad_vertical = np.where(plate_z_numeric.ge(zone_mid), "upper", "lower")
    broad_horizontal = np.where(batter_view_x.ge(0.0), "inside", "outside")
    broad_out_of_zone_bucket = pd.Series(
        np.char.add(
            np.char.add(np.char.add("out_", broad_vertical.astype(str)), "_"),
            broad_horizontal.astype(str),
        ),
        index=plate_x.index,
        dtype="string",
    )
    return in_zone_bucket.where(in_zone_mask, broad_out_of_zone_bucket)


def _relative_band_label(
    series: pd.Series,
    *,
    tight_threshold: float,
    column_prefix: str):

    numeric = _safe_numeric(series)
    return pd.Series(
        np.select(
            [
                numeric.lt(-tight_threshold),
                numeric.gt(tight_threshold),
            ],
            [f"{column_prefix}_below", f"{column_prefix}_above"],
            default=f"{column_prefix}_center",
        ),
        index=series.index,
        dtype="string",
    )


def derive_relative_pitch_shape_bucket(
    delta_velo: pd.Series,
    delta_spin_rate: pd.Series,
    delta_h_mov: pd.Series,
    delta_v_mov: pd.Series,
    *,
    level: str):

    if level == "tight":
        velo_threshold = 1.0
        spin_threshold = 150.0
        movement_threshold = 2.0
    elif level == "medium":
        velo_threshold = 2.5
        spin_threshold = 300.0
        movement_threshold = 4.0
    else:
        raise ValueError(f"Unsupported pitch shape bucket level: {level}")

    velo_bucket = _relative_band_label(delta_velo, tight_threshold=velo_threshold, column_prefix="velo")
    spin_bucket = _relative_band_label(delta_spin_rate, tight_threshold=spin_threshold, column_prefix="spin")
    h_mov_bucket = _relative_band_label(delta_h_mov, tight_threshold=movement_threshold, column_prefix="hmov")
    v_mov_bucket = _relative_band_label(delta_v_mov, tight_threshold=movement_threshold, column_prefix="vmov")
    return (
        velo_bucket
        .str.cat(spin_bucket, sep="|")
        .str.cat(h_mov_bucket, sep="|")
        .str.cat(v_mov_bucket, sep="|")
    )


def build_pitch_type_shape_reference_lookup(
    frame: pd.DataFrame,
    *,
    prefix: str):

    working = frame.copy()
    pitch_type_column = f"{prefix}_type"
    stat_columns = ("velo", "spin_rate", "h_mov", "v_mov")
    source_columns = [f"{prefix}_{stat_column}" for stat_column in stat_columns]
    required_columns = {"pitcher_id", pitch_type_column, *source_columns}
    if not required_columns.issubset(working.columns):
        return {
            "prefix": prefix,
            "by_pitcher_type": {},
            "global_by_type": {stat_column: {} for stat_column in stat_columns},
            "overall": {stat_column: 0.0 for stat_column in stat_columns},
        }

    baseline = _baseline_reference_frame(working)
    overall = {
        stat_column: float(_safe_numeric(baseline[f"{prefix}_{stat_column}"]).mean())
        for stat_column in stat_columns
    }
    global_by_type: dict[str, dict[str, float]] = {stat_column: {} for stat_column in stat_columns}
    by_pitcher_type: dict[tuple[int, str], dict[str, float]] = {}

    for stat_column in stat_columns:
        source_column = f"{prefix}_{stat_column}"
        grouped_by_type = baseline.groupby(pitch_type_column, dropna=False)[source_column].mean()
        global_by_type[stat_column] = {
            str(pitch_type).upper(): float(value)
            for pitch_type, value in grouped_by_type.items()
            if pd.notna(value)
        }

    grouped_by_pitcher = (
        baseline.groupby(["pitcher_id", pitch_type_column], dropna=False)[source_columns]
        .mean()
        .reset_index()
    )
    for row in grouped_by_pitcher.itertuples(index=False):
        pitcher_id = pd.to_numeric(getattr(row, "pitcher_id"), errors="coerce")
        if pd.isna(pitcher_id):
            continue
        pitch_type = str(getattr(row, pitch_type_column)).upper()
        by_pitcher_type[(int(pitcher_id), pitch_type)] = {
            stat_column: float(getattr(row, f"{prefix}_{stat_column}"))
            for stat_column in stat_columns
        }

    return {
        "prefix": prefix,
        "by_pitcher_type": by_pitcher_type,
        "global_by_type": global_by_type,
        "overall": overall,
    }


def pitch_shape_deltas_from_reference_lookup(
    reference_lookup: dict[str, object],
    *,
    pitcher_id: int | str | float,
    pitch_type: str,
    velo: float,
    spin_rate: float,
    h_mov: float,
    v_mov: float):

    normalized_pitch_type = str(pitch_type or "").upper()
    numeric_pitcher_id = pd.to_numeric(pd.Series([pitcher_id]), errors="coerce").iloc[0]
    by_pitcher_type = reference_lookup.get("by_pitcher_type", {})
    global_by_type = reference_lookup.get("global_by_type", {})
    overall = reference_lookup.get("overall", {})

    reference_source = "global_overall"
    reference_values: dict[str, float] | None = None
    if pd.notna(numeric_pitcher_id):
        reference_values = by_pitcher_type.get((int(numeric_pitcher_id), normalized_pitch_type))
        if reference_values is not None:
            reference_source = "pitcher_type_mean"
    if reference_values is None:
        candidate_values = {
            stat_column: global_by_type.get(stat_column, {}).get(normalized_pitch_type)
            for stat_column in ("velo", "spin_rate", "h_mov", "v_mov")
        }
        if all(value is not None and np.isfinite(value) for value in candidate_values.values()):
            reference_values = {
                stat_column: float(candidate_values[stat_column])
                for stat_column in ("velo", "spin_rate", "h_mov", "v_mov")
            }
            reference_source = "global_type_mean"
    if reference_values is None:
        reference_values = {
            stat_column: float(overall.get(stat_column, 0.0))
            for stat_column in ("velo", "spin_rate", "h_mov", "v_mov")
        }

    deltas = (
        float(velo - reference_values["velo"]),
        float(spin_rate - reference_values["spin_rate"]),
        float(h_mov - reference_values["h_mov"]),
        float(v_mov - reference_values["v_mov"]),
    )
    return deltas, reference_source


def _attach_pitch_type_shape_reference(frame: pd.DataFrame, *, prefix: str):
    working = frame.copy()
    reference_lookup = build_pitch_type_shape_reference_lookup(working, prefix=prefix)
    source_columns = {stat_column: f"{prefix}_{stat_column}" for stat_column in ("velo", "spin_rate", "h_mov", "v_mov")}
    delta_columns = {
        "velo": f"{prefix}_delta_velo_vs_type_mean",
        "spin_rate": f"{prefix}_delta_spin_rate_vs_type_mean",
        "h_mov": f"{prefix}_delta_h_mov_vs_type_mean",
        "v_mov": f"{prefix}_delta_v_mov_vs_type_mean",
    }
    type_mean_columns = {
        "velo": f"{prefix}_type_mean_velo_prior",
        "spin_rate": f"{prefix}_type_mean_spin_rate_prior",
        "h_mov": f"{prefix}_type_mean_h_mov_prior",
        "v_mov": f"{prefix}_type_mean_v_mov_prior",
    }

    delta_records: list[tuple[float, float, float, float]] = []
    for row in working.itertuples(index=False):
        deltas, _ = pitch_shape_deltas_from_reference_lookup(
            reference_lookup,
            pitcher_id=getattr(row, "pitcher_id", np.nan),
            pitch_type=getattr(row, f"{prefix}_type", ""),
            velo=float(getattr(row, source_columns["velo"])),
            spin_rate=float(getattr(row, source_columns["spin_rate"])),
            h_mov=float(getattr(row, source_columns["h_mov"])),
            v_mov=float(getattr(row, source_columns["v_mov"])),
        )
        delta_records.append(deltas)

    if delta_records:
        delta_frame = pd.DataFrame(
            delta_records,
            columns=[
                delta_columns["velo"],
                delta_columns["spin_rate"],
                delta_columns["h_mov"],
                delta_columns["v_mov"],
            ],
            index=working.index,
            dtype=float,
        )
        for stat_column, delta_column in delta_columns.items():
            source_series = _safe_numeric(working[source_columns[stat_column]])
            delta_series = delta_frame[delta_column]
            working[delta_column] = delta_series
            working[type_mean_columns[stat_column]] = source_series - delta_series
    else:
        for stat_column, delta_column in delta_columns.items():
            source_series = _safe_numeric(working[source_columns[stat_column]])
            working[delta_column] = 0.0
            working[type_mean_columns[stat_column]] = source_series
    return working


def _pitch_family_flags(pitch_types: pd.Series):
    normalized = pitch_types.fillna("UN").astype(str).str.upper()
    return (
        normalized.isin(FB_types).astype(int),
        normalized.isin(BB_types).astype(int),
        normalized.isin(OFF_types).astype(int),
    )


def _pitch_family_label(pitch_types: pd.Series):
    normalized = pitch_types.fillna("UN").astype(str).str.upper()
    return pd.Series(
        np.select(
            [
                normalized.isin(FB_types),
                normalized.isin(BB_types),
                normalized.isin(OFF_types),
            ],
            ["fastball", "breaking", "offspeed"],
            default="other",
        ),
        index=pitch_types.index,
        dtype="string",
    )


def _derive_batted_ball_type(launch_angle: pd.Series):
    angle = _safe_numeric(launch_angle)
    labels = np.select(
        [
            angle.lt(10.0),
            angle.lt(25.0),
        ],
        ["groundball", "line_drive"],
        default="fly_ball",
    )
    labels = pd.Series(labels, index=launch_angle.index, dtype="string")
    labels = labels.where(angle.notna(), "unknown")
    return labels


def _derive_exit_velo_band(exit_velo: pd.Series):
    velo = _safe_numeric(exit_velo)
    labels = np.select(
        [
            velo.lt(90.0),
            velo.lt(95.0),
            velo.lt(100.0),
            velo.lt(105.0),
        ],
        ["lt_90", "90_95", "95_100", "100_105"],
        default="ge_105",
    )
    labels = pd.Series(labels, index=exit_velo.index, dtype="string")
    labels = labels.where(velo.notna(), "unknown")
    return labels


def _attach_primary_fastball_reference(
    frame: pd.DataFrame,
    *,
    pitch_type_column: str,
    velo_column: str,
    h_mov_column: str,
    v_mov_column: str,
    prefix: str):

    working = frame.copy()
    baseline = _baseline_reference_frame(working)
    baseline_fastballs = baseline.loc[
        baseline[pitch_type_column].fillna("UN").astype(str).str.upper().isin(FB_types)
    ].copy()
    global_fastball_type = (
        baseline_fastballs[pitch_type_column].fillna("FF").astype(str).str.upper().mode().iloc[0]
        if len(baseline_fastballs)
        else "FF"
    )
    global_fastball_velo = float(_safe_numeric(baseline_fastballs[velo_column]).mean()) if len(baseline_fastballs) else 0.0
    global_fastball_h_mov = float(_safe_numeric(baseline_fastballs[h_mov_column]).mean()) if len(baseline_fastballs) else 0.0
    global_fastball_v_mov = float(_safe_numeric(baseline_fastballs[v_mov_column]).mean()) if len(baseline_fastballs) else 0.0

    primary_types = pd.Series(index=working.index, dtype="string")
    primary_velo = pd.Series(index=working.index, dtype="float64")
    primary_h_mov = pd.Series(index=working.index, dtype="float64")
    primary_v_mov = pd.Series(index=working.index, dtype="float64")

    for _, group in working.groupby("pitcher_id", sort=False, dropna=False):
        counts: dict[str, int] = {}
        sums: dict[str, dict[str, float]] = {}
        for idx, row in group.iterrows():
            if counts:
                primary_type = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
                stats = sums[primary_type]
                primary_types.loc[idx] = primary_type
                primary_velo.loc[idx] = stats["velo"] / max(counts[primary_type], 1)
                primary_h_mov.loc[idx] = stats["h_mov"] / max(counts[primary_type], 1)
                primary_v_mov.loc[idx] = stats["v_mov"] / max(counts[primary_type], 1)
            else:
                current_type = str(row.get(pitch_type_column, "UN") or "UN").upper()
                primary_types.loc[idx] = current_type if current_type in FB_types else global_fastball_type
                primary_velo.loc[idx] = global_fastball_velo
                primary_h_mov.loc[idx] = global_fastball_h_mov
                primary_v_mov.loc[idx] = global_fastball_v_mov

            current_type = str(row.get(pitch_type_column, "UN") or "UN").upper()
            if current_type not in FB_types:
                continue
            counts[current_type] = counts.get(current_type, 0) + 1
            current_sums = sums.setdefault(current_type, {"velo": 0.0, "h_mov": 0.0, "v_mov": 0.0})
            current_sums["velo"] += float(_safe_numeric(pd.Series([row.get(velo_column)])).fillna(0.0).iloc[0])
            current_sums["h_mov"] += float(_safe_numeric(pd.Series([row.get(h_mov_column)])).fillna(0.0).iloc[0])
            current_sums["v_mov"] += float(_safe_numeric(pd.Series([row.get(v_mov_column)])).fillna(0.0).iloc[0])

    working[f"{prefix}_primary_fastball_type"] = primary_types.fillna(global_fastball_type)
    working[f"{prefix}_primary_fastball_velo"] = primary_velo.fillna(global_fastball_velo)
    working[f"{prefix}_primary_fastball_h_mov"] = primary_h_mov.fillna(global_fastball_h_mov)
    working[f"{prefix}_primary_fastball_v_mov"] = primary_v_mov.fillna(global_fastball_v_mov)
    return working


def _baseline_reference_frame(frame: pd.DataFrame):
    if frame.empty:
        return frame
    seasons = pd.to_numeric(frame["season"], errors="coerce")
    valid_seasons = seasons.dropna()
    if valid_seasons.empty:
        return frame
    first_season = int(valid_seasons.min())
    baseline = frame.loc[seasons.eq(first_season)].copy()
    return baseline if not baseline.empty else frame


def build_empirical_count_state_values(frame: pd.DataFrame):
    working = frame.copy()
    working["count_bucket"] = (
        pd.to_numeric(working["balls_before_p2"], errors="coerce").fillna(0).astype(int).astype(str)
        + "-"
        + pd.to_numeric(working["strikes_before_p2"], errors="coerce").fillna(0).astype(int).astype(str)
    )
    working["realized_pitcher_value_proxy"] = (
        working["pitch_2_take"] * (
            working["pitch_2_called_strike"] * 0.02
            + working["pitch_2_ball"] * -0.02
        )
        + working["pitch_2_swung_at"] * (
            working["pitch_2_whiff"] * 0.025
            + working["pitch_2_contact"] * working["pitcher_value_on_contact"]
            + working["pitch_2_foul"] * 0.005
        )
    )
    grouped = (
        working.groupby("count_bucket", dropna=False)
        .agg(
            future_pitcher_value=("realized_pitcher_value_proxy", "mean"),
            n_obs=("realized_pitcher_value_proxy", "size"),
        )
        .reset_index()
    )
    split = grouped["count_bucket"].str.split("-", n=1, expand=True)
    grouped["balls"] = pd.to_numeric(split[0], errors="coerce").fillna(0).astype(int)
    grouped["strikes"] = pd.to_numeric(split[1], errors="coerce").fillna(0).astype(int)
    return grouped[["balls", "strikes", "count_bucket", "n_obs", "future_pitcher_value"]]


def build_pitch2_event_premium_lookup(
    pitch_level: pd.DataFrame,
    count_state_lookup: dict[str, float] | None = None):

    event_defaults = {
        "called_strike": 0.0,
        "ball": 0.0,
        "whiff": 0.005,
        "foul": 0.0,
    }
    if pitch_level.empty:
        return {event_kind: {"_default": default} for event_kind, default in event_defaults.items()}

    season_column = "Season" if "Season" in pitch_level.columns else "season"
    required_columns = {
        season_column,
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "balls_before_pitch",
        "strikes_before_pitch",
        "delta_run_exp",
        "is_ball",
    }
    if any(column not in pitch_level.columns for column in required_columns):
        return {event_kind: {"_default": default} for event_kind, default in event_defaults.items()}

    working = pitch_level.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).copy()
    working["delta_run_exp"] = pd.to_numeric(working["delta_run_exp"], errors="coerce").fillna(0.0)
    pa_key = [season_column, "game_date", "game_pk", "at_bat_number"]
    future_from_current = working.groupby(pa_key)["delta_run_exp"].transform(
        lambda series: series.iloc[::-1].cumsum().iloc[::-1]
    )
    working["future_pitcher_value_after_pitch"] = -(future_from_current - working["delta_run_exp"])

    pitch_number = pd.to_numeric(working["pitch_number"], errors="coerce")
    working = working.loc[pitch_number.eq(2)].copy()
    if working.empty:
        return {event_kind: {"_default": default} for event_kind, default in event_defaults.items()}

    balls = pd.to_numeric(working["balls_before_pitch"], errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(working["strikes_before_pitch"], errors="coerce").fillna(0).astype(int)
    default_count_value = float((count_state_lookup or {}).get("_default", 0.0))
    count_state_lookup = count_state_lookup or {"_default": default_count_value}

    called_strike_series = (
        working["is_called_strike"]
        if "is_called_strike" in working.columns
        else pd.Series(0, index=working.index)
    )
    called_strike = pd.to_numeric(called_strike_series, errors="coerce").fillna(0).astype(int)
    if "is_whiff" in working.columns:
        whiff = pd.to_numeric(working["is_whiff"], errors="coerce").fillna(0).astype(int)
    else:
        swing_series = working["is_swing"] if "is_swing" in working.columns else pd.Series(0, index=working.index)
        contact_series = working["is_contact"] if "is_contact" in working.columns else pd.Series(0, index=working.index)
        swing = pd.to_numeric(swing_series, errors="coerce").fillna(0).astype(int)
        contact = pd.to_numeric(contact_series, errors="coerce").fillna(0).astype(int)
        whiff = (swing.eq(1) & contact.eq(0)).astype(int)
    ball = pd.to_numeric(working["is_ball"], errors="coerce").fillna(0).astype(int)
    contact_series = working["is_contact"] if "is_contact" in working.columns else pd.Series(0, index=working.index)
    in_play_series = working["is_in_play"] if "is_in_play" in working.columns else pd.Series(0, index=working.index)
    contact = pd.to_numeric(contact_series, errors="coerce").fillna(0).astype(int)
    in_play = pd.to_numeric(in_play_series, errors="coerce").fillna(0).astype(int)
    foul = (contact.eq(1) & in_play.eq(0)).astype(int)

    next_strike_bucket = balls.astype(str) + "-" + (strikes.clip(upper=1) + 1).clip(upper=2).astype(str)
    next_ball_bucket = (balls + 1).clip(upper=3).astype(str) + "-" + strikes.astype(str)
    future_pitcher_value = pd.to_numeric(working["future_pitcher_value_after_pitch"], errors="coerce")
    event_rows = [
        pd.DataFrame(
            {
                "event_kind": "called_strike",
                "count_bucket": next_strike_bucket,
                "future_pitcher_value": future_pitcher_value,
            }
        ).loc[called_strike.eq(1)].copy(),
        pd.DataFrame(
            {
                "event_kind": "ball",
                "count_bucket": next_ball_bucket,
                "future_pitcher_value": future_pitcher_value,
            }
        ).loc[ball.eq(1)].copy(),
        pd.DataFrame(
            {
                "event_kind": "whiff",
                "count_bucket": next_strike_bucket,
                "future_pitcher_value": future_pitcher_value,
            }
        ).loc[whiff.eq(1)].copy(),
        pd.DataFrame(
            {
                "event_kind": "foul",
                "count_bucket": next_strike_bucket,
                "future_pitcher_value": future_pitcher_value,
            }
        ).loc[foul.eq(1)].copy(),
    ]
    event_rows = pd.concat(event_rows, ignore_index=True)
    if event_rows.empty:
        return {event_kind: {"_default": default} for event_kind, default in event_defaults.items()}

    count_value_series = event_rows["count_bucket"].map(count_state_lookup).fillna(default_count_value)
    event_rows["premium"] = event_rows["future_pitcher_value"] - count_value_series
    summary = (
        event_rows.groupby(["count_bucket", "event_kind"], dropna=False)["premium"]
        .agg(["mean", "size"])
        .reset_index()
    )
    lookup: dict[str, dict[str, float]] = {}
    for event_kind, default in event_defaults.items():
        event_summary = summary.loc[summary["event_kind"].eq(event_kind)].copy()
        event_lookup: dict[str, float] = {}
        valid = event_summary.loc[event_summary["size"].ge(25)].copy()
        for row in valid.itertuples(index=False):
            event_lookup[str(row.count_bucket)] = float(np.clip(row.mean, -0.05, 0.05))
        global_premium = event_rows.loc[event_rows["event_kind"].eq(event_kind), "premium"].mean()
        event_lookup["_default"] = float(
            np.clip(global_premium, -0.05, 0.05) if pd.notna(global_premium) else default
        )
        lookup[event_kind] = event_lookup
    return lookup

@lru_cache(maxsize=1)
def load_available_pitch_level_history():
    frames: list[pd.DataFrame] = []
    for path in sorted(Path(data).glob(glob_pitch_level)):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        season_column = "Season" if "Season" in frame.columns else "season"
        if season_column not in frame.columns:
            continue
        frame[season_column] = pd.to_numeric(frame[season_column], errors="coerce")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined


def _eligible_batter_pitcher_pairs(frame: pd.DataFrame):
    working = frame.copy()
    working["season"] = pd.to_numeric(working["season"], errors="coerce").astype("Int64")
    working["batter_id"] = pd.to_numeric(working["batter_id"], errors="coerce").astype("Int64")
    working["pitcher_id"] = pd.to_numeric(working["pitcher_id"], errors="coerce").astype("Int64")
    valid = working.dropna(subset=["season", "batter_id", "pitcher_id"]).copy()
    batter_counts = (
        valid.groupby(["season", "batter_id"], dropna=False)
        .size()
        .reset_index(name="plate_appearances")
    )
    eligible_batters = {
        (int(row.season), int(row.batter_id))
        for row in batter_counts.itertuples(index=False)
        if int(row.plate_appearances) >= 250
    }

    pitch_level = load_available_pitch_level_history()
    if pitch_level.empty:
        return eligible_batters, set()
    season_column = "Season" if "Season" in pitch_level.columns else "season"
    required_columns = {
        season_column,
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "mlbam_id",
        "outs_when_up",
    }
    if any(column not in pitch_level.columns for column in required_columns):
        return eligible_batters, set()

    pa = pitch_level[list(required_columns)].copy()
    pa = pa.rename(columns={season_column: "season", "mlbam_id": "pitcher_id"})
    pa["season"] = pd.to_numeric(pa["season"], errors="coerce").astype("Int64")
    pa["pitcher_id"] = pd.to_numeric(pa["pitcher_id"], errors="coerce").astype("Int64")
    pa["game_pk"] = pd.to_numeric(pa["game_pk"], errors="coerce").astype("Int64")
    pa["at_bat_number"] = pd.to_numeric(pa["at_bat_number"], errors="coerce").astype("Int64")
    pa["pitch_number"] = pd.to_numeric(pa["pitch_number"], errors="coerce")
    pa["outs_when_up"] = pd.to_numeric(pa["outs_when_up"], errors="coerce")
    pa = pa.dropna(
        subset=["season", "pitcher_id", "game_pk", "at_bat_number", "pitch_number", "outs_when_up"]
    ).copy()
    if pa.empty:
        return eligible_batters, set()

    pa = pa.sort_values(
        ["season", "game_pk", "at_bat_number", "pitch_number"],
        kind="mergesort",
    ).copy()
    pa_summary = (
        pa.groupby(["season", "game_pk", "at_bat_number"], dropna=False)
        .agg(
            pitcher_id=("pitcher_id", "first"),
            start_outs=("outs_when_up", "first"),
        )
        .reset_index()
    )
    pa_summary = pa_summary.sort_values(
        ["season", "game_pk", "at_bat_number"],
        kind="mergesort",
    ).reset_index(drop=True)
    group_key = ["season", "game_pk"]
    pa_summary["next_start_outs"] = pa_summary.groupby(group_key, dropna=False)["start_outs"].shift(-1)
    outs_delta = pa_summary["next_start_outs"] - pa_summary["start_outs"]
    inning_reset = pa_summary["next_start_outs"].lt(pa_summary["start_outs"])
    pa_summary["outs_recorded"] = outs_delta.where(~inning_reset, 3 - pa_summary["start_outs"])
    pa_summary["outs_recorded"] = pa_summary["outs_recorded"].where(
        pa_summary["next_start_outs"].notna(),
        3 - pa_summary["start_outs"],
    )
    pa_summary["outs_recorded"] = (
        pd.to_numeric(pa_summary["outs_recorded"], errors="coerce").fillna(0).clip(lower=0, upper=3)
    )
    pitcher_outs = (
        pa_summary.groupby(["season", "pitcher_id"], dropna=False)["outs_recorded"]
        .sum()
        .reset_index(name="outs_recorded")
    )
    eligible_pitchers = {
        (int(row.season), int(row.pitcher_id))
        for row in pitcher_outs.itertuples(index=False)
        if float(row.outs_recorded) / 3.0 > 50.0
    }
    return eligible_batters, eligible_pitchers


@lru_cache(maxsize=1)
def load_pitch_level_zone_context():
    use_columns = {
        "Season",
        "season",
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "batter",
        "sz_top",
        "sz_bot",
    }
    frames: list[pd.DataFrame] = []
    for path in sorted(Path(data).glob(glob_pitch_level)):
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in use_columns)
        except Exception:
            continue
        season_column = "Season" if "Season" in frame.columns else "season"
        if season_column not in frame.columns:
            continue
        frame = frame.rename(
            columns={
                season_column: "season",
                "pitch_number": "pitch_2_pitch_number",
                "batter": "batter_id",
                "sz_top": "pitch_2_sz_top",
                "sz_bot": "pitch_2_sz_bot",
            }
        )
        frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype("Int64")
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
        frame["game_pk"] = pd.to_numeric(frame["game_pk"], errors="coerce").astype("Int64")
        frame["at_bat_number"] = pd.to_numeric(frame["at_bat_number"], errors="coerce").astype("Int64")
        frame["pitch_2_pitch_number"] = pd.to_numeric(frame["pitch_2_pitch_number"], errors="coerce").astype("Int64")
        frame["batter_id"] = pd.to_numeric(frame["batter_id"], errors="coerce").astype("Int64")
        frame["pitch_2_sz_top"] = pd.to_numeric(frame["pitch_2_sz_top"], errors="coerce")
        frame["pitch_2_sz_bot"] = pd.to_numeric(frame["pitch_2_sz_bot"], errors="coerce")
        frames.append(
            frame[
                [
                    "season",
                    "game_date",
                    "game_pk",
                    "at_bat_number",
                    "pitch_2_pitch_number",
                    "batter_id",
                    "pitch_2_sz_top",
                    "pitch_2_sz_bot",
                ]
            ].copy()
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "season",
                "game_date",
                "game_pk",
                "at_bat_number",
                "pitch_2_pitch_number",
                "batter_id",
                "pitch_2_sz_top",
                "pitch_2_sz_bot",
            ]
        )
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["season", "game_pk", "at_bat_number", "pitch_2_pitch_number"],
        keep="first",
    ).copy()
    return combined


def build_true_continuation_count_state_values(pitch_level: pd.DataFrame):
    if pitch_level.empty:
        return pd.DataFrame(columns=["balls", "strikes", "count_bucket", "n_obs", "future_pitcher_value"])

    season_column = "Season" if "Season" in pitch_level.columns else "season"
    working = pitch_level.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).copy()
    working["delta_run_exp"] = pd.to_numeric(working["delta_run_exp"], errors="coerce").fillna(0.0)
    pa_key = [season_column, "game_date", "game_pk", "at_bat_number"]
    working["future_batter_value_from_state"] = (
        working.groupby(pa_key)["delta_run_exp"]
        .transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
        .astype("float64")
    )
    continuation_states = pd.DataFrame(PITCH2_CONTINUATION_STATES, columns=["balls_before_pitch", "strikes_before_pitch"])
    working = working.loc[pd.to_numeric(working["pitch_number"], errors="coerce").eq(3)].copy()
    working = working.merge(
        continuation_states,
        on=["balls_before_pitch", "strikes_before_pitch"],
        how="inner",
    )
    if working.empty:
        return pd.DataFrame(columns=["balls", "strikes", "count_bucket", "n_obs", "future_pitcher_value"])

    count_values = (
        working.groupby(["balls_before_pitch", "strikes_before_pitch"], as_index=False)
        .agg(
            n_obs=("pitch_number", "size"),
            future_batter_value=("future_batter_value_from_state", "mean"),
        )
        .rename(columns={"balls_before_pitch": "balls", "strikes_before_pitch": "strikes"})
    )
    count_values["future_pitcher_value"] = -pd.to_numeric(count_values["future_batter_value"], errors="coerce")
    count_values["count_bucket"] = (
        count_values["balls"].astype(int).astype(str)
        + "-"
        + count_values["strikes"].astype(int).astype(str)
    )
    return count_values[["balls", "strikes", "count_bucket", "n_obs", "future_pitcher_value"]]


def _attach_batter_height_context(frame: pd.DataFrame):
    zone_context = load_pitch_level_zone_context()
    if zone_context.empty:
        enriched = frame.copy()
        enriched["batter_height_ft"] = default_batter_height_ft
        return enriched

    merge_columns = ["season", "game_pk", "at_bat_number", "pitch_2_pitch_number"]
    enriched = frame.copy()
    for column in merge_columns:
        if column not in enriched.columns:
            enriched["batter_height_ft"] = default_batter_height_ft
            return enriched

    enriched["pitch_2_pitch_number"] = pd.to_numeric(enriched["pitch_2_pitch_number"], errors="coerce").astype("Int64")
    enriched = enriched.merge(
        zone_context,
        on=merge_columns,
        how="left",
        suffixes=("", "_zone"),
        validate="one_to_one",
    )

    top_height = enriched["pitch_2_sz_top"] / called_strike_height_top_ratio
    bot_height = enriched["pitch_2_sz_bot"] / called_strike_height_bot_ratio
    row_height = pd.concat([top_height, bot_height], axis=1).mean(axis=1, skipna=True)
    row_height = row_height.clip(lower=min_batter_height_ft, upper=max_batter_height_ft)

    batter_height_lookup = (
        pd.DataFrame(
            {
                "batter_id": pd.to_numeric(enriched["batter_id"], errors="coerce"),
                "row_height_ft": row_height,
            }
        )
        .dropna(subset=["batter_id", "row_height_ft"])
        .groupby("batter_id", dropna=False)["row_height_ft"]
        .median()
    )
    default_height = float(batter_height_lookup.median()) if len(batter_height_lookup) else default_batter_height_ft
    batter_ids = pd.to_numeric(enriched["batter_id"], errors="coerce")
    enriched["batter_height_ft"] = (
        batter_ids.map(batter_height_lookup)
        .fillna(row_height)
        .fillna(default_height)
        .clip(lower=min_batter_height_ft, upper=max_batter_height_ft)
    )
    return enriched.drop(columns=["batter_id_zone", "pitch_2_sz_top", "pitch_2_sz_bot"], errors="ignore")


def load_source_dataset(source_path: str):
    frame = pd.read_csv(source_path)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["_source_row_order"] = np.arange(len(frame), dtype=int)
    chronological_sort_columns = [
        "season",
        "game_date",
        "game_pk",
        "at_bat_number",
        "_source_row_order",
    ]
    available_sort_columns = [column for column in chronological_sort_columns if column in frame.columns]
    frame = frame.sort_values(available_sort_columns, kind="mergesort").reset_index(drop=True)
    eligible_batters, eligible_pitchers = _eligible_batter_pitcher_pairs(frame)
    if eligible_batters and eligible_pitchers:
        season_series = pd.to_numeric(frame["season"], errors="coerce")
        batter_series = pd.to_numeric(frame["batter_id"], errors="coerce")
        pitcher_series = pd.to_numeric(frame["pitcher_id"], errors="coerce")
        batter_keys = list(
            zip(
                season_series.fillna(-1).astype(int).tolist(),
                batter_series.fillna(-1).astype(int).tolist(),
            )
        )
        pitcher_keys = list(
            zip(
                season_series.fillna(-1).astype(int).tolist(),
                pitcher_series.fillna(-1).astype(int).tolist(),
            )
        )
        eligible_mask = pd.Series(
            [
                batter_key in eligible_batters and pitcher_key in eligible_pitchers
                for batter_key, pitcher_key in zip(batter_keys, pitcher_keys)
            ],
            index=frame.index,
        )
        frame = frame.loc[eligible_mask].copy().reset_index(drop=True)
    if "game_pk" not in frame.columns:
        frame["game_pk"] = pd.Series(pd.array([pd.NA] * len(frame), dtype="Int64"), index=frame.index)
    else:
        frame["game_pk"] = pd.to_numeric(frame["game_pk"], errors="coerce").astype("Int64")
    if "at_bat_number" not in frame.columns:
        frame["at_bat_number"] = pd.Series(pd.array([pd.NA] * len(frame), dtype="Int64"), index=frame.index)
    else:
        frame["at_bat_number"] = pd.to_numeric(frame["at_bat_number"], errors="coerce").astype("Int64")
    if "pitch_2_pitch_number" in frame.columns:
        frame["pitch_2_pitch_number"] = pd.to_numeric(frame["pitch_2_pitch_number"], errors="coerce").astype("Int64")

    frame["state_id"] = (
        "state_"
        + frame.index.astype(int).astype(str).str.zfill(7)
    )
    frame["balls_before_p2"] = np.where(frame["pitch_1_strike"].eq(1), 0, 1)
    frame["strikes_before_p2"] = np.where(frame["pitch_1_strike"].eq(1), 1, 0)
    frame["count_bucket"] = (
        frame["balls_before_p2"].astype(int).astype(str)
        + "-"
        + frame["strikes_before_p2"].astype(int).astype(str)
    )
    frame["outs"] = pd.NA
    frame["base_state"] = "unknown"
    frame["score_diff"] = pd.NA
    frame["pitch_2_take"] = 1 - frame["pitch_2_swung_at"]
    frame["pitch_2_whiff"] = np.where(
        frame["pitch_2_swung_at"].eq(1) & frame["pitch_2_contact"].eq(0),
        1,
        0,
    )
    frame["pitch_2_called_strike"] = np.where(
        frame["pitch_2_take"].eq(1) & frame["pitch_2_strike"].eq(1),
        1,
        0,
    )
    frame["pitch_2_ball"] = np.where(
        frame["pitch_2_take"].eq(1)
        & frame["pitch_2_strike"].eq(0)
        & frame["pitch_2_hit_by_pitch"].eq(0),
        1,
        0,
    )
    frame["pitch_2_foul"] = np.where(
        frame["pitch_2_contact"].eq(1) & frame["pitch_2_in_play"].eq(0),
        1,
        0,
    )
    frame["pitch_2_double_or_triple"] = frame["pitch_2_double"] + frame["pitch_2_triple"]
    frame["pitch_1_perceived_velo"] = _perceived_velocity(
        frame["pitch_1_velo"],
        frame["pitch_1_extension"],
    )
    frame["pitch_1_zone_distance"] = _zone_distance(frame["pitch_1_plate_x"], frame["pitch_1_plate_z"])
    frame = _attach_primary_fastball_reference(
        frame,
        pitch_type_column="pitch_1_type",
        velo_column="pitch_1_velo",
        h_mov_column="pitch_1_h_mov",
        v_mov_column="pitch_1_v_mov",
        prefix="pitch_1",
    )
    p1_fastball, p1_breaking, p1_offspeed = _pitch_family_flags(frame["pitch_1_type"])
    p2_fastball, p2_breaking, p2_offspeed = _pitch_family_flags(frame["pitch_2_type"])
    frame["pitch_1_is_fastball"] = p1_fastball
    frame["pitch_2_is_fastball"] = p2_fastball
    frame["pitch_1_is_breaking"] = p1_breaking
    frame["pitch_2_is_breaking"] = p2_breaking
    frame["pitch_1_is_offspeed"] = p1_offspeed
    frame["pitch_2_is_offspeed"] = p2_offspeed
    frame["pitch_1_family"] = _pitch_family_label(frame["pitch_1_type"])
    frame["pitch_2_family"] = _pitch_family_label(frame["pitch_2_type"])
    frame["pitch_1_delta_velo_vs_primary_fastball"] = _safe_numeric(frame["pitch_1_velo"]) - frame["pitch_1_primary_fastball_velo"]
    frame["pitch_1_delta_h_mov_vs_primary_fastball"] = _safe_numeric(frame["pitch_1_h_mov"]) - frame["pitch_1_primary_fastball_h_mov"]
    frame["pitch_1_delta_v_mov_vs_primary_fastball"] = _safe_numeric(frame["pitch_1_v_mov"]) - frame["pitch_1_primary_fastball_v_mov"]
    frame["same_pitch_type"] = (
        frame["pitch_1_type"].fillna("UN").astype(str).str.upper()
        == frame["pitch_2_type"].fillna("UN").astype(str).str.upper()
    ).astype(int)
    frame["same_pitch_family"] = (
        (frame["pitch_1_is_fastball"].eq(1) & frame["pitch_2_is_fastball"].eq(1))
        | (frame["pitch_1_is_breaking"].eq(1) & frame["pitch_2_is_breaking"].eq(1))
        | (frame["pitch_1_is_offspeed"].eq(1) & frame["pitch_2_is_offspeed"].eq(1))
    ).astype(int)
    frame["batter_value_on_contact"] = (
        frame["pitch_2_out"] * hit_value["out"]
        + frame["pitch_2_single"] * hit_value["single"]
        + frame["pitch_2_double_or_triple"] * hit_value["double_or_triple"]
        + frame["pitch_2_home_run"] * hit_value["home_run"]
    )
    frame["pitcher_value_on_contact"] = -frame["batter_value_on_contact"]
    frame = _attach_pitch_type_shape_reference(frame, prefix="pitch_1")
    frame = _attach_pitch_type_shape_reference(frame, prefix="pitch_2")
    frame = _attach_batter_height_context(frame)
    frame["pitch_1_bucket"] = locate_bucket_series(
        _safe_numeric(frame["pitch_1_plate_x"]),
        _safe_numeric(frame["pitch_1_plate_z"]),
        batter_height_ft=frame["batter_height_ft"],
    )
    frame["pitch_1_shape_bucket_tight"] = derive_relative_pitch_shape_bucket(
        frame["pitch_1_delta_velo_vs_type_mean"],
        frame["pitch_1_delta_spin_rate_vs_type_mean"],
        frame["pitch_1_delta_h_mov_vs_type_mean"],
        frame["pitch_1_delta_v_mov_vs_type_mean"],
        level="tight",
    )
    frame["pitch_1_shape_bucket_medium"] = derive_relative_pitch_shape_bucket(
        frame["pitch_1_delta_velo_vs_type_mean"],
        frame["pitch_1_delta_spin_rate_vs_type_mean"],
        frame["pitch_1_delta_h_mov_vs_type_mean"],
        frame["pitch_1_delta_v_mov_vs_type_mean"],
        level="medium",
    )
    frame["pitch_2_shape_bucket_tight"] = derive_relative_pitch_shape_bucket(
        frame["pitch_2_delta_velo_vs_type_mean"],
        frame["pitch_2_delta_spin_rate_vs_type_mean"],
        frame["pitch_2_delta_h_mov_vs_type_mean"],
        frame["pitch_2_delta_v_mov_vs_type_mean"],
        level="tight",
    )
    frame["pitch_2_shape_bucket_medium"] = derive_relative_pitch_shape_bucket(
        frame["pitch_2_delta_velo_vs_type_mean"],
        frame["pitch_2_delta_spin_rate_vs_type_mean"],
        frame["pitch_2_delta_h_mov_vs_type_mean"],
        frame["pitch_2_delta_v_mov_vs_type_mean"],
        level="medium",
    )
    return frame.drop(columns="_source_row_order")


def add_state_priors(frame: pd.DataFrame):
    frame = frame.copy()
    baseline = _baseline_reference_frame(frame)
    global_swing = baseline["pitch_2_swung_at"].mean()
    global_contact = baseline.loc[baseline["pitch_2_swung_at"].eq(1), "pitch_2_contact"].mean()
    global_in_play = baseline.loc[baseline["pitch_2_contact"].eq(1), "pitch_2_in_play"].mean()
    global_exit_velo = float(_safe_numeric(baseline.loc[baseline["pitch_2_in_play"].eq(1), "pitch_2_exit_velo"]).mean())
    global_launch_angle = float(_safe_numeric(baseline.loc[baseline["pitch_2_in_play"].eq(1), "pitch_2_launch_angle"]).mean())

    frame["pitcher_pitch2_swing_prior"] = _group_shifted_rate(
        frame,
        ["pitcher_id", "pitch_2_type"],
        "pitch_2_swung_at",
        prior_mean=float(global_swing),
    )
    frame["pitcher_pitch2_contact_prior"] = _group_shifted_conditional_rate(
        frame,
        ["pitcher_id", "pitch_2_type"],
        "pitch_2_contact",
        "pitch_2_swung_at",
        prior_mean=float(global_contact),
    )
    frame["pitcher_pitch2_in_play_prior"] = _group_shifted_conditional_rate(
        frame,
        ["pitcher_id", "pitch_2_type"],
        "pitch_2_in_play",
        "pitch_2_contact",
        prior_mean=float(global_in_play),
    )
    frame["batter_pitch2_swing_prior"] = _group_shifted_rate(
        frame,
        ["batter_id", "pitch_2_type"],
        "pitch_2_swung_at",
        prior_mean=float(global_swing),
    )
    frame["batter_pitch2_contact_prior"] = _group_shifted_conditional_rate(
        frame,
        ["batter_id", "pitch_2_type"],
        "pitch_2_contact",
        "pitch_2_swung_at",
        prior_mean=float(global_contact),
    )
    frame["batter_pitch2_in_play_prior"] = _group_shifted_conditional_rate(
        frame,
        ["batter_id", "pitch_2_type"],
        "pitch_2_in_play",
        "pitch_2_contact",
        prior_mean=float(global_in_play),
    )
    frame["matchup_pitch2_swing_prior"] = _group_shifted_rate(
        frame,
        ["pitcher_id", "batter_id", "pitch_2_type"],
        "pitch_2_swung_at",
        prior_mean=float(global_swing),
    )
    frame["matchup_pitch2_contact_prior"] = _group_shifted_conditional_rate(
        frame,
        ["pitcher_id", "batter_id", "pitch_2_type"],
        "pitch_2_contact",
        "pitch_2_swung_at",
        prior_mean=float(global_contact),
    )
    frame["pitcher_recent_swing_rate_25"] = _group_shifted_rolling_rate(
        frame,
        ["pitcher_id"],
        "pitch_2_swung_at",
        window=25,
        prior_mean=float(global_swing),
    )
    frame["pitcher_recent_contact_rate_25"] = _group_shifted_rolling_conditional_rate(
        frame,
        ["pitcher_id"],
        "pitch_2_contact",
        "pitch_2_swung_at",
        window=25,
        prior_mean=float(global_contact),
    )
    frame["pitcher_recent_in_play_rate_25"] = _group_shifted_rolling_conditional_rate(
        frame,
        ["pitcher_id"],
        "pitch_2_in_play",
        "pitch_2_contact",
        window=25,
        prior_mean=float(global_in_play),
    )
    frame["batter_recent_swing_rate_25"] = _group_shifted_rolling_rate(
        frame,
        ["batter_id"],
        "pitch_2_swung_at",
        window=25,
        prior_mean=float(global_swing),
    )
    frame["batter_recent_contact_rate_25"] = _group_shifted_rolling_conditional_rate(
        frame,
        ["batter_id"],
        "pitch_2_contact",
        "pitch_2_swung_at",
        window=25,
        prior_mean=float(global_contact),
    )
    frame["batter_recent_in_play_rate_25"] = _group_shifted_rolling_conditional_rate(
        frame,
        ["batter_id"],
        "pitch_2_in_play",
        "pitch_2_contact",
        window=25,
        prior_mean=float(global_in_play),
    )
    frame["matchup_recent_swing_rate_25"] = _group_shifted_rolling_rate(
        frame,
        ["pitcher_id", "batter_id"],
        "pitch_2_swung_at",
        window=25,
        prior_mean=float(global_swing),
    )
    frame["matchup_recent_contact_rate_25"] = _group_shifted_rolling_conditional_rate(
        frame,
        ["pitcher_id", "batter_id"],
        "pitch_2_contact",
        "pitch_2_swung_at",
        window=25,
        prior_mean=float(global_contact),
    )
    frame["matchup_recent_in_play_rate_25"] = _group_shifted_rolling_conditional_rate(
        frame,
        ["pitcher_id", "batter_id"],
        "pitch_2_in_play",
        "pitch_2_contact",
        window=25,
        prior_mean=float(global_in_play),
    )
    pitch_type_global_usage = (
        frame["pitch_2_type"]
        .fillna("Unknown")
        .astype(str)
        .value_counts(normalize=True)
        .mean()
    )
    frame["pitcher_recent_pitchtype_usage_25"] = _group_shifted_rolling_share(
        frame,
        ["pitcher_id", "pitch_2_type"],
        ["pitcher_id"],
        window=25,
        prior_mean=float(pitch_type_global_usage),
    )
    frame["batter_recent_pitchtype_swing_rate_25"] = _group_shifted_rolling_rate(
        frame,
        ["batter_id", "pitch_2_type"],
        "pitch_2_swung_at",
        window=25,
        prior_mean=float(global_swing),
    )
    frame["pitcher_recent_exit_velo_allowed_50"] = _group_shifted_rolling_masked_mean(
        frame,
        ["pitcher_id"],
        "pitch_2_exit_velo",
        "pitch_2_in_play",
        window=50,
        default_value=global_exit_velo,
    )
    frame["pitcher_recent_launch_angle_allowed_50"] = _group_shifted_rolling_masked_mean(
        frame,
        ["pitcher_id"],
        "pitch_2_launch_angle",
        "pitch_2_in_play",
        window=50,
        default_value=global_launch_angle,
    )
    frame["batter_recent_exit_velo_50"] = _group_shifted_rolling_masked_mean(
        frame,
        ["batter_id"],
        "pitch_2_exit_velo",
        "pitch_2_in_play",
        window=50,
        default_value=global_exit_velo,
    )
    frame["batter_recent_launch_angle_50"] = _group_shifted_rolling_masked_mean(
        frame,
        ["batter_id"],
        "pitch_2_launch_angle",
        "pitch_2_in_play",
        window=50,
        default_value=global_launch_angle,
    )
    return frame


def build_pitch2_states(frame: pd.DataFrame):
    base = frame.copy()

    states = base.copy()
    states["called_strike_bucket"] = (
        states["balls_before_p2"].astype(int).astype(str)
        + "-"
        + (states["strikes_before_p2"].clip(upper=1) + 1).clip(upper=2).astype(int).astype(str)
    )
    states["ball_bucket"] = (
        (states["balls_before_p2"] + 1).clip(upper=3).astype(int).astype(str)
        + "-"
        + states["strikes_before_p2"].astype(int).astype(str)
    )
    states["whiff_bucket"] = states["called_strike_bucket"]
    states["foul_bucket"] = states["called_strike_bucket"]
    return states.loc[:, state]


def build_pitch2_action_zone_priors(frame: pd.DataFrame):
    working = frame.copy()
    working["pitch_2_zone_bucket"] = derive_pitch_zone_bucket(
        working["pitch_2_plate_x"],
        working["pitch_2_plate_z"],
        working["batter_handedness"],
        working["batter_height_ft"],
    )
    baseline = _baseline_reference_frame(working)
    global_swing = float(baseline["pitch_2_swung_at"].mean())
    swing_rows = baseline.loc[baseline["pitch_2_swung_at"].eq(1)].copy()
    global_contact = float(swing_rows["pitch_2_contact"].mean()) if len(swing_rows) else 0.0
    global_whiff = float(swing_rows["pitch_2_whiff"].mean()) if len(swing_rows) else 0.0
    contact_rows = baseline.loc[baseline["pitch_2_contact"].eq(1)].copy()
    global_in_play = float(contact_rows["pitch_2_in_play"].mean()) if len(contact_rows) else 0.0

    pitchtype_zone_swing_prior = _group_shifted_rate(
        working,
        ["pitch_2_type", "pitch_2_zone_bucket"],
        "pitch_2_swung_at",
        prior_mean=global_swing,
    )
    pitchtype_zone_contact_prior = _group_shifted_conditional_rate(
        working,
        ["pitch_2_type", "pitch_2_zone_bucket"],
        "pitch_2_contact",
        "pitch_2_swung_at",
        prior_mean=global_contact,
    )
    pitchtype_zone_whiff_prior = _group_shifted_conditional_rate(
        working,
        ["pitch_2_type", "pitch_2_zone_bucket"],
        "pitch_2_whiff",
        "pitch_2_swung_at",
        prior_mean=global_whiff,
    )
    pitchtype_zone_in_play_prior = _group_shifted_conditional_rate(
        working,
        ["pitch_2_type", "pitch_2_zone_bucket"],
        "pitch_2_in_play",
        "pitch_2_contact",
        prior_mean=global_in_play,
    )

    priors = pd.DataFrame({"state_id": working["state_id"]})
    priors["pitch_2_zone_bucket"] = working["pitch_2_zone_bucket"]
    priors["batter_pitchtype_zone_swing_prior"] = _group_shifted_rate_with_prior_series(
        working,
        ["batter_id", "pitch_2_type", "pitch_2_zone_bucket"],
        "pitch_2_swung_at",
        pitchtype_zone_swing_prior,
    )
    priors["batter_pitchtype_zone_contact_prior"] = _group_shifted_conditional_rate_with_prior_series(
        working,
        ["batter_id", "pitch_2_type", "pitch_2_zone_bucket"],
        "pitch_2_contact",
        "pitch_2_swung_at",
        pitchtype_zone_contact_prior,
    )
    priors["batter_pitchtype_zone_whiff_prior"] = _group_shifted_conditional_rate_with_prior_series(
        working,
        ["batter_id", "pitch_2_type", "pitch_2_zone_bucket"],
        "pitch_2_whiff",
        "pitch_2_swung_at",
        pitchtype_zone_whiff_prior,
    )
    priors["batter_pitchtype_zone_in_play_prior"] = _group_shifted_conditional_rate_with_prior_series(
        working,
        ["batter_id", "pitch_2_type", "pitch_2_zone_bucket"],
        "pitch_2_in_play",
        "pitch_2_contact",
        pitchtype_zone_in_play_prior,
    )
    return priors


def build_pitcher_arsenal_profiles(frame: pd.DataFrame):
    working = _attach_primary_fastball_reference(
        frame,
        pitch_type_column="pitch_2_type",
        velo_column="pitch_2_velo",
        h_mov_column="pitch_2_h_mov",
        v_mov_column="pitch_2_v_mov",
        prefix="pitch_2",
    )
    profiles = working[["state_id", "pitcher_id", "pitch_2_type", "game_date", "season"]].copy()
    profiles = profiles.rename(columns={"pitch_2_type": "pitch_type", "game_date": "snapshot_date"})

    baseline = _baseline_reference_frame(working)
    global_usage = (
        baseline["pitch_2_type"]
        .fillna("Unknown")
        .astype(str)
        .value_counts(normalize=True)
        .to_dict()
    )
    total_prior_pitcher = working.groupby("pitcher_id", dropna=False).cumcount()
    prior_type_count = working.groupby(["pitcher_id", "pitch_2_type"], dropna=False).cumcount()
    pitch_type_support = max(int(working["pitch_2_type"].fillna("Unknown").astype(str).nunique()), 1)
    prior_mean_usage = working["pitch_2_type"].fillna("Unknown").astype(str).map(global_usage).fillna(1.0 / pitch_type_support)
    profiles["usage_rate"] = (
        prior_type_count + prior_mean_usage
    ) / (total_prior_pitcher + 1.0)

    numeric_defaults = {
        "pitch_2_velo": float(_safe_numeric(baseline["pitch_2_velo"]).mean()),
        "pitch_2_h_mov": float(_safe_numeric(baseline["pitch_2_h_mov"]).mean()),
        "pitch_2_v_mov": float(_safe_numeric(baseline["pitch_2_v_mov"]).mean()),
        "pitch_2_spin_rate": float(_safe_numeric(baseline["pitch_2_spin_rate"]).mean()),
        "pitch_2_extension": float(_safe_numeric(baseline["pitch_2_extension"]).mean()),
        "pitch_2_release_x": float(_safe_numeric(baseline["pitch_2_release_x"]).mean()),
        "pitch_2_release_z": float(_safe_numeric(baseline["pitch_2_release_z"]).mean()),
        "pitch_2_plate_x": float(_safe_numeric(baseline["pitch_2_plate_x"]).mean()),
        "pitch_2_plate_z": float(_safe_numeric(baseline["pitch_2_plate_z"]).mean()),
    }
    group_columns = ["pitcher_id", "pitch_2_type"]
    profiles["avg_velo"] = _group_shifted_mean(working, group_columns, "pitch_2_velo", numeric_defaults["pitch_2_velo"])
    profiles["avg_h_mov"] = _group_shifted_mean(working, group_columns, "pitch_2_h_mov", numeric_defaults["pitch_2_h_mov"])
    profiles["avg_v_mov"] = _group_shifted_mean(working, group_columns, "pitch_2_v_mov", numeric_defaults["pitch_2_v_mov"])
    profiles["avg_spin_rate"] = _group_shifted_mean(working, group_columns, "pitch_2_spin_rate", numeric_defaults["pitch_2_spin_rate"])
    profiles["avg_extension"] = _group_shifted_mean(working, group_columns, "pitch_2_extension", numeric_defaults["pitch_2_extension"])
    profiles["avg_release_x"] = _group_shifted_mean(working, group_columns, "pitch_2_release_x", numeric_defaults["pitch_2_release_x"])
    profiles["avg_release_z"] = _group_shifted_mean(working, group_columns, "pitch_2_release_z", numeric_defaults["pitch_2_release_z"])
    profiles["avg_plate_x"] = _group_shifted_mean(working, group_columns, "pitch_2_plate_x", numeric_defaults["pitch_2_plate_x"])
    profiles["avg_plate_z"] = _group_shifted_mean(working, group_columns, "pitch_2_plate_z", numeric_defaults["pitch_2_plate_z"])
    profiles["delta_velo_vs_primary_fastball"] = profiles["avg_velo"] - working["pitch_2_primary_fastball_velo"].to_numpy(dtype=float)
    profiles["delta_h_mov_vs_primary_fastball"] = profiles["avg_h_mov"] - working["pitch_2_primary_fastball_h_mov"].to_numpy(dtype=float)
    profiles["delta_v_mov_vs_primary_fastball"] = profiles["avg_v_mov"] - working["pitch_2_primary_fastball_v_mov"].to_numpy(dtype=float)
    profiles["n_pitches"] = prior_type_count
    return profiles.loc[:, arsenal]


def build_pitch_target_distributions(frame: pd.DataFrame):
    working = frame.sort_values(["game_date", "state_id"]).copy()
    working["pitch_2_bucket"] = locate_bucket_series(
        _safe_numeric(working["pitch_2_plate_x"]),
        _safe_numeric(working["pitch_2_plate_z"]),
        batter_height_ft=working["batter_height_ft"],
    )

    batter_context_history: dict[tuple[object, ...], Counter[str]] = defaultdict(Counter)
    hand_context_history: dict[tuple[object, ...], Counter[str]] = defaultdict(Counter)
    broad_context_history: dict[tuple[object, ...], Counter[str]] = defaultdict(Counter)
    default_context_history: dict[tuple[object, ...], Counter[str]] = defaultdict(Counter)
    records: list[dict[str, object]] = []

    def emit_top_buckets(
        *,
        state_id: object,
        pitcher_id: object,
        pitch_2_type: object,
        pitch_1_type: object,
        pitch_1_bucket: object,
        batter_id: object,
        batter_handedness: object,
        batter_height_ft: object,
        count_bucket: object,
        game_date: object,
        scope: str,
        history: Counter[str]):

        total = int(sum(history.values()))
        if total <= 0:
            return
        ranked = sorted(history.items(), key=lambda item: (-item[1], item[0]))[:3]
        for rank, (bucket_id, bucket_count) in enumerate(ranked, start=1):
            center_x, center_z = bucket_center(bucket_id, batter_height_ft=batter_height_ft)
            records.append(
                {
                    "state_id": state_id,
                    "pitcher_id": pitcher_id,
                    "pitch_type": pitch_2_type,
                    "pitch_1_type": pitch_1_type,
                    "pitch_1_bucket": pitch_1_bucket,
                    "batter_id": batter_id,
                    "batter_handedness": batter_handedness,
                    "count_bucket": count_bucket,
                    "target_context_scope": scope,
                    "snapshot_date": game_date,
                    "pitch_2_bucket": bucket_id,
                    "bucket_center_x": float(center_x),
                    "bucket_center_z": float(center_z),
                    "rank": rank,
                    "weight": float(bucket_count / total),
                    "n_obs": int(bucket_count),
                }
            )

    for row in working.itertuples(index=False):
        batter_context_key = (
            row.batter_id,
            row.pitch_1_type,
            row.pitch_1_bucket,
            row.pitch_2_type,
            row.count_bucket,
        )
        hand_context_key = (
            row.pitcher_id,
            row.pitch_1_type,
            row.pitch_1_bucket,
            row.pitch_2_type,
            row.batter_handedness,
            row.count_bucket,
        )
        broad_context_key = (
            row.pitcher_id,
            row.pitch_2_type,
        )
        default_context_key = (row.pitch_2_type,)

        emit_top_buckets(
            state_id=row.state_id,
            pitcher_id=row.pitcher_id,
            pitch_2_type=row.pitch_2_type,
            pitch_1_type=row.pitch_1_type,
            pitch_1_bucket=row.pitch_1_bucket,
            batter_id=row.batter_id,
            batter_handedness=row.batter_handedness,
            batter_height_ft=row.batter_height_ft,
            count_bucket=row.count_bucket,
            game_date=row.game_date,
            scope=batter_specific_target_scope,
            history=batter_context_history[batter_context_key],
        )
        emit_top_buckets(
            state_id=row.state_id,
            pitcher_id=row.pitcher_id,
            pitch_2_type=row.pitch_2_type,
            pitch_1_type=row.pitch_1_type,
            pitch_1_bucket=row.pitch_1_bucket,
            batter_id=row.batter_id,
            batter_handedness=row.batter_handedness,
            batter_height_ft=row.batter_height_ft,
            count_bucket=row.count_bucket,
            game_date=row.game_date,
            scope=handidness_target_scope,
            history=hand_context_history[hand_context_key],
        )
        emit_top_buckets(
            state_id=row.state_id,
            pitcher_id=row.pitcher_id,
            pitch_2_type=row.pitch_2_type,
            pitch_1_type=row.pitch_1_type,
            pitch_1_bucket=row.pitch_1_bucket,
            batter_id=row.batter_id,
            batter_handedness=row.batter_handedness,
            batter_height_ft=row.batter_height_ft,
            count_bucket=row.count_bucket,
            game_date=row.game_date,
            scope=broad_target_scope,
            history=broad_context_history[broad_context_key],
        )
        emit_top_buckets(
            state_id=row.state_id,
            pitcher_id=row.pitcher_id,
            pitch_2_type=row.pitch_2_type,
            pitch_1_type=row.pitch_1_type,
            pitch_1_bucket=row.pitch_1_bucket,
            batter_id=row.batter_id,
            batter_handedness=row.batter_handedness,
            batter_height_ft=row.batter_height_ft,
            count_bucket=row.count_bucket,
            game_date=row.game_date,
            scope=default_target_scope,
            history=default_context_history[default_context_key],
        )

        realized_bucket = str(row.pitch_2_bucket)
        batter_context_history[batter_context_key][realized_bucket] += 1
        hand_context_history[hand_context_key][realized_bucket] += 1
        broad_context_history[broad_context_key][realized_bucket] += 1
        default_context_history[default_context_key][realized_bucket] += 1

    grouped = pd.DataFrame.from_records(records)
    if grouped.empty:
        grouped = pd.DataFrame(columns=target_distribution)
    return grouped.loc[:, target_distribution]


def build_pitch2_observed_actions(
    frame: pd.DataFrame,
    target_distributions: pd.DataFrame):

    actions = frame.copy()
    actions["pitch_2_bucket"] = locate_bucket_series(
        _safe_numeric(actions["pitch_2_plate_x"]),
        _safe_numeric(actions["pitch_2_plate_z"]),
        batter_height_ft=actions["batter_height_ft"],
    )
    bucket_centers = bucket_center_frame(
        actions["pitch_2_bucket"],
        batter_height_ft=actions["batter_height_ft"],
    )
    actions["pitch_2_bucket_row"] = bucket_centers["bucket_row"].to_numpy()
    actions["pitch_2_bucket_col"] = bucket_centers["bucket_col"].to_numpy()
    actions["intended_target_x"] = bucket_centers["bucket_center_x"].to_numpy(dtype=float)
    actions["intended_target_z"] = bucket_centers["bucket_center_z"].to_numpy(dtype=float)
    actions["observed_plate_x"] = _safe_numeric(actions["pitch_2_plate_x"])
    actions["observed_plate_z"] = _safe_numeric(actions["pitch_2_plate_z"])
    actions["pitch_2_zone_bucket"] = derive_pitch_zone_bucket(
        actions["observed_plate_x"],
        actions["observed_plate_z"],
        actions["batter_handedness"],
        actions["batter_height_ft"],
    )
    for prior_column in (
        "batter_pitchtype_zone_swing_prior",
        "batter_pitchtype_zone_contact_prior",
        "batter_pitchtype_zone_whiff_prior",
        "batter_pitchtype_zone_in_play_prior",
    ):
        actions[prior_column] = np.nan
    return actions.loc[:, observed_action]


def build_pitch2_outcomes(frame: pd.DataFrame):
    outcomes = pd.DataFrame(
        {
            "state_id": frame["state_id"],
            "take": frame["pitch_2_take"],
            "swing": frame["pitch_2_swung_at"],
            "called_strike": frame["pitch_2_called_strike"],
            "ball": frame["pitch_2_ball"],
            "whiff": frame["pitch_2_whiff"],
            "contact": frame["pitch_2_contact"],
            "foul": frame["pitch_2_foul"],
            "in_play": frame["pitch_2_in_play"],
            "exit_velo": frame["pitch_2_exit_velo"],
            "launch_angle": frame["pitch_2_launch_angle"],
            "batted_ball_type": _derive_batted_ball_type(frame["pitch_2_launch_angle"]),
            "ev_band": _derive_exit_velo_band(frame["pitch_2_exit_velo"]),
            "single": frame["pitch_2_single"],
            "double": frame["pitch_2_double"],
            "triple": frame["pitch_2_triple"],
            "double_or_triple": frame["pitch_2_double_or_triple"],
            "home_run": frame["pitch_2_home_run"],
            "out": frame["pitch_2_out"],
            "batter_value_on_contact": frame["batter_value_on_contact"],
            "pitcher_value_on_contact": frame["pitcher_value_on_contact"],
        }
    )
    return outcomes.loc[:, outcomes]


def build_views(
    states: pd.DataFrame,
    observed_actions: pd.DataFrame,
    outcomes: pd.DataFrame,
    arsenal_profiles: pd.DataFrame):

    arsenal = arsenal_profiles.rename(columns={"pitch_type": "pitch_2_type"})
    event_tree = (
        states.merge(observed_actions, on="state_id", how="inner")
        .merge(arsenal, on=["state_id", "pitcher_id", "pitch_2_type", "season"], how="left", validate="one_to_one")
    )
    event_tree = event_tree.merge(outcomes, on="state_id", how="inner")
    planner_eval = event_tree.copy()
    planner_eval["observed_outcome_bucket"] = np.select(
        [
            planner_eval["home_run"].eq(1),
            planner_eval["double_or_triple"].eq(1),
            planner_eval["single"].eq(1),
        ],
        [
            "home_run",
            "double_or_triple",
            "single",
        ],
        default="out",
    )
    return event_tree, planner_eval


def build_v2_tables(source_path: str):
    base = load_source_dataset(source_path)
    states = build_pitch2_states(base)
    arsenal_profiles = build_pitcher_arsenal_profiles(base)
    target_distributions = build_pitch_target_distributions(base)
    observed_actions = build_pitch2_observed_actions(base, target_distributions)
    outcomes = build_pitch2_outcomes(base)
    event_tree_view, planner_eval_view = build_views(
        states,
        observed_actions,
        outcomes,
        arsenal_profiles,
    )
    return V2Tables(
        pitch2_states=states,
        pitcher_arsenal_profiles=arsenal_profiles,
        pitch_target_distributions=target_distributions,
        pitch2_observed_actions=observed_actions,
        pitch2_outcomes=outcomes,
        pitch2_event_tree_view=event_tree_view,
        pitch2_planner_eval_view=planner_eval_view,
    )


def _history_lookup_frame(planner_eval_view: pd.DataFrame):
    frame = planner_eval_view.copy()
    frame = frame.loc[pd.to_numeric(frame["season"], errors="coerce").eq(2025)].copy()
    frame["batter_id"] = frame["batter_id"].fillna("").astype(str)
    frame["pitcher_id"] = frame["pitcher_id"].fillna("").astype(str)
    frame["pitcher_handedness"] = frame["pitcher_handedness"].fillna("").astype(str).str.upper()
    frame["batter_handedness"] = frame["batter_handedness"].fillna("").astype(str).str.upper()
    frame["pitch_1_type"] = frame["pitch_1_type"].fillna("").astype(str).str.upper()
    frame["pitch_1_bucket"] = frame["pitch_1_bucket"].fillna("").astype(str)
    frame["pitch_2_type"] = frame["pitch_2_type"].fillna("").astype(str).str.upper()
    frame["count_bucket"] = frame["count_bucket"].fillna("").astype(str)
    frame["pitch_2_bucket"] = frame["pitch_2_bucket"].fillna("").astype(str)
    frame = frame.loc[frame["pitch_2_bucket"].ne("")].copy()
    return frame[
        [
            "batter_id",
            "pitcher_id",
            "pitcher_handedness",
            "batter_handedness",
            "pitch_1_type",
            "pitch_1_bucket",
            "pitch_2_type",
            "count_bucket",
            "pitch_2_bucket",
        ]
    ].reset_index(drop=True)


def _nested_append_bucket(tree: dict[str, object], path: tuple[str, ...], bucket_id: str):
    current = tree
    for part in path[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    leaf = current.get(path[-1])
    if not isinstance(leaf, list):
        leaf = []
        current[path[-1]] = leaf
    leaf.append(bucket_id)


def build_batter_bucket_history_lookup(planner_eval_view: pd.DataFrame):
    frame = _history_lookup_frame(planner_eval_view)
    lookup: dict[str, object] = {}
    for row in frame.itertuples(index=False):
        path = (
            row.batter_id,
            row.pitcher_handedness,
            row.pitch_1_type,
            row.pitch_1_bucket,
            row.pitch_2_type,
            row.count_bucket,
        )
        _nested_append_bucket(lookup, path, row.pitch_2_bucket)
    return lookup


def build_pitcher_bucket_history_lookup(planner_eval_view: pd.DataFrame):
    frame = _history_lookup_frame(planner_eval_view)
    lookup: dict[str, object] = {}
    for row in frame.itertuples(index=False):
        path = (
            row.pitcher_id,
            row.batter_handedness,
            row.pitch_1_type,
            row.pitch_1_bucket,
            row.pitch_2_type,
            row.count_bucket,
        )
        _nested_append_bucket(lookup, path, row.pitch_2_bucket)
    return lookup


def save_bucket_history_lookup(path: Path, lookup: dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lookup, indent=2, sort_keys=True))


def _top_bucket_locations(bucket_history: list[str], max_locations: int = num_buckets):
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for idx, bucket_id in enumerate(bucket_history):
        normalized = str(bucket_id)
        counts[normalized] += 1
        if normalized not in first_seen:
            first_seen[normalized] = idx
    ranked = sorted(
        counts.items(),
        key=lambda item: (-item[1], first_seen[item[0]], item[0]),
    )
    return [bucket_id for bucket_id, _ in ranked[:max_locations]]


def refine_bucket_history_lookup(
    lookup: dict[str, object],
    max_locations: int = num_buckets):

    refined: dict[str, object] = {}
    for key, value in lookup.items():
        if isinstance(value, dict):
            refined[key] = refine_bucket_history_lookup(value, max_locations=max_locations)
        elif isinstance(value, list):
            refined[key] = _top_bucket_locations(
                [str(bucket_id) for bucket_id in value],
                max_locations=max_locations,
            )
        else:
            refined[key] = value
    return refined


def write_tables(tables: V2Tables):
    table.mkdir(parents=True, exist_ok=True)
    tables.pitch2_states.to_parquet(pitch2_states, index=False)
    tables.pitcher_arsenal_profiles.to_parquet(pitcher_arsenal_profile, index=False)
    tables.pitch_target_distributions.to_parquet(pitch_target_distributions_path, index=False)
    tables.pitch2_observed_actions.to_parquet(pitch2_observed_action, index=False)
    tables.pitch2_outcomes.to_parquet(pitch2_outcome, index=False)
    tables.pitch2_event_tree_view.to_parquet(pitch2_event_tree, index=False)
    tables.pitch2_planner_eval_view.to_parquet(pitch2_plavver_eval_path, index=False)
    batter_history_lookup = build_batter_bucket_history_lookup(tables.pitch2_planner_eval_view)
    pitcher_history_lookup = build_pitcher_bucket_history_lookup(tables.pitch2_planner_eval_view)
    save_bucket_history_lookup(
        batter_bucket,
        batter_history_lookup,
    )
    save_bucket_history_lookup(
        pitcher_bucket,
        pitcher_history_lookup,
    )
    save_bucket_history_lookup(
        batter_top3_cache_path,
        refine_bucket_history_lookup(batter_history_lookup, max_locations=num_buckets),
    )
    save_bucket_history_lookup(
        pitcher_top3_cache_path,
        refine_bucket_history_lookup(pitcher_history_lookup, max_locations=num_buckets),
    )


def load_tables():
    return V2Tables(
        pitch2_states=pd.read_parquet(pitch2_states),
        pitcher_arsenal_profiles=pd.read_parquet(pitcher_arsenal_profile),
        pitch_target_distributions=pd.read_parquet(pitch_target_distributions_path),
        pitch2_observed_actions=pd.read_parquet(pitch2_observed_action),
        pitch2_outcomes=pd.read_parquet(pitch2_outcome),
        pitch2_event_tree_view=pd.read_parquet(pitch2_event_tree),
        pitch2_planner_eval_view=pd.read_parquet(pitch2_plavver_eval_path),
    )


def main():
    args = build_argument_parser().parse_args()
    tables = build_v2_tables(args.source)
    write_tables(tables)
    print(f"Wrote v2 tables to {table}")


if __name__ == "__main__":
    main()
