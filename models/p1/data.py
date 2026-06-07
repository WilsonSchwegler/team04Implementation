from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    PACKAGE_ROOT = Path(__file__).resolve().parents[2]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

    from models.p1.config import (
        end,
        start,
        p1_event_tree,
        observed_action,
        p1_outcome,
        p1_planner_eval_view_path,
        pitch1_states,
        target_dist,
        arsenal_profiles,
        source_data,
        table,
    )

    from models.p1.schema import (
        arsenal,
        observed_action,
        outcome,
        state,
        target_dist,
    )
    from models.v2.data import (
        hit_value,
        FB_types,
        BB_types,
        OFF_types,
        default_batter_height_ft,
        min_batter_height_ft,
        max_batter_height_ft,
        called_strike_height_top_ratio,
        called_strike_height_bot_ratio,
        _baseline_reference_frame,
        _derive_batted_ball_type,
        _derive_exit_velo_band,
        _group_shifted_conditional_rate,
        _group_shifted_mean,
        _group_shifted_rate,
        _group_shifted_rolling_conditional_rate,
        _group_shifted_rolling_masked_mean,
        _group_shifted_rolling_rate,
        _group_shifted_rolling_share,
        _group_shifted_std,
        _pitch_family_label,
        _pitch_family_flags,
        _safe_numeric,
        _attach_primary_fastball_reference,
        load_available_pitch_level_history,
    )

else:
    from .config import (
        end,
        start,
        p1_event_tree,
        observed_action,
        p1_outcome,
        p1_planner_eval_view_path,
        pitch1_states,
        target_dist,
        arsenal_profiles,
        source_data,
        table,
    )
    from .schema import (
        arsenal,
        observed_action,
        outcome,
        state,
        target_dist,
    )
    from ..v2.data import (
        hit_value,
        FB_types,
        BB_types,
        OFF_types,
        default_batter_height_ft,
        min_batter_height_ft,
        max_batter_height_ft,
        called_strike_height_top_ratio,
        called_strike_height_bot_ratio,
        _baseline_reference_frame,
        _derive_batted_ball_type,
        _derive_exit_velo_band,
        _group_shifted_conditional_rate,
        _group_shifted_mean,
        _group_shifted_rate,
        _group_shifted_rolling_conditional_rate,
        _group_shifted_rolling_masked_mean,
        _group_shifted_rolling_rate,
        _group_shifted_rolling_share,
        _group_shifted_std,
        _pitch_family_label,
        _pitch_family_flags,
        _safe_numeric,
        _attach_primary_fastball_reference,
        load_available_pitch_level_history,
    )


out_events = {
    "double_play",
    "field_out",
    "fielders_choice_out",
    "force_out",
    "grounded_into_double_play",
    "other_out",
    "runner_double_play",
    "sac_bunt",
    "sac_bunt_double_play",
    "sac_fly",
    "sac_fly_double_play",
    "strikeout_double_play",
    "triple_play",
}

def _select_policy_component_count(prior_count: int) -> int:
    if prior_count >= 40:
        return 3
    if prior_count >= 12:
        return 2
    return 1

def _fit_multicenter_policy_components(
    points: np.ndarray,
    *,
    max_components: int,
    min_sigma: float = 0.10):

    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("points must be a non-empty (n, 2) array")

    k = min(max_components, len(points))
    order = np.argsort(points[:, 0] + 0.5 * points[:, 1])
    init_idx = order[np.linspace(0, len(points) - 1, k, dtype=int)]
    centers = points[init_idx].copy()

    for _ in range(8):
        distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for component_idx in range(k):
            cluster_points = points[labels == component_idx]
            if len(cluster_points):
                new_centers[component_idx] = cluster_points.mean(axis=0)
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(distances, axis=1)
    components: list[dict[str, float]] = []
    for component_idx in range(k):
        cluster_points = points[labels == component_idx]
        if len(cluster_points) == 0:
            continue
        sigma_x = float(np.std(cluster_points[:, 0])) if len(cluster_points) > 1 else min_sigma
        sigma_z = float(np.std(cluster_points[:, 1])) if len(cluster_points) > 1 else min_sigma
        components.append(
            {
                "component_id": int(component_idx),
                "weight": float(len(cluster_points) / len(points)),
                "target_mu_x": float(cluster_points[:, 0].mean()),
                "target_mu_z": float(cluster_points[:, 1].mean()),
                "target_sigma_x": max(sigma_x, min_sigma),
                "target_sigma_z": max(sigma_z, min_sigma),
                "target_rho": 0.0,
                "n_obs": int(len(points)),
            }
        )
    components = sorted(components, key=lambda row: (-row["weight"], row["component_id"]))
    for new_idx, component in enumerate(components):
        component["component_id"] = new_idx
    return components
excluded_des = {"hit_by_pitch"}
excluded_event = {"catcher_interf", "truncated_pa", "walk"}


@dataclass(frozen=True)
class P1Tables:
    pitch1_states: pd.DataFrame
    pitcher_arsenal_profiles: pd.DataFrame
    pitch_target_distributions: pd.DataFrame
    pitch1_observed_actions: pd.DataFrame
    pitch1_outcomes: pd.DataFrame
    pitch1_event_tree_view: pd.DataFrame
    pitch1_planner_eval_view: pd.DataFrame

def build_argument_parser():
    parser = argparse.ArgumentParser(description="Build normalized pitch-1 research tables.")
    parser.add_argument("--start-season", type=int, default=start)
    parser.add_argument("--end-season", type=int, default=end)
    return parser

def _resolve_matchup_batter_hand(batter_hand: str | None, pitcher_hand: str | None):
    batter = str(batter_hand or "").upper()
    pitcher = str(pitcher_hand or "").upper()
    if batter in {"L", "R"}:
        return batter
    if batter == "S":
        if pitcher == "R":
            return "L"
        if pitcher == "L":
            return "R"
    return None


def _load_name_lookups():
    if not source_data.exists():
        empty_pitchers = pd.DataFrame(columns=["pitcher_id", "pitcher_name", "pitcher_team", "pitcher_handedness"])
        empty_batters = pd.DataFrame(columns=["batter_id", "batter_name", "batter_team", "batter_handedness"])
        return empty_pitchers, empty_batters

    frame = pd.read_csv(
        source_data,
        usecols=[
            "game_date",
            "pitcher_id",
            "pitcher_name",
            "pitcher_team",
            "pitcher_handedness",
            "batter_id",
            "batter_name",
            "batter_team",
            "batter_handedness",
        ],
    )
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    pitchers = (
        frame.sort_values("game_date")
        .drop_duplicates("pitcher_id", keep="last")[
            ["pitcher_id", "pitcher_name", "pitcher_team", "pitcher_handedness"]
        ]
        .copy()
    )
    batters = (
        frame.sort_values("game_date")
        .drop_duplicates("batter_id", keep="last")[
            ["batter_id", "batter_name", "batter_team", "batter_handedness"]
        ]
        .copy()
    )
    return pitchers, batters


def _canonical_pitch_level_frame(frame: pd.DataFrame):
    working = frame.rename(
        columns={
            "Season": "season",
            "Name": "pitcher_name",
            "mlbam_id": "pitcher_id",
            "team": "pitcher_team",
            "pitcher_hand": "pitcher_handedness",
            "batter_hand": "raw_batter_handedness",
            "balls_before_pitch": "balls_before_pitch",
            "strikes_before_pitch": "strikes_before_pitch",
            "outs_when_up": "outs_when_up",
            "launch_speed": "exit_velo",
        }
    ).copy()

    working["season"] = pd.to_numeric(working["season"], errors="coerce").astype("Int64")
    working["game_date"] = pd.to_datetime(working["game_date"], errors="coerce")
    working["game_pk"] = pd.to_numeric(working["game_pk"], errors="coerce").astype("Int64")
    working["at_bat_number"] = pd.to_numeric(working["at_bat_number"], errors="coerce").astype("Int64")
    working["pitch_number"] = pd.to_numeric(working["pitch_number"], errors="coerce").astype("Int64")
    working["pitcher_id"] = pd.to_numeric(working["pitcher_id"], errors="coerce").astype("Int64")
    working["batter_id"] = pd.to_numeric(working["batter"], errors="coerce").astype("Int64")
    working["pitcher_handedness"] = working["pitcher_handedness"].astype("string").str.upper()
    working["raw_batter_handedness"] = working["raw_batter_handedness"].astype("string").str.upper()
    working["batter_handedness"] = [
        _resolve_matchup_batter_hand(batter_hand, pitcher_hand)
        for batter_hand, pitcher_hand in zip(
            working["raw_batter_handedness"].tolist(),
            working["pitcher_handedness"].tolist(),
        )
    ]
    working["batter_handedness"] = pd.Series(working["batter_handedness"], index=working.index, dtype="string")
    return working


def _attach_entity_names(frame: pd.DataFrame):
    pitchers, batters = _load_name_lookups()
    enriched = frame.copy()
    if len(pitchers):
        enriched = enriched.merge(
            pitchers,
            on="pitcher_id",
            how="left",
            suffixes=("", "_lookup"),
        )
        for column in ["pitcher_name", "pitcher_team", "pitcher_handedness"]:
            lookup_column = f"{column}_lookup"
            if lookup_column in enriched.columns:
                enriched[column] = enriched[column].fillna(enriched[lookup_column])
                enriched = enriched.drop(columns=lookup_column)
    if len(batters):
        enriched = enriched.merge(
            batters,
            on="batter_id",
            how="left",
            suffixes=("", "_lookup"),
        )
        for column in ["batter_name", "batter_team", "batter_handedness"]:
            lookup_column = f"{column}_lookup"
            if lookup_column in enriched.columns:
                enriched[column] = enriched[column].fillna(enriched[lookup_column])
                enriched = enriched.drop(columns=lookup_column)

    enriched["pitcher_name"] = enriched["pitcher_name"].fillna(
        enriched["pitcher_id"].astype("Int64").astype(str).radd("pitcher_")
    )
    enriched["batter_name"] = enriched["batter_name"].fillna(
        enriched["batter_id"].astype("Int64").astype(str).radd("batter_")
    )
    return enriched

def _attach_batter_height(frame: pd.DataFrame):
    enriched = frame.copy()
    top_height = _safe_numeric(enriched["sz_top"]) / called_strike_height_top_ratio
    bot_height = _safe_numeric(enriched["sz_bot"]) / called_strike_height_bot_ratio
    row_height = pd.concat([top_height, bot_height], axis=1).mean(axis=1, skipna=True)
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
    return enriched


def load_source_dataset(
    *,
    start_season: int = start,
    end_season: int = end):
    history = load_available_pitch_level_history()
    if history.empty:
        raise ValueError("No pitch-level history found in data/pitch_level_*.csv.")

    frame = _canonical_pitch_level_frame(history)
    frame = frame.loc[
        frame["season"].between(start_season, end_season, inclusive="both")
        & frame["pitch_number"].eq(1)
        & frame["pitch_type"].notna()
        & frame["pitcher_id"].notna()
        & frame["batter_id"].notna()
    ].copy()
    frame = frame.loc[frame["batter_handedness"].isin(["L", "R"])].copy()
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["events"] = frame["events"].fillna("").astype(str)
    frame = frame.loc[
        ~frame["description"].isin(excluded_des)
        & ~frame["events"].isin(excluded_event)
    ].copy()
    frame["_source_row_order"] = np.arange(len(frame), dtype=int)
    frame = frame.sort_values(
        ["season", "game_date", "game_pk", "at_bat_number", "_source_row_order"],
        kind="mergesort",
    ).reset_index(drop=True)
    frame = _attach_entity_names(frame)
    frame = _attach_batter_height(frame)

    frame["state_id"] = "state_" + frame.index.astype(int).astype(str).str.zfill(7)
    frame["balls_before_p1"] = 0
    frame["strikes_before_p1"] = 0
    frame["count_bucket"] = "0-0"
    frame["pitch_1_type"] = frame["pitch_type"].fillna("Unknown").astype(str).str.upper()

    frame["take"] = 1 - _safe_numeric(frame["is_swing"]).fillna(0).astype(int)
    frame["swing"] = _safe_numeric(frame["is_swing"]).fillna(0).astype(int)
    frame["called_strike"] = _safe_numeric(frame["is_called_strike"]).fillna(0).astype(int)
    frame["ball"] = _safe_numeric(frame["is_ball"]).fillna(0).astype(int)
    frame["whiff"] = _safe_numeric(frame["is_whiff"]).fillna(0).astype(int)
    frame["contact"] = _safe_numeric(frame["is_contact"]).fillna(0).astype(int)
    frame["in_play"] = _safe_numeric(frame["is_in_play"]).fillna(0).astype(int)
    frame["foul"] = np.where(frame["contact"].eq(1) & frame["in_play"].eq(0), 1, 0)

    valid_take = frame["take"].eq(1) & frame["called_strike"].add(frame["ball"]).eq(1)
    valid_swing = (
        frame["swing"].eq(1)
        & frame["whiff"].add(frame["contact"]).eq(1)
        & (
            frame["contact"].eq(0)
            | frame["foul"].add(frame["in_play"]).eq(1)
        )
    )
    frame = frame.loc[valid_take | valid_swing].copy()

    events = frame["events"].astype(str).str.lower()
    frame["single"] = events.eq("single").astype(int)
    frame["double"] = events.eq("double").astype(int)
    frame["triple"] = events.eq("triple").astype(int)
    frame["home_run"] = events.eq("home_run").astype(int)
    frame["double_or_triple"] = frame["double"] + frame["triple"]
    frame["out"] = np.where(
        frame["in_play"].eq(1)
        & frame["single"].add(frame["double_or_triple"]).add(frame["home_run"]).eq(0),
        1,
        0,
    )
    frame["out"] = np.where(events.isin(out_events), 1, frame["out"])

    p1_fastball, p1_breaking, p1_offspeed = _pitch_family_flags(frame["pitch_1_type"])
    frame = _attach_primary_fastball_reference(
        frame,
        pitch_type_column="pitch_1_type",
        velo_column="velo",
        h_mov_column="h_mov",
        v_mov_column="v_mov",
        prefix="pitch_1",
    )
    frame["pitch_1_family"] = _pitch_family_label(frame["pitch_1_type"])
    frame["pitch_1_is_fastball"] = p1_fastball
    frame["pitch_1_is_breaking"] = p1_breaking
    frame["pitch_1_is_offspeed"] = p1_offspeed

    frame["batter_value_on_contact"] = (
        frame["out"] * hit_value["out"]
        + frame["single"] * hit_value["single"]
        + frame["double_or_triple"] * hit_value["double_or_triple"]
        + frame["home_run"] * hit_value["home_run"]
    )
    frame["pitcher_value_on_contact"] = -frame["batter_value_on_contact"]
    return frame.drop(columns="_source_row_order")


def add_state_priors(frame: pd.DataFrame):
    enriched = frame.copy()
    baseline = _baseline_reference_frame(enriched)
    global_swing = float(_safe_numeric(baseline["swing"]).mean())
    global_contact = float(_safe_numeric(baseline.loc[baseline["swing"].eq(1), "contact"]).mean())
    global_in_play = float(_safe_numeric(baseline.loc[baseline["contact"].eq(1), "in_play"]).mean())
    global_exit_velo = float(_safe_numeric(baseline.loc[baseline["in_play"].eq(1), "exit_velo"]).mean())
    global_launch_angle = float(_safe_numeric(baseline.loc[baseline["in_play"].eq(1), "launch_angle"]).mean())

    enriched["pitcher_pitch1_swing_prior"] = _group_shifted_rate(
        enriched,
        ["pitcher_id", "pitch_1_type"],
        "swing",
        prior_mean=global_swing,
    )
    enriched["pitcher_pitch1_contact_prior"] = _group_shifted_conditional_rate(
        enriched,
        ["pitcher_id", "pitch_1_type"],
        "contact",
        "swing",
        prior_mean=global_contact,
    )
    enriched["pitcher_pitch1_in_play_prior"] = _group_shifted_conditional_rate(
        enriched,
        ["pitcher_id", "pitch_1_type"],
        "in_play",
        "contact",
        prior_mean=global_in_play,
    )
    enriched["batter_pitch1_swing_prior"] = _group_shifted_rate(
        enriched,
        ["batter_id", "pitch_1_type"],
        "swing",
        prior_mean=global_swing,
    )
    enriched["batter_pitch1_contact_prior"] = _group_shifted_conditional_rate(
        enriched,
        ["batter_id", "pitch_1_type"],
        "contact",
        "swing",
        prior_mean=global_contact,
    )
    enriched["batter_pitch1_in_play_prior"] = _group_shifted_conditional_rate(
        enriched,
        ["batter_id", "pitch_1_type"],
        "in_play",
        "contact",
        prior_mean=global_in_play,
    )
    enriched["matchup_pitch1_swing_prior"] = _group_shifted_rate(
        enriched,
        ["pitcher_id", "batter_id", "pitch_1_type"],
        "swing",
        prior_mean=global_swing,
    )
    enriched["matchup_pitch1_contact_prior"] = _group_shifted_conditional_rate(
        enriched,
        ["pitcher_id", "batter_id", "pitch_1_type"],
        "contact",
        "swing",
        prior_mean=global_contact,
    )

    enriched["pitcher_recent_swing_rate_25"] = _group_shifted_rolling_rate(
        enriched, ["pitcher_id"], "swing", 25, global_swing
    )
    enriched["pitcher_recent_contact_rate_25"] = _group_shifted_rolling_conditional_rate(
        enriched, ["pitcher_id"], "contact", "swing", 25, global_contact
    )
    enriched["pitcher_recent_in_play_rate_25"] = _group_shifted_rolling_conditional_rate(
        enriched, ["pitcher_id"], "in_play", "contact", 25, global_in_play
    )
    enriched["batter_recent_swing_rate_25"] = _group_shifted_rolling_rate(
        enriched, ["batter_id"], "swing", 25, global_swing
    )
    enriched["batter_recent_contact_rate_25"] = _group_shifted_rolling_conditional_rate(
        enriched, ["batter_id"], "contact", "swing", 25, global_contact
    )
    enriched["batter_recent_in_play_rate_25"] = _group_shifted_rolling_conditional_rate(
        enriched, ["batter_id"], "in_play", "contact", 25, global_in_play
    )
    enriched["matchup_recent_swing_rate_25"] = _group_shifted_rolling_rate(
        enriched, ["pitcher_id", "batter_id"], "swing", 25, global_swing
    )
    enriched["matchup_recent_contact_rate_25"] = _group_shifted_rolling_conditional_rate(
        enriched, ["pitcher_id", "batter_id"], "contact", "swing", 25, global_contact
    )
    enriched["matchup_recent_in_play_rate_25"] = _group_shifted_rolling_conditional_rate(
        enriched, ["pitcher_id", "batter_id"], "in_play", "contact", 25, global_in_play
    )
    pitch_type_global_usage = (
        enriched["pitch_1_type"].fillna("Unknown").astype(str).value_counts(normalize=True).mean()
    )
    enriched["pitcher_recent_pitchtype_usage_25"] = _group_shifted_rolling_share(
        enriched,
        ["pitcher_id", "pitch_1_type"],
        ["pitcher_id"],
        25,
        float(pitch_type_global_usage),
    )
    enriched["batter_recent_pitchtype_swing_rate_25"] = _group_shifted_rolling_rate(
        enriched, ["batter_id", "pitch_1_type"], "swing", 25, global_swing
    )
    enriched["pitcher_recent_exit_velo_allowed_50"] = _group_shifted_rolling_masked_mean(
        enriched, ["pitcher_id"], "exit_velo", "in_play", 50, global_exit_velo
    )
    enriched["pitcher_recent_launch_angle_allowed_50"] = _group_shifted_rolling_masked_mean(
        enriched, ["pitcher_id"], "launch_angle", "in_play", 50, global_launch_angle
    )
    enriched["batter_recent_exit_velo_50"] = _group_shifted_rolling_masked_mean(
        enriched, ["batter_id"], "exit_velo", "in_play", 50, global_exit_velo
    )
    enriched["batter_recent_launch_angle_50"] = _group_shifted_rolling_masked_mean(
        enriched, ["batter_id"], "launch_angle", "in_play", 50, global_launch_angle
    )
    return enriched

def build_pitch1_states(frame: pd.DataFrame):
    base = add_state_priors(frame)
    states = base.copy()
    states["called_strike_bucket"] = "0-1"
    states["ball_bucket"] = "1-0"
    states["whiff_bucket"] = "0-1"
    states["foul_bucket"] = "0-1"
    return states.loc[:, state]


def build_pitcher_arsenal_profiles(frame: pd.DataFrame):
    working = _attach_primary_fastball_reference(
        frame,
        pitch_type_column="pitch_1_type",
        velo_column="velo",
        h_mov_column="h_mov",
        v_mov_column="v_mov",
        prefix="pitch_1",
    )
    profiles = working[["state_id", "pitcher_id", "pitch_1_type", "game_date", "season"]].copy()
    profiles = profiles.rename(columns={"pitch_1_type": "pitch_type", "game_date": "snapshot_date"})

    baseline = _baseline_reference_frame(working)
    global_usage = (
        baseline["pitch_1_type"].fillna("Unknown").astype(str).value_counts(normalize=True).to_dict()
    )
    total_prior_pitcher = working.groupby("pitcher_id", dropna=False).cumcount()
    prior_type_count = working.groupby(["pitcher_id", "pitch_1_type"], dropna=False).cumcount()
    pitch_type_support = max(int(working["pitch_1_type"].fillna("Unknown").astype(str).nunique()), 1)
    prior_mean_usage = working["pitch_1_type"].fillna("Unknown").astype(str).map(global_usage).fillna(1.0 / pitch_type_support)
    profiles["usage_rate"] = (prior_type_count + prior_mean_usage) / (total_prior_pitcher + 1.0)

    defaults = {
        "velo": float(_safe_numeric(baseline["velo"]).mean()),
        "h_mov": float(_safe_numeric(baseline["h_mov"]).mean()),
        "v_mov": float(_safe_numeric(baseline["v_mov"]).mean()),
        "spin_rate": float(_safe_numeric(baseline["spin_rate"]).mean()),
        "extension": float(_safe_numeric(baseline["extension"]).mean()),
        "release_x": float(_safe_numeric(baseline["release_x"]).mean()),
        "release_z": float(_safe_numeric(baseline["release_z"]).mean()),
        "plate_x": float(_safe_numeric(baseline["plate_x"]).mean()),
        "plate_z": float(_safe_numeric(baseline["plate_z"]).mean()),
    }
    group_columns = ["pitcher_id", "pitch_1_type"]
    profiles["avg_velo"] = _group_shifted_mean(working, group_columns, "velo", defaults["velo"])
    profiles["avg_h_mov"] = _group_shifted_mean(working, group_columns, "h_mov", defaults["h_mov"])
    profiles["avg_v_mov"] = _group_shifted_mean(working, group_columns, "v_mov", defaults["v_mov"])
    profiles["avg_spin_rate"] = _group_shifted_mean(working, group_columns, "spin_rate", defaults["spin_rate"])
    profiles["avg_extension"] = _group_shifted_mean(working, group_columns, "extension", defaults["extension"])
    profiles["avg_release_x"] = _group_shifted_mean(working, group_columns, "release_x", defaults["release_x"])
    profiles["avg_release_z"] = _group_shifted_mean(working, group_columns, "release_z", defaults["release_z"])
    profiles["avg_plate_x"] = _group_shifted_mean(working, group_columns, "plate_x", defaults["plate_x"])
    profiles["avg_plate_z"] = _group_shifted_mean(working, group_columns, "plate_z", defaults["plate_z"])
    profiles["delta_velo_vs_primary_fastball"] = profiles["avg_velo"] - working["pitch_1_primary_fastball_velo"].to_numpy(dtype=float)
    profiles["delta_h_mov_vs_primary_fastball"] = profiles["avg_h_mov"] - working["pitch_1_primary_fastball_h_mov"].to_numpy(dtype=float)
    profiles["delta_v_mov_vs_primary_fastball"] = profiles["avg_v_mov"] - working["pitch_1_primary_fastball_v_mov"].to_numpy(dtype=float)
    profiles["n_pitches"] = prior_type_count
    return profiles.loc[:, arsenal]

def build_pitch_target_distributions(frame: pd.DataFrame):
    target_group_columns = ["pitcher_id", "pitch_1_type", "batter_handedness", "count_bucket"]
    broad_group_columns = ["pitcher_id", "pitch_1_type"]
    baseline = _baseline_reference_frame(frame)
    global_mu_x = float(_safe_numeric(baseline["plate_x"]).mean())
    global_mu_z = float(_safe_numeric(baseline["plate_z"]).mean())
    broad_mu_x = _group_shifted_mean(frame, broad_group_columns, "plate_x", global_mu_x)
    broad_mu_z = _group_shifted_mean(frame, broad_group_columns, "plate_z", global_mu_z)
    broad_sigma_x = _group_shifted_std(frame, broad_group_columns, "plate_x", 0.35).clip(lower=0.10)
    broad_sigma_z = _group_shifted_std(frame, broad_group_columns, "plate_z", 0.35).clip(lower=0.10)

    records: list[dict[str, object]] = []
    for _, group in frame.groupby(target_group_columns, sort=False, dropna=False):
        working = group.sort_values(["game_date", "state_id"]).copy()
        context_keys = list(
            zip(
                working["pitcher_id"],
                working["pitch_1_type"],
                working["batter_handedness"],
                working["count_bucket"],
            )
        )
        realized_points = working[["plate_x", "plate_z"]].to_numpy(dtype=float)
        prior_points_by_context: dict[tuple[object, ...], list[np.ndarray]] = {}
        for row_idx, (source_idx, row) in enumerate(working.iterrows()):
            state_key = context_keys[row_idx]
            prior_points = prior_points_by_context.get(state_key, [])
            if prior_points:
                prior_array = np.vstack(prior_points[-150:])
                components = _fit_multicenter_policy_components(
                    prior_array,
                    max_components=_select_policy_component_count(len(prior_array)),
                )
                intent_source = "point_in_time_multicenter_policy"
            else:
                components = [
                    {
                        "component_id": 0,
                        "weight": 1.0,
                        "target_mu_x": float(broad_mu_x.loc[source_idx]),
                        "target_mu_z": float(broad_mu_z.loc[source_idx]),
                        "target_sigma_x": float(broad_sigma_x.loc[source_idx]),
                        "target_sigma_z": float(broad_sigma_z.loc[source_idx]),
                        "target_rho": 0.0,
                        "n_obs": 0,
                    }
                ]
                intent_source = "broad_pitchtype_fallback"

            for component in components:
                records.append(
                    {
                        "state_id": row["state_id"],
                        "pitcher_id": row["pitcher_id"],
                        "pitch_type": row["pitch_1_type"],
                        "batter_handedness": row["batter_handedness"],
                        "count_bucket": row["count_bucket"],
                        "snapshot_date": row["game_date"],
                        "component_id": component["component_id"],
                        "weight": component["weight"],
                        "target_mu_x": component["target_mu_x"],
                        "target_mu_z": component["target_mu_z"],
                        "target_sigma_x": component["target_sigma_x"],
                        "target_sigma_z": component["target_sigma_z"],
                        "target_rho": component["target_rho"],
                        "n_obs": component["n_obs"],
                        "intent_source": intent_source,
                    }
                )
            prior_points_by_context.setdefault(state_key, []).append(realized_points[row_idx])

    grouped = pd.DataFrame.from_records(records)
    return grouped.loc[:, target_dist]


def build_pitch1_observed_actions(frame: pd.DataFrame, target_distributions: pd.DataFrame):
    intent_lookup = (
        target_distributions.sort_values(["state_id", "weight", "component_id"], ascending=[True, False, True])
        .drop_duplicates(subset=["state_id"], keep="first")
    )[["state_id", "target_mu_x", "target_mu_z", "component_id", "intent_source"]]
    actions = frame.merge(intent_lookup, on="state_id", how="left", validate="one_to_one")
    actions["intended_target_x"] = actions["target_mu_x"].fillna(actions["plate_x"])
    actions["intended_target_z"] = actions["target_mu_z"].fillna(actions["plate_z"])
    actions["intended_target_component_id"] = actions["component_id"].fillna(0).astype(int)
    actions["intent_source"] = actions["intent_source"].fillna("realized_location_fallback")
    actions["observed_plate_x"] = _safe_numeric(actions["plate_x"])
    actions["observed_plate_z"] = _safe_numeric(actions["plate_z"])
    return actions.loc[:, observed_action]


def build_pitch1_outcomes(frame: pd.DataFrame):
    outcomes = pd.DataFrame(
        {
            "state_id": frame["state_id"],
            "take": frame["take"],
            "swing": frame["swing"],
            "called_strike": frame["called_strike"],
            "ball": frame["ball"],
            "whiff": frame["whiff"],
            "contact": frame["contact"],
            "foul": frame["foul"],
            "in_play": frame["in_play"],
            "exit_velo": frame["exit_velo"],
            "launch_angle": frame["launch_angle"],
            "batted_ball_type": _derive_batted_ball_type(frame["launch_angle"]),
            "ev_band": _derive_exit_velo_band(frame["exit_velo"]),
            "single": frame["single"],
            "double": frame["double"],
            "triple": frame["triple"],
            "double_or_triple": frame["double_or_triple"],
            "home_run": frame["home_run"],
            "out": frame["out"],
            "batter_value_on_contact": frame["batter_value_on_contact"],
            "pitcher_value_on_contact": frame["pitcher_value_on_contact"],
        }
    )
    return outcomes.loc[:, outcome]


def build_views(
    states: pd.DataFrame,
    observed_actions: pd.DataFrame,
    outcomes: pd.DataFrame,
    arsenal_profiles: pd.DataFrame):

    arsenal = arsenal_profiles.rename(columns={"pitch_type": "pitch_1_type"})
    event_tree = (
        states.merge(observed_actions, on=["state_id", "pitch_1_type"], how="inner", validate="one_to_one")
        .merge(arsenal, on=["state_id", "pitcher_id", "pitch_1_type", "season"], how="left", validate="one_to_one")
        .merge(outcomes, on="state_id", how="inner")
    )
    planner_eval = event_tree.copy()
    planner_eval["observed_outcome_bucket"] = np.select(
        [
            planner_eval["home_run"].eq(1),
            planner_eval["double_or_triple"].eq(1),
            planner_eval["single"].eq(1),
        ],
        ["home_run", "double_or_triple", "single"],
        default="out",
    )
    return event_tree, planner_eval


def build_p1_tables(
    *,
    start_season: int = start,
    end_season: int = end):

    base = load_source_dataset(start_season=start_season, end_season=end_season)
    states = build_pitch1_states(base)
    arsenal_profiles = build_pitcher_arsenal_profiles(base)
    target_distributions = build_pitch_target_distributions(base)
    observed_actions = build_pitch1_observed_actions(base, target_distributions)
    outcomes = build_pitch1_outcomes(base)
    event_tree_view, planner_eval_view = build_views(states, observed_actions, outcomes, arsenal_profiles)
    return P1Tables(
        pitch1_states=states,
        pitcher_arsenal_profiles=arsenal_profiles,
        pitch_target_distributions=target_distributions,
        pitch1_observed_actions=observed_actions,
        pitch1_outcomes=outcomes,
        pitch1_event_tree_view=event_tree_view,
        pitch1_planner_eval_view=planner_eval_view,
    )


def write_tables(tables: P1Tables):
    table.mkdir(parents=True, exist_ok=True)
    tables.pitch1_states.to_parquet(pitch1_states, index=False)
    tables.pitcher_arsenal_profiles.to_parquet(arsenal_profiles, index=False)
    tables.pitch_target_distributions.to_parquet(target_dist, index=False)
    tables.pitch1_observed_actions.to_parquet(observed_action, index=False)
    tables.pitch1_outcomes.to_parquet(p1_outcome, index=False)
    tables.pitch1_event_tree_view.to_parquet(p1_event_tree, index=False)
    tables.pitch1_planner_eval_view.to_parquet(p1_planner_eval_view_path, index=False)


def load_tables():
    return P1Tables(
        pitch1_states=pd.read_parquet(pitch1_states),
        pitcher_arsenal_profiles=pd.read_parquet(arsenal_profiles),
        pitch_target_distributions=pd.read_parquet(target_dist),
        pitch1_observed_actions=pd.read_parquet(observed_action),
        pitch1_outcomes=pd.read_parquet(p1_outcome),
        pitch1_event_tree_view=pd.read_parquet(p1_event_tree),
        pitch1_planner_eval_view=pd.read_parquet(p1_planner_eval_view_path),
    )


def main():
    args = build_argument_parser().parse_args()
    tables = build_p1_tables(start_season=args.start_season, end_season=args.end_season)
    write_tables(tables)
    print(f"Wrote pitch-1 tables to {table}")


if __name__ == "__main__":
    main()
