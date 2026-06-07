from __future__ import annotations
import pickle
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    package = Path(__file__).resolve().parents[2]
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    from models.p1.config import (
        default_x,
        default_z,
        event_tree_model_path,
        planner_metadata,
        random_seed,
    )
    from models.p1.data import P1Tables
    from models.v2.modeling import (
        MulticlassModelWrapper,
        _called_strike_eval_frame,
        _progress,
        _safe_auc,
        fit_called_strike_surface_model,
    )
else:
    from .config import default_x, default_z, event_tree_model_path, planner_metadata, random_seed
    from .data import P1Tables
    from ..v2.modeling import (
        MulticlassModelWrapper,
        _called_strike_eval_frame,
        _progress,
        _safe_auc,
        fit_called_strike_surface_model,
    )

state_features = [
    "balls_before_p1",
    "strikes_before_p1",
    "outs_when_up",
    "pitch_1_is_fastball",
    "pitch_1_is_breaking",
    "pitch_1_is_offspeed",
    "pitcher_pitch1_swing_prior",
    "pitcher_pitch1_contact_prior",
    "pitcher_pitch1_in_play_prior",
    "batter_pitch1_swing_prior",
    "batter_pitch1_contact_prior",
    "batter_pitch1_in_play_prior",
    "matchup_pitch1_swing_prior",
    "matchup_pitch1_contact_prior",
    "pitcher_recent_swing_rate_25",
    "pitcher_recent_contact_rate_25",
    "pitcher_recent_in_play_rate_25",
    "batter_recent_swing_rate_25",
    "batter_recent_contact_rate_25",
    "batter_recent_in_play_rate_25",
    "matchup_recent_swing_rate_25",
    "matchup_recent_contact_rate_25",
    "matchup_recent_in_play_rate_25",
    "pitcher_recent_pitchtype_usage_25",
    "batter_recent_pitchtype_swing_rate_25",
    "pitcher_recent_exit_velo_allowed_50",
    "pitcher_recent_launch_angle_allowed_50",
    "batter_recent_exit_velo_50",
    "batter_recent_launch_angle_50",
]
categorical_features = [
    "pitcher_id",
    "batter_id",
    "pitch_1_type",
    "pitcher_handedness",
    "batter_handedness",
    "count_bucket",
]
action_features = [
    "intended_target_x",
    "intended_target_z",
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
default_bucket = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


#legacy pickle alias
class BucketModelWrapper(MulticlassModelWrapper):
    "Backward-compatible alias for older serialized pitch1 artifacts."
    "Too lazy to fix, but project works as intended"
    pass

def _pickle_dump(path: Path, payload: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


#time-based split
def _season_split(
    frame: pd.DataFrame,
    *,
    train_end: int = 2023,
    val_season: int = 2024,
    test_season: int = 2025):

    seasons = pd.to_numeric(frame["season"], errors="coerce")
    return (
        frame.loc[seasons <= train_end].copy(),
        frame.loc[seasons == val_season].copy(),
        frame.loc[seasons == test_season].copy(),
    )

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
            season_frame = season_frame.sample(n=max_rows_per_season, random_state=random_seed).copy()
        sampled_parts.append(season_frame)
    combined = pd.concat(sampled_parts, ignore_index=False) if sampled_parts else frame.head(0).copy()
    return combined.sort_values(["season", "state_id"]).reset_index(drop=True)


#pitch-1 take-only branch
def _event_probabilities(frame: pd.DataFrame, event_bundle: dict[str, object]):
    p_called_strike = event_bundle["called_strike_model"].predict_proba(frame)[:, 1]
    n_rows = len(frame)
    zeros = np.zeros(n_rows, dtype=float)
    ones = np.ones(n_rows, dtype=float)
    bucket_probs = np.tile(default_bucket, (n_rows, 1))
    return {
        "p_swing": zeros,
        "p_take": ones,
        "p_called_strike": p_called_strike,
        "p_ball": 1.0 - p_called_strike,
        "p_contact": zeros,
        "p_whiff": zeros,
        "p_in_play": zeros,
        "p_foul": zeros,
        "bucket_probs": bucket_probs,
    }


def predict_event_probabilities(frame: pd.DataFrame, event_bundle: dict[str, object]):
    return _event_probabilities(frame, event_bundle)


def _policy_target_rows_for_row(row: pd.Series, targets: pd.DataFrame):
    row_date = pd.to_datetime(row.get("game_date"), errors="coerce")
    target_dates = pd.to_datetime(targets["snapshot_date"], errors="coerce")
    row_state_num = pd.to_numeric(str(row.get("state_id", "")).replace("state_", ""), errors="coerce")
    target_state_num = pd.to_numeric(targets["state_id"].astype(str).str.replace("state_", "", regex=False), errors="coerce")
    if pd.notna(row_date):
        base_mask = target_dates.lt(row_date) | (target_dates.eq(row_date) & target_state_num.lt(row_state_num))
    else:
        base_mask = target_state_num.lt(row_state_num)

    exact_mask = (
        base_mask
        & targets["pitcher_id"].astype(str).eq(str(row["pitcher_id"]))
        & targets["pitch_type"].astype(str).eq(str(row["pitch_1_type"]))
        & targets["batter_handedness"].astype(str).eq(str(row["batter_handedness"]))
        & targets["count_bucket"].astype(str).eq(str(row["count_bucket"]))
    )
    target_rows = targets.loc[exact_mask].copy()

    if target_rows.empty:
        broad_mask = (
            base_mask
            & targets["pitcher_id"].astype(str).eq(str(row["pitcher_id"]))
            & targets["pitch_type"].astype(str).eq(str(row["pitch_1_type"]))
        )
        target_rows = targets.loc[broad_mask].copy()

    if target_rows.empty:
        return pd.DataFrame()
    target_rows = target_rows.assign(_state_num=target_state_num.loc[target_rows.index].to_numpy())
    latest_state_num = target_rows["_state_num"].max()
    latest_rows = target_rows.loc[target_rows["_state_num"].eq(latest_state_num)].copy()
    return latest_rows.drop(columns="_state_num")


def _candidate_frame_for_row(row: pd.Series, targets: pd.DataFrame):
    target_rows = _policy_target_rows_for_row(row, targets)
    if target_rows.empty:
        grid = [(x, z) for x in default_x for z in default_z]
        target_rows = pd.DataFrame(
            {
                "target_mu_x": [x for x, _ in grid],
                "target_mu_z": [z for _, z in grid],
                "component_id": 0,
                "intent_source": "default_grid",
            }
        )
    else:
        grid: list[tuple[float, float]] = []
        component_ids: list[int] = []
        intent_sources: list[str] = []
        for _, target_row in target_rows.iterrows():
            sx = max(float(target_row.get("target_sigma_x", 0.25)), 0.10)
            sz = max(float(target_row.get("target_sigma_z", 0.25)), 0.10)
            for dx in (-sx, 0.0, sx):
                for dz in (-sz, 0.0, sz):
                    grid.append((float(target_row["target_mu_x"]) + dx, float(target_row["target_mu_z"]) + dz))
                    component_ids.append(int(target_row.get("component_id", 0)))
                    intent_sources.append(str(target_row.get("intent_source", "contextual_policy_grid")))
        target_rows = pd.DataFrame(
            {
                "target_mu_x": [x for x, _ in grid],
                "target_mu_z": [z for _, z in grid],
                "component_id": component_ids,
                "intent_source": intent_sources,
            }
        )

    candidates = pd.DataFrame([row.to_dict()] * len(target_rows))
    candidates["intended_target_x"] = target_rows["target_mu_x"].to_numpy()
    candidates["intended_target_z"] = target_rows["target_mu_z"].to_numpy()
    candidates["intended_target_component_id"] = target_rows["component_id"].to_numpy()
    candidates["intent_source"] = target_rows["intent_source"].to_numpy()
    observed_candidate = pd.DataFrame([row.to_dict()])
    observed_candidate["candidate_kind"] = "observed"
    candidates["candidate_kind"] = "generated"
    candidates = pd.concat([observed_candidate, candidates], ignore_index=True)
    return candidates.drop_duplicates(
        subset=["intended_target_x", "intended_target_z", "intended_target_component_id"],
        keep="first",
    ).reset_index(drop=True)


def planner_expected_pitcher_value(
    frame: pd.DataFrame,
    event_bundle: dict[str, object],
    continuation_scorer: object | None = None):

    del continuation_scorer
    #We assume the hitter is taking, so the actionable score is just the called-strike probability at the chosen target.
    return _event_probabilities(frame, event_bundle)["p_called_strike"]


def _planner_window_diagnostics(
    planner_view: pd.DataFrame,
    targets: pd.DataFrame,
    event_bundle: dict[str, object],
    *,
    max_states: int = 750):

    frame = planner_view.copy()
    if max_states > 0 and len(frame) > max_states:
        frame = frame.sample(n=max_states, random_state=random_seed).sort_values("state_id").copy()
    print(f"Pitch1 planner diagnostics: scoring {len(frame):,} states")
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
            _progress("Pitch1 diagnostics", idx, total, f"state_id={row['state_id']}")
        candidates = _candidate_frame_for_row(row, targets)
        scores = planner_expected_pitcher_value(candidates, event_bundle)
        best_idx = int(np.argmax(scores))
        observed_idx = int(candidates.index[candidates["candidate_kind"].eq("observed")][0])
        records.append(
            {
                "observed_outcome_bucket": row["observed_outcome_bucket"],
                "observed_score": float(scores[observed_idx]),
                "best_candidate_score": float(scores[best_idx]),
                "planner_gain": float(scores[best_idx] - scores[observed_idx]),
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


def _called_strike_metrics(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    label: str):

    model = fit_called_strike_surface_model(train_frame)
    eval_take = _called_strike_eval_frame(eval_frame)
    return {
        "window": label,
        "swing_auc": np.nan,
        "called_strike_auc": (
            _safe_auc(eval_take["called_strike"], model.predict_proba(eval_take)[:, 1]) if len(eval_take) else np.nan
        ),
        "contact_auc": np.nan,
        "in_play_auc": np.nan,
    }


def fit_event_tree_model(
    event_tree_view: pd.DataFrame,
    *,
    train_end: int = 2023,
    val_season: int = 2024,
    test_season: int = 2025):

    print(f"Pitch1 event tree: preparing split train<={train_end}, val={val_season}, test={test_season}")
    train, val, test = _season_split(
        event_tree_view,
        train_end=train_end,
        val_season=val_season,
        test_season=test_season,
    )
    print(f"Pitch1 event tree: rows train={len(train):,}, val={len(val):,}, test={len(test):,}")
    print("Pitch1 event tree: fitting take-only called-strike surface model")
    called_strike_model = fit_called_strike_surface_model(train)

    rolling_metrics: list[dict[str, object]] = []
    rolling_windows = _rolling_origin_windows(event_tree_view)
    for idx, (rolling_train, rolling_eval, label) in enumerate(rolling_windows, start=1):
        _progress("Pitch1 event rolling", idx, len(rolling_windows), label)
        rolling_metrics.append(_called_strike_metrics(rolling_train, rolling_eval, label))

    val_take = _called_strike_eval_frame(val)
    test_take = _called_strike_eval_frame(test)
    metrics = {
        "swing_val_auc": np.nan,
        "swing_test_auc": np.nan,
        "called_strike_val_auc": (
            _safe_auc(val_take["called_strike"], called_strike_model.predict_proba(val_take)[:, 1]) if len(val_take) else np.nan
        ),
        "called_strike_test_auc": (
            _safe_auc(test_take["called_strike"], called_strike_model.predict_proba(test_take)[:, 1]) if len(test_take) else np.nan
        ),
        "contact_val_auc": np.nan,
        "contact_test_auc": np.nan,
        "in_play_val_auc": np.nan,
        "in_play_test_auc": np.nan,
        "rolling_origin": rolling_metrics,
    }
    return {
        "called_strike_model": called_strike_model,
        "metrics": metrics,
        "assumption": "take_only_called_strike_surface",
    }


def save_bundles(event_bundle: dict[str, object], metadata: dict[str, object]):
    _pickle_dump(event_tree_model_path, event_bundle)
    _pickle_dump(planner_metadata, metadata)


def train_from_tables(
    tables: P1Tables,
    *,
    max_rows_per_season: int = 0,
    max_planner_states: int = 750):
    event_tree_view = tables.pitch1_event_tree_view
    if max_rows_per_season > 0:
        print(
            f"Training pitch1 model: using deterministic quick sample with up to {max_rows_per_season:,} rows per season"
        )
        event_tree_view = _sample_rows_per_season(event_tree_view, max_rows_per_season)

    print("Training pitch1 model: fitting take-only called-strike model")
    event_bundle = fit_event_tree_model(event_tree_view)

    metadata = {
        "modeling_mode": "pitch1_called_strike_surface_assume_take",
        "event_metrics": event_bundle["metrics"],
        "planner_metrics": {
            "status": "not_run_take_only_pitch1",
            "reason": "pitch1 training now fits only the called-strike surface used to derive the pitch2 count",
        },
        "feature_spec": {
            "numeric_state_features": state_features,
            "categorical_state_features": categorical_features,
            "action_numeric_features": action_features,
        },
        "continuation_modeling": {
            "pitch2_continuation_source": "not_used_take_only",
            "assumption": "derive_pitch2_count_from_called_strike_probability_only",
        },
        "debug_run_config": {
            "max_rows_per_season": max_rows_per_season,
            "max_planner_states": max_planner_states,
        },
    }
    save_bundles(event_bundle, metadata)
    print("Training pitch1 model: saved bundles and metadata")
    return {
        "event_bundle": event_bundle,
        "metadata": metadata,
        "tables": tables,
    }
