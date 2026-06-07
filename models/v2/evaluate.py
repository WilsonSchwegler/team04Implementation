from __future__ import annotations
import argparse
from functools import lru_cache
import json
import pickle
from pathlib import Path
import sys
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    package = Path(__file__).resolve().parents[2]
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    from models.v2.config import (
        batter_top3_cache_path,
        eval_report,
        event_tree_evaluator,
        event_tree_model_path,
        planner_eval_metadata,
        planner_metadata_path,
        pitch2_plavver_eval_path,
        pitch_target_distributions_path,
        pitcher_top3_cache_path,
        random_seed,
    )
    from models.v2.cache_helpers import (
        append_cached_bucket_records,
        assign_crossfit_fold,
        batter_top3_cache_path_for_row,
        load_top3_cache,
        nested_bucket_lookup,
        pitcher_top3_cache_path_for_row,
    )
    from models.v2.location_grid import bucket_center, locate_bucket_id
    from models.v2.modeling import (
        batter_specific_target_scope,
        broad_target_scope,
        default_target_scope,
        handidness_target_scope,
        _candidate_frame_from_target_rows,
        _default_target_rows,
        _generated_candidate_frame_from_row,
        _generated_candidate_rows_from_target_rows,
        _select_pool_target_rows,
        build_target_lookup_indexes,
        planner_expected_pitcher_value,
        prepare_target_lookup_frame,
    )
else:
    from .config import (
        batter_top3_cache_path,
        eval_report,
        event_tree_evaluator,
        event_tree_model_path,
        planner_eval_metadata,
        planner_metadata_path,
        pitch2_plavver_eval_path,
        pitch_target_distributions_path,
        pitcher_top3_cache_path,
        random_seed,
    )
    from .cache_helpers import (
        append_cached_bucket_records,
        assign_crossfit_fold,
        batter_top3_cache_path_for_row,
        load_top3_cache,
        nested_bucket_lookup,
        pitcher_top3_cache_path_for_row,
    )
    from .location_grid import bucket_center, locate_bucket_id
    from .modeling import (
        batter_specific_target_scope,
        broad_target_scope,
        default_target_scope,
        handidness_target_scope,
        _candidate_frame_from_target_rows,
        _default_target_rows,
        _generated_candidate_frame_from_row,
        _generated_candidate_rows_from_target_rows,
        _select_pool_target_rows,
        build_target_lookup_indexes,
        planner_expected_pitcher_value,
        prepare_target_lookup_frame,
    )


OUTCOME_SAMPLE_ORDER = (
    "out",
    "single",
    "double_or_triple",
    "home_run",
)

def _pickle_load(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _traceability_metadata():
    metadata: dict[str, object] = {}
    if Path(planner_metadata_path).exists():
        loaded = _pickle_load(planner_metadata_path)
        if isinstance(loaded, dict):
            metadata = loaded
    evaluator_metadata: dict[str, object] = {}
    if Path(planner_eval_metadata).exists():
        loaded = _pickle_load(planner_eval_metadata)
        if isinstance(loaded, dict):
            evaluator_metadata = loaded
    event_bundle = _pickle_load(event_tree_evaluator if Path(event_tree_evaluator).exists() else event_tree_model_path)
    event_premium_lookup = event_bundle.get("event_premium_lookup", {})
    event_premium_lookup_keys = {
        str(event_kind): sorted(str(key) for key in event_lookup.keys())
        for event_kind, event_lookup in event_premium_lookup.items()
        if isinstance(event_lookup, dict)
    }
    research_components = metadata.get("research_objective_components") or {
        "evaluation_bundle_source": "saved_research_event_bundle",
        "called_strike_runtime_source": "saved_research_called_strike_model",
        "called_strike_model_class": type(event_bundle.get("called_strike_model")).__name__,
        "continuation_value_source": "saved_training_split_count_state_lookup",
        "continuation_lookup_keys": sorted(str(key) for key in event_bundle.get("count_state_lookup", {}).keys()),
        "event_premium_source": "saved_training_split_pitch2_event_premium_lookup",
        "event_premium_lookup_keys": event_premium_lookup_keys,
        "planner_value_path": "planner_expected_pitcher_value_using_saved_event_bundle",
        "offline_evaluation_contract": (
            "models/v2/evaluate.py scores the saved research event bundle directly and does not apply "
            "deployment-only runtime overrides."
        ),
    }
    deployment_overrides = metadata.get("deployment_objective_overrides") or {
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
        "contract_version": metadata.get("contract_version", "v2_traceability_v2"),
        "research_objective_components": research_components,
        "deployment_objective_overrides": deployment_overrides,
        "cross_fit_contract": evaluator_metadata.get("cross_fit_contract") or metadata.get("cross_fit_contract") or {},
    }


def _planner_metadata():
    if not Path(planner_metadata_path).exists():
        return {}
    loaded = _pickle_load(planner_metadata_path)
    return loaded if isinstance(loaded, dict) else {}


def _planner_evaluator_metadata():
    if not Path(planner_eval_metadata).exists():
        return {}
    loaded = _pickle_load(planner_eval_metadata)
    return loaded if isinstance(loaded, dict) else {}


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Evaluate the v2 pitch planner.")
    parser.add_argument(
        "--eval-view",
        default=str(pitch2_plavver_eval_path),
        help="Planner evaluation parquet path.",
    )
    parser.add_argument(
        "--target-distributions",
        default=str(pitch_target_distributions_path),
        help="Pitch target distributions parquet path.",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=100,
        help="Optional per-outcome sample cap for balanced held-out evaluation.",
    )
    parser.add_argument(
        "--recommender-bundle",
        default=str(event_tree_model_path),
        help="Bundle used to choose recommended buckets.",
    )
    parser.add_argument(
        "--evaluator-bundle",
        default=str(event_tree_evaluator),
        help="Independent bundle used to score recommended vs observed buckets.",
    )
    return parser


def _progress(prefix: str, current: int, total: int, detail: str):
    total = max(int(total), 1)
    current = max(0, min(int(current), total))
    bar_len = 20
    filled = int(round(bar_len * current / total))
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"{prefix} [{bar}] {current}/{total} {detail}")


def _safe_float(value: object, default: float):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float(default)


def _display_path(path: str | Path):
    resolved = Path(path)
    try:
        return str(resolved.resolve().relative_to(package))
    except Exception:
        return resolved.name if resolved.name else str(resolved)


def _score_target(
    row: pd.Series,
    event_bundle: dict[str, object],
    *,
    target_x: float,
    target_z: float):

    target_frame = pd.DataFrame([row.to_dict()])
    target_frame["intended_target_x"] = float(target_x)
    target_frame["intended_target_z"] = float(target_z)
    return float(planner_expected_pitcher_value(target_frame, event_bundle)[0])


def _score_frame(frame: pd.DataFrame, event_bundle: dict[str, object]):
    if frame.empty:
        return np.asarray([], dtype=float)
    return np.asarray(planner_expected_pitcher_value(frame, event_bundle), dtype=float)


def _observed_bucket(row: pd.Series):
    existing_bucket = row.get("pitch_2_bucket")
    if existing_bucket is not None and str(existing_bucket):
        return str(existing_bucket)
    observed_x = pd.to_numeric(pd.Series([row.get("observed_plate_x")]), errors="coerce").iloc[0]
    observed_z = pd.to_numeric(pd.Series([row.get("observed_plate_z")]), errors="coerce").iloc[0]
    return str(locate_bucket_id(observed_x, observed_z, batter_height_ft=row.get("batter_height_ft")))


def _balanced_outcome_sample(frame: pd.DataFrame, per_outcome_cap: int):
    if per_outcome_cap <= 0 or "observed_outcome_bucket" not in frame.columns:
        return frame.copy()
    sampled_frames: list[pd.DataFrame] = []
    for outcome_bucket in OUTCOME_SAMPLE_ORDER:
        bucket_frame = frame.loc[frame["observed_outcome_bucket"].astype(str).eq(outcome_bucket)].copy()
        if bucket_frame.empty:
            continue
        if len(bucket_frame) > per_outcome_cap:
            bucket_frame = bucket_frame.sample(n=per_outcome_cap, random_state=random_seed).copy()
        sampled_frames.append(bucket_frame)
    if not sampled_frames:
        return frame.iloc[0:0].copy()
    return pd.concat(sampled_frames, ignore_index=True).sort_values("state_id").reset_index(drop=True)


def _batter_cache_path(row: pd.Series):
    return batter_top3_cache_path_for_row(row)


def _pitcher_cache_path(row: pd.Series):
    return pitcher_top3_cache_path_for_row(row)


@lru_cache(maxsize=1)
def _runtime_planner():
    from backend.v2_runtime import load_v2_backend_runtime

    runtime = load_v2_backend_runtime()
    return runtime["planner_runtime"]


def _session_from_eval_row(row: pd.Series):
    from backend.v2_runtime import AtBatSession, PitchRecord

    batter_hand = str(row.get("batter_handedness", "")).upper()
    session = AtBatSession(
        at_bat_id=f"eval_{row['state_id']}",
        pitcher_id=int(row["pitcher_id"]),
        pitcher_name=str(row.get("pitcher_name", "")),
        pitcher_team=None if pd.isna(row.get("pitcher_team")) else str(row.get("pitcher_team")),
        pitcher_handedness=str(row.get("pitcher_handedness", "")).upper(),
        batter_id=int(row["batter_id"]),
        batter_name=str(row.get("batter_name", "")),
        batter_team=None if pd.isna(row.get("batter_team")) else str(row.get("batter_team")),
        batter_handedness=batter_hand,
        balls=int(pd.to_numeric(pd.Series([row.get("balls_before_p2", 0)]), errors="coerce").fillna(0).iloc[0]),
        strikes=int(pd.to_numeric(pd.Series([row.get("strikes_before_p2", 0)]), errors="coerce").fillna(0).iloc[0]),
        next_pitch_number=2,
        terminal=False,
        pitch_history=[
            PitchRecord(
                pitch_number=1,
                pitch_type=str(row.get("pitch_1_type", "")).upper(),
                target_x=float(pd.to_numeric(pd.Series([row.get("pitch_1_plate_x")]), errors="coerce").fillna(0.0).iloc[0]),
                target_z=float(pd.to_numeric(pd.Series([row.get("pitch_1_plate_z")]), errors="coerce").fillna(0.0).iloc[0]),
            )
        ],
    )
    return session


def _generated_candidate_frame(
    row: pd.Series,
    target_rows: pd.DataFrame):

    if target_rows.empty:
        target_rows = _default_target_rows().assign(candidate_pool=broad_target_scope)
    generated_target_rows = _generated_candidate_rows_from_target_rows(target_rows)
    if generated_target_rows.empty:
        generated_target_rows = _generated_candidate_rows_from_target_rows(
            _default_target_rows().assign(candidate_pool=broad_target_scope)
        )
    return _generated_candidate_frame_from_row(row, generated_target_rows).reset_index(drop=True)


def _candidate_target_rows_from_top3_caches(
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

    batter_hit, batter_buckets = nested_bucket_lookup(batter_top3_cache, _batter_cache_path(row))
    if batter_hit:
        stats["batter_cache_hits"] = 1
        append_cached_bucket_records(
            selected,
            seen_buckets,
            batter_buckets,
            candidate_pool=batter_specific_target_scope,
            target_context_scope=batter_specific_target_scope,
        )
    else:
        stats["batter_fallbacks"] = 1
        selected.extend(
            _select_pool_target_rows(
                row,
                targets,
                candidate_pool=batter_specific_target_scope,
                scope_order=(batter_specific_target_scope,),
                seen_buckets=seen_buckets,
                lookup_indexes=lookup_indexes,
            )
        )

    pitcher_hit, pitcher_buckets = nested_bucket_lookup(pitcher_top3_cache, _pitcher_cache_path(row))
    if pitcher_hit:
        stats["pitcher_cache_hits"] = 1
        append_cached_bucket_records(
            selected,
            seen_buckets,
            pitcher_buckets,
            candidate_pool=handidness_target_scope,
            target_context_scope=handidness_target_scope,
        )
    else:
        stats["pitcher_fallbacks"] = 1
        selected.extend(
            _select_pool_target_rows(
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
        )

    if not selected:
        return pd.DataFrame(), stats
    return pd.DataFrame.from_records(selected), stats


def evaluate_location_only_planner(
    eval_view_path: str,
    target_distribution_path: str,
    max_states: int,
    recommender_bundle_path: str,
    evaluator_bundle_path: str):

    frame = pd.read_parquet(eval_view_path)
    frame = frame.loc[pd.to_numeric(frame["season"], errors="coerce").eq(2025)].copy()
    frame = frame.loc[assign_crossfit_fold(frame).eq(4)].copy()
    frame = _balanced_outcome_sample(frame, max_states)

    recommender_bundle = _pickle_load(recommender_bundle_path)
    evaluator_bundle = _pickle_load(evaluator_bundle_path)
    targets = prepare_target_lookup_frame(pd.read_parquet(target_distribution_path))
    lookup_indexes = build_target_lookup_indexes(targets)
    batter_top3_cache = load_top3_cache(batter_top3_cache_path)
    pitcher_top3_cache = load_top3_cache(pitcher_top3_cache_path)
    cache_stats = {
        "batter_cache_hits": 0,
        "pitcher_cache_hits": 0,
        "batter_fallbacks": 0,
        "pitcher_fallbacks": 0,
        "batter_top3_cache_path": _display_path(batter_top3_cache_path),
        "pitcher_top3_cache_path": _display_path(pitcher_top3_cache_path),
        "recommender_bundle_path": _display_path(recommender_bundle_path),
        "evaluator_bundle_path": _display_path(evaluator_bundle_path),
    }
    records: list[dict[str, object]] = []
    total = len(frame)
    print(f"Evaluating v2 planner on {total:,} held-out states")
    for idx, (_, row) in enumerate(frame.iterrows(), start=1):
        if idx == 1 or idx == total or idx % 250 == 0:
            _progress("Evaluation", idx, total, f"state_id={row['state_id']}")
        target_rows, row_stats = _candidate_target_rows_from_top3_caches(
            row,
            batter_top3_cache=batter_top3_cache,
            pitcher_top3_cache=pitcher_top3_cache,
            targets=targets,
            lookup_indexes=lookup_indexes,
        )
        for key in ("batter_cache_hits", "pitcher_cache_hits", "batter_fallbacks", "pitcher_fallbacks"):
            cache_stats[key] += int(row_stats[key])
        candidates = _candidate_frame_from_target_rows(row, target_rows, recommender_bundle)
        recommendation_scores = planner_expected_pitcher_value(candidates, recommender_bundle)
        best_idx = int(np.argmax(recommendation_scores))
        actual_bucket = _observed_bucket(row)
        actual_x, actual_z = bucket_center(actual_bucket, batter_height_ft=row.get("batter_height_ft"))
        actual_score = _score_target(
            row,
            evaluator_bundle,
            target_x=actual_x,
            target_z=actual_z,
        )
        recommended_score = _score_target(
            row,
            evaluator_bundle,
            target_x=float(candidates.iloc[best_idx]["intended_target_x"]),
            target_z=float(candidates.iloc[best_idx]["intended_target_z"]),
        )
        records.append(
            {
                "state_id": row["state_id"],
                "observed_pitch_type": row["pitch_2_type"],
                "observed_outcome_bucket": row["observed_outcome_bucket"],
                "actual_bucket": actual_bucket,
                "actual_target_x": actual_x,
                "actual_target_z": actual_z,
                "actual_score": actual_score,
                "recommended_score": float(recommended_score),
                "recommender_selection_score": float(recommendation_scores[best_idx]),
                "recommended_bucket": str(candidates.iloc[best_idx].get("pitch_2_bucket", "")),
                "recommended_target_x": float(candidates.iloc[best_idx]["intended_target_x"]),
                "recommended_target_z": float(candidates.iloc[best_idx]["intended_target_z"]),
                "bucket_match": str(candidates.iloc[best_idx].get("pitch_2_bucket", "")) == actual_bucket,
                "gain": float(recommended_score - actual_score),
            }
        )
    report = pd.DataFrame(records)
    if "bucket_match" in report.columns and len(report):
        cache_stats["total_rows_scored"] = int(len(report))
        cache_stats["matching_rows_filtered_out"] = int(report["bucket_match"].sum())
        cache_stats["non_matching_rows_kept"] = int((~report["bucket_match"]).sum())
    return report, cache_stats


def evaluate_full_recommendation_planner(
    eval_view_path: str,
    target_distribution_path: str,
    max_states: int,
    recommender_bundle_path: str,
    evaluator_bundle_path: str):

    frame = pd.read_parquet(eval_view_path)
    frame = frame.loc[pd.to_numeric(frame["season"], errors="coerce").eq(2025)].copy()
    frame = frame.loc[assign_crossfit_fold(frame).eq(4)].copy()
    frame = _balanced_outcome_sample(frame, max_states)

    planner_runtime = _runtime_planner()
    recommender_bundle = _pickle_load(recommender_bundle_path)
    evaluator_bundle = _pickle_load(evaluator_bundle_path)
    targets = prepare_target_lookup_frame(pd.read_parquet(target_distribution_path))
    lookup_indexes = build_target_lookup_indexes(targets)
    batter_top3_cache = load_top3_cache(batter_top3_cache_path)
    pitcher_top3_cache = load_top3_cache(pitcher_top3_cache_path)
    cache_stats = {
        "batter_cache_hits": 0,
        "pitcher_cache_hits": 0,
        "batter_fallbacks": 0,
        "pitcher_fallbacks": 0,
        "pitch_type_scenarios_scored": 0,
        "rows_with_any_batter_cache_hit": 0,
        "rows_with_any_pitcher_cache_hit": 0,
        "rows_with_no_pitch_type_candidates": 0,
        "batter_top3_cache_path": _display_path(batter_top3_cache_path),
        "pitcher_top3_cache_path": _display_path(pitcher_top3_cache_path),
        "recommender_bundle_path": _display_path(recommender_bundle_path),
        "evaluator_bundle_path": _display_path(evaluator_bundle_path),
    }
    records: list[dict[str, object]] = []
    total = len(frame)
    print(f"Evaluating v2 planner (full pitch-type + location) on {total:,} held-out states")
    for idx, (_, row) in enumerate(frame.iterrows(), start=1):
        if idx == 1 or idx == total or idx % 250 == 0:
            _progress("Full Eval", idx, total, f"state_id={row['state_id']}")
        session = _session_from_eval_row(row)
        candidate_pitch_types = list(planner_runtime.available_pitch_types(int(row["pitcher_id"])))
        observed_pitch_type = str(row.get("pitch_2_type", "")).upper()
        if observed_pitch_type and observed_pitch_type not in candidate_pitch_types:
            candidate_pitch_types.append(observed_pitch_type)

        candidate_frames: list[pd.DataFrame] = []
        row_had_batter_hit = False
        row_had_pitcher_hit = False
        for pitch_type in candidate_pitch_types:
            try:
                synthetic_row, match_level = planner_runtime._build_synthetic_row(session, pitch_type)
            except ValueError:
                continue
            target_rows, row_stats = _candidate_target_rows_from_top3_caches(
                synthetic_row,
                batter_top3_cache=batter_top3_cache,
                pitcher_top3_cache=pitcher_top3_cache,
                targets=targets,
                lookup_indexes=lookup_indexes,
            )
            for key in ("batter_cache_hits", "pitcher_cache_hits", "batter_fallbacks", "pitcher_fallbacks"):
                cache_stats[key] += int(row_stats[key])
            row_had_batter_hit = row_had_batter_hit or bool(row_stats["batter_cache_hits"])
            row_had_pitcher_hit = row_had_pitcher_hit or bool(row_stats["pitcher_cache_hits"])
            cache_stats["pitch_type_scenarios_scored"] += 1
            candidates = _generated_candidate_frame(synthetic_row, target_rows).copy()
            if candidates.empty:
                continue
            candidates["template_match_level"] = match_level
            candidates["recommendation_group"] = str(pitch_type).upper()
            candidate_frames.append(candidates)

        if row_had_batter_hit:
            cache_stats["rows_with_any_batter_cache_hit"] += 1
        if row_had_pitcher_hit:
            cache_stats["rows_with_any_pitcher_cache_hit"] += 1
        if not candidate_frames:
            cache_stats["rows_with_no_pitch_type_candidates"] += 1
            continue

        all_candidates = pd.concat(candidate_frames, ignore_index=True)
        recommendation_scores = _score_frame(all_candidates, recommender_bundle)
        best_idx = int(np.argmax(recommendation_scores))
        best_candidate = all_candidates.iloc[[best_idx]].copy()
        best_row = best_candidate.iloc[0]
        actual_bucket = _observed_bucket(row)
        actual_x, actual_z = bucket_center(actual_bucket, batter_height_ft=row.get("batter_height_ft"))
        actual_score = _score_target(
            row,
            evaluator_bundle,
            target_x=actual_x,
            target_z=actual_z,
        )
        recommended_score = float(_score_frame(best_candidate, evaluator_bundle)[0])
        records.append(
            {
                "state_id": row["state_id"],
                "observed_pitch_type": row["pitch_2_type"],
                "observed_outcome_bucket": row["observed_outcome_bucket"],
                "actual_bucket": actual_bucket,
                "actual_target_x": actual_x,
                "actual_target_z": actual_z,
                "actual_score": actual_score,
                "recommended_pitch_type": str(best_row.get("pitch_2_type", "")),
                "recommended_bucket": str(best_row.get("pitch_2_bucket", "")),
                "recommended_target_x": float(best_row["intended_target_x"]),
                "recommended_target_z": float(best_row["intended_target_z"]),
                "recommended_score": recommended_score,
                "recommender_selection_score": float(recommendation_scores[best_idx]),
                "pitch_type_match": str(best_row.get("pitch_2_type", "")).upper() == observed_pitch_type,
                "bucket_match": str(best_row.get("pitch_2_bucket", "")) == actual_bucket,
                "full_match": (
                    str(best_row.get("pitch_2_type", "")).upper() == observed_pitch_type
                    and str(best_row.get("pitch_2_bucket", "")) == actual_bucket
                ),
                "template_match_level": str(best_row.get("template_match_level", "")),
                "candidate_pool": str(best_row.get("candidate_pool", "")),
                "gain": float(recommended_score - actual_score),
            }
        )

    report = pd.DataFrame(records)
    if "full_match" in report.columns and len(report):
        cache_stats["total_rows_scored"] = int(len(report))
        cache_stats["matching_rows_filtered_out"] = int(report["full_match"].sum())
        cache_stats["non_matching_rows_kept"] = int((~report["full_match"]).sum())
    return report, cache_stats


def summarize_report(report: pd.DataFrame, *, evaluation_mode: str | None = None):
    if report.empty:
        summary = pd.DataFrame(
            [
                {
                    "observed_outcome_bucket": "OVERALL",
                    "rows": 0,
                    "avg_recommended_score": np.nan,
                    "avg_actual_score": np.nan,
                    "avg_gain": np.nan,
                }
            ]
        )
        if evaluation_mode is not None:
            summary.insert(0, "evaluation_mode", evaluation_mode)
        return summary
    bucket_summary = (
        report.groupby("observed_outcome_bucket", dropna=False)
        .agg(
            rows=("state_id", "size"),
            avg_recommended_score=("recommended_score", "mean"),
            avg_actual_score=("actual_score", "mean"),
            avg_gain=("gain", "mean"),
        )
        .reset_index()
        .sort_values("avg_actual_score")
    )
    overall = pd.DataFrame(
        [
            {
                "observed_outcome_bucket": "OVERALL",
                "rows": len(report),
                "avg_recommended_score": report["recommended_score"].mean(),
                "avg_actual_score": report["actual_score"].mean(),
                "avg_gain": report["gain"].mean(),
            }
        ]
    )
    bucket_summary = pd.concat([overall, bucket_summary], ignore_index=True)
    if evaluation_mode is not None:
        bucket_summary.insert(0, "evaluation_mode", evaluation_mode)
    return bucket_summary


def main():
    args = build_argument_parser().parse_args()
    location_report, location_cache_stats = evaluate_location_only_planner(
        args.eval_view,
        args.target_distributions,
        args.max_states,
        args.recommender_bundle,
        args.evaluator_bundle,
    )
    full_report, full_cache_stats = evaluate_full_recommendation_planner(
        args.eval_view,
        args.target_distributions,
        args.max_states,
        args.recommender_bundle,
        args.evaluator_bundle,
    )
    location_summary = summarize_report(location_report, evaluation_mode="location_only")
    full_summary = summarize_report(full_report, evaluation_mode="pitch_type_and_location")
    summary = pd.concat([location_summary, full_summary], ignore_index=True)
    eval_report.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(eval_report, index=False)
    planner_metadata = _planner_metadata()
    evaluator_metadata = _planner_evaluator_metadata()
    traceability = _traceability_metadata()
    print("Location-only evaluation:")
    print(location_summary.drop(columns=["evaluation_mode"], errors="ignore").to_string(index=False))
    print("\nFull pitch-type + location evaluation:")
    print(full_summary.drop(columns=["evaluation_mode"], errors="ignore").to_string(index=False))
    if location_cache_stats:
        print("\nTop-3 Cache Usage (location-only):")
        print(json.dumps(location_cache_stats, indent=2, sort_keys=True))
    if full_cache_stats:
        print("\nTop-3 Cache Usage (full pitch-type + location):")
        print(json.dumps(full_cache_stats, indent=2, sort_keys=True))
    recommender_lower_tree = (
        planner_metadata.get("event_metrics", {}).get("lower_tree_heldout")
        if isinstance(planner_metadata.get("event_metrics"), dict)
        else None
    )
    evaluator_lower_tree = (
        evaluator_metadata.get("event_metrics", {}).get("lower_tree_heldout")
        if isinstance(evaluator_metadata.get("event_metrics"), dict)
        else None
    )
    if recommender_lower_tree:
        print("\nHeld-out lower-tree diagnostics (recommender):")
        print(json.dumps(recommender_lower_tree, indent=2, sort_keys=True))
    if evaluator_lower_tree:
        print("\nHeld-out lower-tree diagnostics (evaluator):")
        print(json.dumps(evaluator_lower_tree, indent=2, sort_keys=True))
    if traceability:
        print("\nEvaluation traceability contract:")
        print(
            json.dumps(
                {
                    "contract_version": traceability.get("contract_version", "unknown"),
                    "research_objective_components": traceability.get("research_objective_components", {}),
                    "deployment_objective_overrides": traceability.get("deployment_objective_overrides", {}),
                    "cross_fit_contract": traceability.get("cross_fit_contract", {}),
                },
                indent=2,
                sort_keys=True,
            )
        )
    print(f"Wrote v2 evaluation summary to {_display_path(eval_report)}")


if __name__ == "__main__":
    main()
