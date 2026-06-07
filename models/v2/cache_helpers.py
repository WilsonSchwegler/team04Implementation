from __future__ import annotations
import json
from pathlib import Path
from typing import Callable
import pandas as pd


def assign_crossfit_fold(frame: pd.DataFrame):
    working = frame.copy()
    season_values = pd.to_numeric(working.get("season"), errors="coerce").fillna(-1).astype(int)
    if "state_id" in working.columns:
        state_values = working["state_id"].fillna("").astype(str)
    else:
        state_values = pd.Series(working.index.astype(str), index=working.index)
    fold_keys = season_values.astype(str) + "::" + state_values
    hashed = pd.util.hash_pandas_object(fold_keys, index=False).astype("uint64")
    return pd.Series((hashed % 5).astype(int), index=working.index, name="crossfit_fold")


def load_top3_cache(path: str | Path):
    resolved = Path(path)
    if not resolved.exists():
        return {}
    try:
        loaded = json.loads(resolved.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def nested_bucket_lookup(tree: dict[str, object], path: tuple[str, ...]):
    current: object = tree
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return False, []
        current = current[part]
    if not isinstance(current, list):
        return False, []
    return True, [str(bucket_id) for bucket_id in current]


def top_level_lookup_ids(tree: dict[str, object]):
    ids: set[int] = set()
    if not isinstance(tree, dict):
        return ids
    for key in tree.keys():
        try:
            ids.add(int(str(key)))
        except (TypeError, ValueError):
            continue
    return ids


def batter_top3_cache_path_for_row(
    row: pd.Series,
    *,
    pitch_1_bucket_getter: Callable[[pd.Series], str] | None = None):

    pitch_1_bucket = (
        str(pitch_1_bucket_getter(row))
        if pitch_1_bucket_getter is not None
        else str(row.get("pitch_1_bucket", ""))
    )
    return (
        str(row.get("batter_id", "")),
        str(row.get("pitcher_handedness", "")).upper(),
        str(row.get("pitch_1_type", "")).upper(),
        pitch_1_bucket,
        str(row.get("pitch_2_type", "")).upper(),
        str(row.get("count_bucket", "")),
    )


def pitcher_top3_cache_path_for_row(
    row: pd.Series,
    *,
    pitch_1_bucket_getter: Callable[[pd.Series], str] | None = None):
    
    pitch_1_bucket = (
        str(pitch_1_bucket_getter(row))
        if pitch_1_bucket_getter is not None
        else str(row.get("pitch_1_bucket", ""))
    )
    return (
        str(row.get("pitcher_id", "")),
        str(row.get("batter_handedness", "")).upper(),
        str(row.get("pitch_1_type", "")).upper(),
        pitch_1_bucket,
        str(row.get("pitch_2_type", "")).upper(),
        str(row.get("count_bucket", "")),
    )


def append_cached_bucket_records(
    selected: list[dict[str, object]],
    seen_buckets: set[str],
    bucket_ids: list[str],
    *,
    candidate_pool: str,
    target_context_scope: str):
    
    for rank, bucket_id in enumerate(bucket_ids, start=1):
        normalized = str(bucket_id)
        if not normalized or normalized in seen_buckets:
            continue
        selected.append(
            {
                "pitch_2_bucket": normalized,
                "candidate_pool": candidate_pool,
                "target_context_scope": target_context_scope,
                "rank": rank,
            }
        )
        seen_buckets.add(normalized)
