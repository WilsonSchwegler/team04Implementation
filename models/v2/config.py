from __future__ import annotations

from pathlib import Path


dir = Path(__file__).resolve().parent
model = dir.parent
project = model.parent
data = project / "data"
artifact = dir / "artifacts"
table = dir / "tables"
report = dir / "reports"
glob_pitch_level = "pitch_level_*.csv"

source_data = model / "first_two_pitch_at_bats_2021_2025.csv"

pitch2_states = table / "pitch2_states.parquet"
pitcher_arsenal_profile = table / "pitcher_arsenal_profiles.parquet"
pitch_target_distributions_path = table / "pitch_target_distributions.parquet"
pitch2_observed_action = table / "pitch2_observed_actions.parquet"
pitch2_outcome = table / "pitch2_outcomes.parquet"
pitch2_event_tree = table / "pitch2_event_tree_view.parquet"
pitch2_plavver_eval_path = table / "pitch2_planner_eval_view.parquet"

event_tree_model_path = artifact / "event_tree.pkl"
planner_metadata_path = artifact / "planner_metadata.pkl"
event_tree_evaluator = artifact / "event_tree_evaluator.pkl"
planner_eval_metadata = artifact / "planner_metadata_evaluator.pkl"
batter_bucket = artifact / "batter_bucket_history_lookup.json"
pitcher_bucket = artifact / "pitcher_bucket_history_lookup.json"
batter_top3_cache_path = artifact / "batter_bucket_top3_cache.json"
pitcher_top3_cache_path = artifact / "pitcher_bucket_top3_cache.json"
eval_report = report / "planner_evaluation_summary.csv"

random_seed = 42
