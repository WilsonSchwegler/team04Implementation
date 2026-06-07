from __future__ import annotations

from pathlib import Path


p1_dir = Path(__file__).resolve().parent
models = p1_dir.parent
project = models.parent
data = project / "data"
artifact = p1_dir / "artifacts"
table = p1_dir / "tables"
report = p1_dir / "reports"
pitch_level = "pitch_level_*.csv"

source_data = models / "first_two_pitch_at_bats_2021_2025.csv"

pitch1_states = table / "pitch1_states.parquet"
arsenal_profiles = table / "pitcher_arsenal_profiles.parquet"
target_dist = table / "pitch_target_distributions.parquet"
observed_action = table / "pitch1_observed_actions.parquet"
p1_outcome = table / "pitch1_outcomes.parquet"
p1_event_tree = table / "pitch1_event_tree_view.parquet"
p1_planner_eval_view_path = table / "pitch1_planner_eval_view.parquet"

event_tree_model_path = artifact / "event_tree.pkl"
planner_metadata = artifact / "planner_metadata.pkl"
eval_report = report / "planner_evaluation_summary.csv"

default_x = (-1.25, -0.75, -0.25, 0.0, 0.25, 0.75, 1.25)
default_z = (1.2, 1.8, 2.3, 2.8, 3.3, 3.8)
random_seed = 42
start = 2021
end = 2025

