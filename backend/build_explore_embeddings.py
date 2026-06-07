from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.v2_runtime import load_v2_backend_runtime

data_dir = repo_root / "data"
table_dir = repo_root / "models" / "v2" / "tables"
public_dir = repo_root / "frontend" / "public" / "explore"

pitches = ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CS", "CH", "FS", "FO"]
family = ["fastball", "breaking", "offspeed"]
generic_ids = {-1001, -1002}

generic_labels = {
    -1001: "Generic LHB",
    -1002: "Generic RHB",
}
generic_hand = {
    -1001: "L",
    -1002: "R",
}


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _normalize_rows(coords: np.ndarray):
    return np.asarray(coords, dtype=float)


def _nearest_neighbor_ids(ids: list[int], coords: np.ndarray, limit: int = 3):
    coords = np.asarray(coords, dtype=float)
    neighbors: dict[int, list[int]] = {}
    if len(ids) <= 1:
        return {int(idx): [] for idx in ids}
    for i, item_id in enumerate(ids):
        deltas = coords - coords[i]
        distances = np.sqrt((deltas ** 2).sum(axis=1))
        order = np.argsort(distances)
        matches = [int(ids[j]) for j in order if j != i][:limit]
        neighbors[int(item_id)] = matches
    return neighbors


def _pitch_name(code: str):
    names = {
        "FF": "4-Seam Fastball",
        "SI": "Sinker",
        "FC": "Cutter",
        "SL": "Slider",
        "ST": "Sweeper",
        "SV": "Slurve",
        "CU": "Curveball",
        "KC": "Knuckle Curve",
        "CS": "Slow Curve",
        "CH": "Changeup",
        "FS": "Split-Finger",
        "FO": "Forkball",
    }
    return names.get(code, code)


#embed to 2d
def _fit_embedding(frame: pd.DataFrame, feature_columns: list[str], cluster_count: int):
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
    ])

    coords = pipeline.fit_transform(frame[feature_columns])
    coords = _normalize_rows(coords)
    n_clusters = max(2, min(cluster_count, len(frame)))
    clusters = KMeans(n_clusters=n_clusters, n_init=20, random_state=42).fit_predict(coords)
    return coords, clusters.astype(int)


#pitcher explore features
def build_pitcher_points(runtime):
    pitcher_rows = pd.DataFrame(runtime["planner_runtime"].pitcher_list())
    pitcher_ids = set(pd.to_numeric(pitcher_rows["pitcher_id"], errors="coerce").dropna().astype(int))
    averages = pd.read_csv(data_dir / "pitch_type_averages_2025.csv")
    averages = averages.loc[averages["mlbam_id"].isin(pitcher_ids)].copy()
    averages["pitcher_hand"] = averages["pitcher_hand"].fillna("Unknown").astype(str).str.upper()

    records = []
    for row in averages.itertuples(index=False):
        total = sum(_safe_float(getattr(row, f"{pitch}_pitch_count", 0.0), 0.0) for pitch in pitches)
        total = max(total, 1.0)
        record = {
            "pitcher_id": int(row.mlbam_id),
            "pitcher_name": str(row.Name),
            "pitcher_team": str(row.team or ""),
            "pitcher_handedness": str(row.pitcher_hand or "Unknown").upper(),
            "total_pitches": total,
        }
        weighted_velo = 0.0
        weighted_spin = 0.0
        weighted_h = 0.0
        weighted_v = 0.0
        weighted_ext = 0.0
        usage_values = {}
        for pitch in pitches:
            count = _safe_float(getattr(row, f"{pitch}_pitch_count", 0.0), 0.0)
            usage = count / total if total else 0.0
            usage_values[pitch] = usage
            record[f"usage_{pitch}"] = usage
            velo = _safe_float(getattr(row, f"{pitch}_velo", np.nan), np.nan)
            spin = _safe_float(getattr(row, f"{pitch}_spin_rate", np.nan), np.nan)
            h_mov = _safe_float(getattr(row, f"{pitch}_h_mov", np.nan), np.nan)
            v_mov = _safe_float(getattr(row, f"{pitch}_v_mov", np.nan), np.nan)
            ext = _safe_float(getattr(row, f"{pitch}_extension", np.nan), np.nan)
            record[f"velo_{pitch}"] = 0.0 if np.isnan(velo) else velo
            record[f"spin_{pitch}"] = 0.0 if np.isnan(spin) else spin
            record[f"hmov_{pitch}"] = 0.0 if np.isnan(h_mov) else h_mov
            record[f"vmov_{pitch}"] = 0.0 if np.isnan(v_mov) else v_mov
            if usage > 0 and np.isfinite(velo):
                weighted_velo += usage * velo
            if usage > 0 and np.isfinite(spin):
                weighted_spin += usage * spin
            if usage > 0 and np.isfinite(h_mov):
                weighted_h += usage * h_mov
            if usage > 0 and np.isfinite(v_mov):
                weighted_v += usage * v_mov
            if usage > 0 and np.isfinite(ext):
                weighted_ext += usage * ext
        record["weighted_velo"] = weighted_velo
        record["weighted_spin"] = weighted_spin
        record["weighted_hmov"] = weighted_h
        record["weighted_vmov"] = weighted_v
        record["weighted_extension"] = weighted_ext
        dominant_pitch = max(usage_values, key=usage_values.get)
        record["dominant_pitch"] = dominant_pitch
        record["dominant_pitch_usage"] = usage_values[dominant_pitch]
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    feature_columns = [col for col in frame.columns if col.startswith(("usage_", "velo_", "spin_", "hmov_", "vmov_"))]
    feature_columns += ["weighted_velo", "weighted_spin", "weighted_hmov", "weighted_vmov", "weighted_extension", "dominant_pitch_usage"]
    coords, clusters = _fit_embedding(frame, feature_columns, cluster_count=7)
    neighbors = _nearest_neighbor_ids(frame["pitcher_id"].tolist(), coords, limit=3)
    points = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        dominant_pitch = row["dominant_pitch"]
        weighted_velo = _safe_float(row["weighted_velo"])
        if dominant_pitch in {"FF", "SI", "FC"} and weighted_velo >= 94:
            archetype = "Power fastball mix"
        elif dominant_pitch in {"FF", "SI", "FC"}:
            archetype = "Fastball-first mix"
        elif dominant_pitch in {"SL", "ST", "SV", "CU", "KC", "CS"}:
            archetype = "Breaking-ball leaning"
        else:
            archetype = "Offspeed leaning"
        summary = f"{_pitch_name(dominant_pitch)}-led repertoire · {weighted_velo:.1f} mph weighted velo"
        points.append({
            "id": int(row["pitcher_id"]),
            "name": row["pitcher_name"],
            "team": row["pitcher_team"],
            "hand": row["pitcher_handedness"],
            "x": round(float(coords[idx, 0]), 4),
            "y": round(float(coords[idx, 1]), 4),
            "cluster": int(clusters[idx]),
            "archetype": archetype,
            "summary": summary,
            "dominant_pitch": dominant_pitch,
            "dominant_pitch_name": _pitch_name(dominant_pitch),
            "neighbors": neighbors[int(row["pitcher_id"])],
        })
    return {"generated_from": "pitch_type_averages_2025.csv", "points": points}


#hitter explore features
def build_hitter_points(runtime):
    batter_rows = pd.DataFrame(runtime["planner_runtime"].batter_list())
    real_batters = batter_rows.loc[~batter_rows["batter_id"].isin(generic_ids)].copy()
    batter_ids = set(pd.to_numeric(real_batters["batter_id"], errors="coerce").dropna().astype(int))
    view = pd.read_parquet(table_dir / "pitch2_planner_eval_view.parquet", columns=[
        "batter_id", "batter_name", "batter_team", "batter_handedness", "pitch_2_family",
        "swing", "whiff", "contact", "in_play", "single", "double_or_triple", "home_run", "out", "pitcher_value_on_contact"
    ])
    view = view.loc[view["batter_id"].isin(batter_ids)].copy()
    view["pitch_2_family"] = view["pitch_2_family"].fillna("unknown").astype(str)
    metrics = ["swing", "whiff", "contact", "in_play", "single", "double_or_triple", "home_run", "out", "pitcher_value_on_contact"]
    grouped = view.groupby(["batter_id", "pitch_2_family"], dropna=False).agg({**{m: "mean" for m in metrics}, "batter_name": "last", "batter_team": "last", "batter_handedness": "last"}).reset_index()
    counts = view.groupby(["batter_id", "pitch_2_family"], dropna=False).size().reset_index(name="pitch_count")
    grouped = grouped.merge(counts, on=["batter_id", "pitch_2_family"], how="left")

    records = []
    for batter_id, batter_group in grouped.groupby("batter_id", dropna=False):
        batter_group = batter_group.copy()
        total = float(batter_group["pitch_count"].sum()) if len(batter_group) else 1.0
        record = {
            "batter_id": int(batter_id),
            "batter_name": str(batter_group["batter_name"].iloc[-1]),
            "batter_team": str(batter_group["batter_team"].iloc[-1] or ""),
            "batter_handedness": str(batter_group["batter_handedness"].iloc[-1] or "Unknown").upper(),
        }
        overall = view.loc[view["batter_id"].eq(batter_id)]
        record["overall_swing"] = overall["swing"].mean()
        record["overall_whiff"] = overall["whiff"].mean()
        record["overall_contact"] = overall["contact"].mean()
        record["overall_in_play"] = overall["in_play"].mean()
        record["overall_pitcher_value_on_contact"] = overall["pitcher_value_on_contact"].mean()
        for family in family:
            family_row = batter_group.loc[batter_group["pitch_2_family"].eq(family)]
            family_count = float(family_row["pitch_count"].iloc[0]) if not family_row.empty else 0.0
            record[f"share_{family}"] = family_count / total if total else 0.0
            for metric in metrics:
                record[f"{family}_{metric}"] = float(family_row[metric].iloc[0]) if not family_row.empty else np.nan
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    feature_columns = [c for c in frame.columns if any(c.startswith(prefix) for prefix in ["overall_", "share_", "fastball_", "breaking_", "offspeed_"])]
    coords, clusters = _fit_embedding(frame, feature_columns, cluster_count=6)
    neighbors = _nearest_neighbor_ids(frame["batter_id"].tolist(), coords, limit=3)

    generic_points = []
    feature_frame = frame.copy()
    scaler_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
    ])
    feature_coords = scaler_pipeline.fit_transform(feature_frame[feature_columns])
    kmeans = KMeans(n_clusters=max(2, min(6, len(feature_frame))), n_init=20, random_state=42).fit(feature_coords)
    points = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        whiff_family = max(family, key=lambda family: _safe_float(row.get(f"{family}_whiff", 0.0), 0.0))
        contact_family = max(family, key=lambda family: _safe_float(row.get(f"{family}_contact", 0.0), 0.0))
        if whiff_family == "breaking":
            archetype = "Breaking-ball vulnerable"
        elif whiff_family == "offspeed":
            archetype = "Offspeed vulnerable"
        else:
            archetype = "Fastball-oriented"
        summary = f"Most swing-and-miss vs {whiff_family} · best contact vs {contact_family}"
        points.append({
            "id": int(row["batter_id"]),
            "name": row["batter_name"],
            "team": row["batter_team"],
            "hand": row["batter_handedness"],
            "x": round(float(coords[idx, 0]), 4),
            "y": round(float(coords[idx, 1]), 4),
            "cluster": int(clusters[idx]),
            "archetype": archetype,
            "summary": summary,
            "neighbors": neighbors[int(row["batter_id"])],
            "is_generic": False,
        })

    for generic_id, generic_name in generic_labels.items():
        hand = generic_hand[generic_id]
        hand_frame = frame.loc[frame["batter_handedness"].eq(hand)].copy()
        if hand_frame.empty:
            hand_frame = frame.copy()
        feature_vector = hand_frame[feature_columns].mean(numeric_only=True)
        transformed = scaler_pipeline.transform(pd.DataFrame([feature_vector], columns=feature_columns))
        cluster = int(kmeans.predict(transformed)[0])
        coords_generic = transformed[0]
        generic_points.append({
            "id": int(generic_id),
            "name": generic_name,
            "team": "",
            "hand": hand,
            "x": round(float(coords_generic[0]), 4),
            "y": round(float(coords_generic[1]), 4),
            "cluster": cluster,
            "archetype": f"League-average {hand}HB",
            "summary": f"League-average {hand.lower()}-handed hitter profile across pitch families",
            "neighbors": [],
            "is_generic": True,
        })

    return {"generated_from": "pitch2_planner_eval_view.parquet", "points": points + generic_points}


def main():
    public_dir.mkdir(parents=True, exist_ok=True)
    runtime = load_v2_backend_runtime()
    pitcher_payload = build_pitcher_points(runtime)
    hitter_payload = build_hitter_points(runtime)
    (public_dir / 'pitchers.json').write_text(json.dumps(pitcher_payload))
    (public_dir / 'hitters.json').write_text(json.dumps(hitter_payload))
    print(f"Wrote {len(pitcher_payload['points'])} pitcher points to {public_dir / 'pitchers.json'}")
    print(f"Wrote {len(hitter_payload['points'])} hitter points to {public_dir / 'hitters.json'}")


if __name__ == '__main__':
    main()
