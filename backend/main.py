from __future__ import annotations
from contextlib import asynccontextmanager
import logging
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.v2_runtime import load_v2_backend_runtime, prewarm_backend_runtime

class PredictRequest(BaseModel):
    pitcher_id: int
    batter_id: int
    pitch_type_1: str
    plate_x_1: float
    plate_z_1: float
    candidate_pitch_types: list[str] | None = Field(default=None)


runtime: dict[str, Any] | None = None
load_error: str | None = None
logger = logging.getLogger(__name__)

#shared runtime handle
def _runtime():
    if runtime is None:
        detail = load_error or "Backend runtime is not loaded."
        raise HTTPException(status_code=503, detail=detail)
    return runtime


def _bad_request(error: ValueError):
    message = str(error)
    status = 404 if "not found" in message.lower() else 400
    raise HTTPException(status_code=status, detail=message) from error


@asynccontextmanager
async def lifespan(_: FastAPI):
    global runtime, load_error
    try:
        #load saved runtime
        runtime = load_v2_backend_runtime()
        try:
            runtime["prewarm_summary"] = prewarm_backend_runtime(runtime)
        except Exception as exc:  
            runtime["prewarm_summary"] = {
                "status": "failed",
                "error": str(exc),
            }
        load_error = None
    except Exception as exc: 
        runtime = None
        load_error = str(exc)
        logger.exception('Failed to load backend runtime')
    yield


app = FastAPI(
    title="Pitch Planner Backend",
    version="2.0.0",
    description="Backend for the pitch planning app",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok" if runtime is not None else "degraded",
        "runtime_loaded": runtime is not None,
        "load_error": load_error,
        "prewarm": runtime.get("prewarm_summary") if runtime is not None else None,
        "model_notes_version": (
            runtime.get("traceability", {}).get("contract_version") if runtime is not None else None
        ),
    }


@app.get("/api/pitchers")
def pitchers():
    return _runtime()["planner_runtime"].pitcher_list()


@app.get("/api/batters")
def batters():
    return _runtime()["planner_runtime"].batter_list()


@app.get("/api/pitcher/{pitcher_id}/trajectories")
def pitcher_trajectories(pitcher_id: int):
    runtime = _runtime()
    planner = runtime["planner_runtime"]
    try:
        return planner.pitcher_trajectories(pitcher_id)
    except ValueError as exc:
        _bad_request(exc)


@app.get("/api/batters/{batter_id}/strike-zone")
def batter_strike_zone(batter_id: int, pitcher_id: int | None = None):
    runtime = _runtime()
    planner = runtime["planner_runtime"]
    contour_store = runtime["contour_store"]
    try:
        batter = planner.matchup_batter_info(batter_id, pitcher_id=pitcher_id)
        response = contour_store.strike_zone_contour(
            batter_id=batter_id,
            batter_hand=str(batter["batter_handedness"]).upper(),
        )
        response["batter_handedness"] = str(planner.batter_info(batter_id).get("batter_handedness", "")).upper()
        response["resolved_batter_handedness"] = str(batter.get("resolved_batter_handedness", batter["batter_handedness"])).upper()
        response["batter_is_switch_hitter"] = bool(planner.batter_info(batter_id).get("batter_is_switch_hitter", False))
        return response
    except ValueError as exc:
        _bad_request(exc)


@app.post("/api/predict")
#main recommendation endpoint
def predict(request: PredictRequest):
    runtime = _runtime()
    sessions = runtime["session_manager"]
    planner = runtime["planner_runtime"]
    pitch_one_service = runtime["pitch_one_service"]
    traceability = runtime.get("traceability", {})

    try:
        pitcher = planner.pitcher_info(request.pitcher_id)
        batter = planner.matchup_batter_info(
            request.batter_id,
            pitcher_handedness=str(pitcher["pitcher_handedness"]).upper(),
        )
        if not planner.available_pitch_types(request.pitcher_id):
            raise ValueError(
                f"Pitcher {request.pitcher_id} does not have deployable 2025 pitch profiles for the v2 backend."
            )

        session = sessions.create_session(pitcher, batter)
        try:
            assessment = pitch_one_service.score_pitch(
                pitcher_id=request.pitcher_id,
                batter_id=request.batter_id,
                batter_hand=str(batter["batter_handedness"]).upper(),
                pitch_type=request.pitch_type_1,
                target_x=request.plate_x_1,
                target_z=request.plate_z_1,
            )
            session = sessions.score_pitch_one(
                session.at_bat_id,
                pitch_type=request.pitch_type_1,
                target_x=request.plate_x_1,
                target_z=request.plate_z_1,
                predicted_count_bucket=str(assessment["predicted_pitch2_count_bucket"]),
            )
            recommendation_payload = planner.recommend_next(session, request.candidate_pitch_types)
        finally:
            sessions.close(session.at_bat_id)

        transformed_recommendations: list[dict[str, Any]] = []
        for row in recommendation_payload["recommendations"]:
            pitch_profile = planner.get_pitch_profile(request.pitcher_id, row["pitch_type"])
            transformed_recommendations.append(
                {
                    "pitch_type": row["pitch_type"],
                    "pitch_name": row["pitch_type_name"],
                    "plate_x": row["recommended_target_x"],
                    "plate_z": row["recommended_target_z"],
                    "bucket": row.get("recommended_bucket"),
                    "score": row["expected_pitcher_value"],
                    "template_match_level": row["template_match_level"],
                    "candidate_pool": row.get("candidate_pool"),
                    "bucket_map": row.get("bucket_map"),
                    "pitch_outlook": row.get("pitch_outlook"),
                    "velo": round(float(pitch_profile.velo), 1) if pitch_profile is not None else None,
                    "spin_rate": round(float(pitch_profile.spin_rate), 0) if pitch_profile is not None else None,
                    "h_mov": round(float(pitch_profile.h_mov), 1) if pitch_profile is not None else None,
                    "v_mov": round(float(pitch_profile.v_mov), 1) if pitch_profile is not None else None,
                    "extension": round(float(pitch_profile.extension), 2) if pitch_profile is not None else None,
                }
            )

        pitch_1_assessment = {
            "pitch_type": str(request.pitch_type_1).upper(),
            "plate_x": float(request.plate_x_1),
            "plate_z": float(request.plate_z_1),
            "count_strike_probability": assessment["called_strike_given_take_probability"],
            "count_ball_probability": assessment["ball_given_take_probability"],
            "adds_strike": str(assessment["predicted_pitch2_count_bucket"]) == "0-1",
            "balls_before_pitch_2": int(session.balls),
            "strikes_before_pitch_2": int(session.strikes),
            "count_bucket_before_pitch_2": str(assessment["predicted_pitch2_count_bucket"]),
            "called_strike_probability_source": assessment["called_strike_probability_source"],
            "predicted_pitch1_mode": assessment["predicted_pitch1_mode"],
        }

        return {
            "pitch_1_assessment": pitch_1_assessment,
            "recommendations": transformed_recommendations,
            "objective_detail": {
                "objective": "rank_pitch_2_options",
                "contract_version": traceability.get("contract_version"),
                "description": (
                    "Treat pitch 1 as a take, estimate whether the next count is 0-1 or 1-0, "
                    "then rank pitch-2 target buckets by the model score."
                ),
            },
            "assumptions": recommendation_payload["assumptions"],
            "derived_count": recommendation_payload["count"],
            "traceability": traceability,
        }
    except ValueError as exc:
        _bad_request(exc)
