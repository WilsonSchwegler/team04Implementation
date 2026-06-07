from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    PACKAGE_ROOT = Path(__file__).resolve().parents[2]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
else:
    from .config import random_seed  # pragma: no cover


rows = 5
cols = 5
x_range = (-1.5, 1.5)
default_id = "r3_c3"
default_batter_height_ft = 6.0
min_batter_height_ft = 5.0
max_batter_height_ft = 7.5
default_zone_top = 0.535
default_zone_bottom = 0.27

x_edges = np.linspace(x_range[0], x_range[1], cols + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0

bucket_ids = tuple(
    f"r{row_idx + 1}_c{col_idx + 1}"
    for row_idx in range(rows)
    for col_idx in range(cols)
)


@dataclass(frozen=True)
class BucketCenter:
    bucket_id: str
    row: int
    col: int
    center_x: float
    center_z: float


BUCKET_LAYOUT = {
    f"r{row_idx + 1}_c{col_idx + 1}": BucketCenter(
        bucket_id=f"r{row_idx + 1}_c{col_idx + 1}",
        row=row_idx + 1,
        col=col_idx + 1,
        center_x=float(x_centers[col_idx]),
        center_z=0.0,
    )
    for row_idx in range(rows)
    for col_idx in range(cols)
}


def _coerce_coordinate(value: object, default: float):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return float(numeric)
    return float(default)


def _coerce_height(value: object):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return float(np.clip(float(numeric), min_batter_height_ft, max_batter_height_ft))
    return float(default_batter_height_ft)


#batter-relative zone bounds
def strike_zone_bounds(
    *,
    batter_height_ft: object | None = None,
    zone_bottom: object | None = None,
    zone_top: object | None = None):

    if zone_bottom is not None and zone_top is not None:
        bottom = _coerce_coordinate(zone_bottom, default_batter_height_ft * default_zone_bottom)
        top = _coerce_coordinate(zone_top, default_batter_height_ft * default_zone_top)
    else:
        height = _coerce_height(batter_height_ft)
        bottom = float(height * default_zone_bottom)
        top = float(height * default_zone_top)
    if top <= bottom:
        midpoint = (top + bottom) / 2.0
        half_height = max(abs(top - bottom), 1e-3) / 2.0
        bottom = midpoint - half_height
        top = midpoint + half_height
    return float(bottom), float(top)


def strike_zone_band_height(
    *,
    batter_height_ft: object | None = None,
    zone_bottom: object | None = None,
    zone_top: object | None = None):

    bottom, top = strike_zone_bounds(
        batter_height_ft=batter_height_ft,
        zone_bottom=zone_bottom,
        zone_top=zone_top,
    )
    return float((top - bottom) / 3.0)


def grid_vertical_edges(
    *,
    batter_height_ft: object | None = None,
    zone_bottom: object | None = None,
    zone_top: object | None = None):

    bottom, top = strike_zone_bounds(
        batter_height_ft=batter_height_ft,
        zone_bottom=zone_bottom,
        zone_top=zone_top,
    )
    band_height = max((top - bottom) / 3.0, 1e-6)
    grid_bottom = bottom - band_height
    return np.asarray([grid_bottom + band_height * idx for idx in range(rows + 1)], dtype=float)


def grid_vertical_bounds(
    *,
    batter_height_ft: object | None = None,
    zone_bottom: object | None = None,
    zone_top: object | None = None):

    edges = grid_vertical_edges(
        batter_height_ft=batter_height_ft,
        zone_bottom=zone_bottom,
        zone_top=zone_top,
    )
    return float(edges[0]), float(edges[-1])


def _aligned_series(
    value: pd.Series | object | None,
    index: pd.Index,
    *,
    default: float | None = None):

    if isinstance(value, pd.Series):
        return value.reindex(index)
    if value is None:
        fill_value = np.nan if default is None else default
        return pd.Series(fill_value, index=index)
    return pd.Series([value] * len(index), index=index)


def bucket_id_for_indices(row: int, col: int):
    row = int(np.clip(row, 1, rows))
    col = int(np.clip(col, 1, cols))
    return f"r{row}_c{col}"


def bucket_center(
    bucket_id: str,
    *,
    batter_height_ft: object | None = None,
    zone_bottom: object | None = None,
    zone_top: object | None = None):

    info = BUCKET_LAYOUT.get(str(bucket_id))
    if info is None:
        info = BUCKET_LAYOUT[default_id]
    edges = grid_vertical_edges(
        batter_height_ft=batter_height_ft,
        zone_bottom=zone_bottom,
        zone_top=zone_top,
    )
    z_centers = (edges[:-1] + edges[1:]) / 2.0
    return float(info.center_x), float(z_centers[rows - info.row])


def bucket_indices(bucket_id: str):
    info = BUCKET_LAYOUT.get(str(bucket_id))
    if info is None:
        return 3, 3
    return info.row, info.col


#map pitch location to bucket
def locate_bucket_id(
    plate_x: object,
    plate_z: object,
    *,
    batter_height_ft: object | None = None,
    zone_bottom: object | None = None,
    zone_top: object | None = None):

    x_value = _coerce_coordinate(plate_x, default=0.0)
    default_bottom, default_top = strike_zone_bounds(
        batter_height_ft=batter_height_ft,
        zone_bottom=zone_bottom,
        zone_top=zone_top,
    )
    z_value = _coerce_coordinate(plate_z, default=(default_bottom + default_top) / 2.0)
    clipped_x = float(np.clip(x_value, x_range[0] + 1e-8, x_range[1] - 1e-8))
    vertical_edges = grid_vertical_edges(
        batter_height_ft=batter_height_ft,
        zone_bottom=zone_bottom,
        zone_top=zone_top,
    )
    clipped_z = float(np.clip(z_value, vertical_edges[0] + 1e-8, vertical_edges[-1] - 1e-8))
    col_idx = int(np.searchsorted(x_edges, clipped_x, side="right") - 1)
    row_idx = int(np.searchsorted(vertical_edges, clipped_z, side="right") - 1)
    row_idx = rows - 1 - row_idx
    row_idx = int(np.clip(row_idx, 0, rows - 1))
    col_idx = int(np.clip(col_idx, 0, cols - 1))
    return bucket_id_for_indices(row_idx + 1, col_idx + 1)


def locate_bucket_series(
    plate_x: pd.Series,
    plate_z: pd.Series,
    *,
    batter_height_ft: pd.Series | object | None = None,
    zone_bottom: pd.Series | object | None = None,
    zone_top: pd.Series | object | None = None):

    heights = _aligned_series(batter_height_ft, plate_x.index)
    bottoms = _aligned_series(zone_bottom, plate_x.index)
    tops = _aligned_series(zone_top, plate_x.index)
    return pd.Series(
        [
            locate_bucket_id(
                x,
                z,
                batter_height_ft=height,
                zone_bottom=bottom,
                zone_top=top,
            )
            for x, z, height, bottom, top in zip(
                plate_x.tolist(),
                plate_z.tolist(),
                heights.tolist(),
                bottoms.tolist(),
                tops.tolist(),
                strict=False,
            )
        ],
        index=plate_x.index,
        dtype=object,
    )


def bucket_center_frame(
    bucket_ids: pd.Series,
    *,
    batter_height_ft: pd.Series | object | None = None,
    zone_bottom: pd.Series | object | None = None,
    zone_top: pd.Series | object | None = None):

    rows = []
    heights = _aligned_series(batter_height_ft, bucket_ids.index)
    bottoms = _aligned_series(zone_bottom, bucket_ids.index)
    tops = _aligned_series(zone_top, bucket_ids.index)
    for bucket_id, height, bottom, top in zip(
        bucket_ids.fillna(default_id).astype(str).tolist(),
        heights.tolist(),
        bottoms.tolist(),
        tops.tolist(),
        strict=False,
    ):
        row_idx, col_idx = bucket_indices(bucket_id)
        center_x, center_z = bucket_center(
            bucket_id,
            batter_height_ft=height,
            zone_bottom=bottom,
            zone_top=top,
        )
        rows.append(
            {
                "bucket_center_x": center_x,
                "bucket_center_z": center_z,
                "bucket_row": row_idx,
                "bucket_col": col_idx,
            }
        )
    return pd.DataFrame(rows, index=bucket_ids.index)


def default_bucket_prior_rows():
    
    ranked_bucket_ids = sorted(
        bucket_ids,
        key=lambda bucket_id: (
            abs(bucket_indices(bucket_id)[0] - 3) + abs(bucket_indices(bucket_id)[1] - 3),
            abs(bucket_indices(bucket_id)[0] - 3),
            abs(bucket_indices(bucket_id)[1] - 3),
            bucket_id,
        ),
    )
    rows = []
    for rank, bucket_id in enumerate(ranked_bucket_ids, start=1):
        center_x, center_z = bucket_center(bucket_id, batter_height_ft=default_batter_height_ft)
        rows.append(
            {
                "pitch_2_bucket": bucket_id,
                "bucket_center_x": center_x,
                "bucket_center_z": center_z,
                "rank": rank,
                "weight": 0.0,
                "n_obs": 0,
            }
        )
    return pd.DataFrame.from_records(rows)
