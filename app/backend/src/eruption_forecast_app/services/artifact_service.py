"""Discover and preview artifacts without loading models."""

import json
import math
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from eruption_forecast_app.repositories.database import Database


EXTENSIONS = {
    ".csv",
    ".parquet",
    ".png",
    ".pdf",
    ".json",
    ".yaml",
    ".yml",
    ".pkl",
    ".txt",
}


def discover(directory: Path, db: Database, job_id: str | None = None) -> list[dict]:
    """Index artifacts without deserializing executable model files.

    Example:
        >>> discover(Path('output'), database)
    """
    found = []
    for path in directory.rglob("*"):
        if (
            path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in EXTENSIONS
            and path.resolve().is_relative_to(directory.resolve())
        ):
            found.append(db.artifact(path, job_id))
            if len(found) >= 5000:
                break
    return found


def preview(path: Path, limit: int = 200, offset: int = 0, chart: bool = False) -> dict:
    """Read a bounded table preview or downsampled chart.

    Example:
        >>> preview(Path('output/tremor.csv'), limit=20)
    """
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        rows = []
        size = limit if chart else limit + offset
        for batch in parquet.iter_batches(batch_size=size):
            rows.append(batch.to_pandas())
            if sum(len(frame) for frame in rows) >= size:
                break
        df = pd.concat(rows) if rows else pd.DataFrame()
        df = df.head(size) if chart else df.iloc[offset : offset + limit]
        total = parquet.metadata.num_rows
    elif path.suffix == ".csv":
        if chart:
            count = sum(1 for _ in path.open(encoding="utf-8")) - 1
            stride = max(1, math.ceil(count / limit))
            pieces = []
            position = 0
            for chunk in pd.read_csv(path, chunksize=10000):
                pieces.append(
                    chunk.iloc[
                        [i for i in range(len(chunk)) if (position + i) % stride == 0]
                    ]
                )
                position += len(chunk)
            df = pd.concat(pieces).head(limit) if pieces else pd.DataFrame()
            total = count
        else:
            df = pd.read_csv(path, nrows=limit + offset).iloc[offset:]
            total = None
    else:
        raise ValueError("Table previews support CSV and Parquet")
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    df.columns = [str(column) for column in df.columns]
    return {
        "columns": list(df.columns),
        "rows": json.loads(df.to_json(orient="records", date_format="iso") or "[]"),
        "total": total,
        "offset": offset,
        "limit": limit,
        "sampled": chart,
    }
