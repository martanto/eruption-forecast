"""Tests for the content-addressable cache layer on ``BaseModel``.

``BaseModel`` is abstract, so these tests drive it through ``_StubModel``, a
minimal concrete subclass defined at module level so joblib can pickle it.
Everything runs against ``tmp_path`` with a synthetic tremor frame.
"""

import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from eruption_forecast.model.base_model import BaseModel


class _StubModel(BaseModel):
    """Smallest concrete ``BaseModel`` that satisfies the abstract surface."""

    def set_directories(self) -> tuple:
        return (self.output_dir,)

    def create_directories(self) -> None:
        for path in self.set_directories():
            os.makedirs(path, exist_ok=True)

    def describe(self) -> str:
        return "stub"

    def to_dict(self) -> dict:
        return {"start_date": self.start_date_str, "end_date": self.end_date_str}

    def to_prompt(self) -> str:
        return "- stub"

    def save_config(self, path=None, fmt="yaml") -> str:
        return path or ""


@pytest.fixture()
def stub_model(tmp_path, synthetic_tremor_df: pd.DataFrame) -> _StubModel:
    """Return a ``_StubModel`` writing under ``tmp_path``."""
    return _StubModel(
        tremor_data=synthetic_tremor_df,
        start_date="2025-01-02",
        end_date="2025-01-08",
        output_dir=str(tmp_path / "output"),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_derived_fields(self, stub_model: _StubModel) -> None:
        assert stub_model.start_date_str == "2025-01-02"
        assert stub_model.end_date_str == "2025-01-08"
        assert stub_model.n_days == 7
        assert stub_model.total_cpu >= 1
        assert stub_model.id is None

    def test_eruption_dates_are_sorted(self, synthetic_tremor_df) -> None:
        model = _StubModel(
            tremor_data=synthetic_tremor_df,
            start_date="2025-01-02",
            end_date="2025-01-08",
            eruption_dates=["2025-01-07", "2025-01-03"],
        )
        assert model.eruption_dates == ["2025-01-03", "2025-01-07"]

    def test_start_after_end_raises(self, synthetic_tremor_df) -> None:
        with pytest.raises(ValueError):
            _StubModel(
                tremor_data=synthetic_tremor_df,
                start_date="2025-01-08",
                end_date="2025-01-02",
            )

    def test_n_jobs_clamped_to_total_cpu(self, synthetic_tremor_df) -> None:
        model = _StubModel(
            tremor_data=synthetic_tremor_df,
            start_date="2025-01-02",
            end_date="2025-01-08",
            n_jobs=10_000,
        )
        assert model.n_jobs == model.total_cpu

    def test_tremor_data_from_csv_path(self, synthetic_tremor_csv: str) -> None:
        model = _StubModel(
            tremor_data=synthetic_tremor_csv,
            start_date="2025-01-02",
            end_date="2025-01-08",
        )
        assert isinstance(model.tremor_df.index, pd.DatetimeIndex)
        assert "rsam_f2" in model.tremor_df.columns

    def test_tremor_data_invalid_type_raises(self) -> None:
        model = _StubModel(
            tremor_data=123,  # type: ignore[arg-type]
            start_date="2025-01-02",
            end_date="2025-01-08",
        )
        with pytest.raises(TypeError):
            _ = model.tremor_data

    def test_sync_dates_clamps_to_tremor_range(self, synthetic_tremor_df) -> None:
        model = _StubModel(
            tremor_data=synthetic_tremor_df,
            start_date="2024-12-01",
            end_date="2025-02-01",
        )
        model._sync_dates_to_tremor()
        assert model.start_date >= synthetic_tremor_df.index.min()
        assert model.end_date <= synthetic_tremor_df.index.max()

    def test_build_identity_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            _StubModel.build_identity(foo=1)


# ---------------------------------------------------------------------------
# _canonicalize
# ---------------------------------------------------------------------------


class TestCanonicalize:
    @pytest.mark.parametrize("value", [None, "abc", True, False, 3, 2.5])
    def test_scalars_pass_through(self, value) -> None:
        assert BaseModel._canonicalize(value) == value

    def test_datetime_and_timestamp_to_isoformat(self) -> None:
        dt = datetime(2025, 1, 2, 3, 4, 5)
        assert BaseModel._canonicalize(dt) == "2025-01-02T03:04:05"
        assert BaseModel._canonicalize(pd.Timestamp(dt)) == "2025-01-02T03:04:05"

    def test_path_to_posix(self) -> None:
        assert BaseModel._canonicalize(Path("a") / "b" / "c.csv") == "a/b/c.csv"

    def test_numpy_scalars_become_python(self) -> None:
        assert type(BaseModel._canonicalize(np.int64(3))) is int
        assert type(BaseModel._canonicalize(np.float32(1.5))) is float

    def test_ndarray_to_list(self) -> None:
        assert BaseModel._canonicalize(np.array([1, 2, 3])) == [1, 2, 3]

    def test_dict_keys_sorted_and_stringified(self) -> None:
        result = BaseModel._canonicalize({"b": 1, "a": 2, 3: "x"})
        assert list(result) == ["3", "a", "b"]

    def test_tuple_and_list_are_equivalent(self) -> None:
        assert BaseModel._canonicalize((1, 2)) == BaseModel._canonicalize([1, 2])

    def test_set_is_order_independent(self) -> None:
        assert BaseModel._canonicalize({"rf", "xgb", "gb"}) == ["gb", "rf", "xgb"]

    def test_nested_structures(self) -> None:
        value = {"dates": (datetime(2025, 1, 1),), "inner": {"n": np.int32(2)}}
        assert BaseModel._canonicalize(value) == {
            "dates": ["2025-01-01T00:00:00"],
            "inner": {"n": 2},
        }

    def test_unknown_object_falls_back_to_str(self) -> None:
        class Thing:
            def __str__(self) -> str:
                return "thing"

        assert BaseModel._canonicalize(Thing()) == "thing"


# ---------------------------------------------------------------------------
# compute_hash / cache_path
# ---------------------------------------------------------------------------


class TestComputeHash:
    def test_stable_across_calls(self) -> None:
        identity = {"a": 1, "b": [1, 2]}
        assert BaseModel.compute_hash(identity) == BaseModel.compute_hash(identity)

    def test_key_order_independent(self) -> None:
        assert BaseModel.compute_hash({"a": 1, "b": 2}) == BaseModel.compute_hash(
            {"b": 2, "a": 1}
        )

    def test_value_change_changes_hash(self) -> None:
        assert BaseModel.compute_hash({"a": 1}) != BaseModel.compute_hash({"a": 2})

    def test_list_order_matters(self) -> None:
        # Lists are positional (e.g. classifier priority), unlike sets.
        assert BaseModel.compute_hash({"c": ["rf", "xgb"]}) != BaseModel.compute_hash(
            {"c": ["xgb", "rf"]}
        )

    def test_numpy_and_python_values_hash_equal(self) -> None:
        assert BaseModel.compute_hash({"n": np.int64(5)}) == BaseModel.compute_hash(
            {"n": 5}
        )

    @pytest.mark.parametrize("length", [8, 12, 64])
    def test_length(self, length: int) -> None:
        digest = BaseModel.compute_hash({"a": 1}, length=length)
        assert len(digest) == length
        int(digest, 16)  # Must be valid hex.

    def test_cache_path_format(self, tmp_path) -> None:
        identity = {"a": 1}
        path = _StubModel.cache_path(str(tmp_path), identity)
        expected_name = f"{_StubModel.compute_hash(identity)}._StubModel.pkl"
        assert path == os.path.join(str(tmp_path), expected_name)


# ---------------------------------------------------------------------------
# tremor_fingerprint
# ---------------------------------------------------------------------------


class TestTremorFingerprint:
    def test_empty_frame(self) -> None:
        assert BaseModel.tremor_fingerprint(pd.DataFrame()) == {
            "start": None,
            "end": None,
            "n_rows": 0,
            "columns": [],
        }

    def test_fields(self, synthetic_tremor_df: pd.DataFrame) -> None:
        fingerprint = BaseModel.tremor_fingerprint(synthetic_tremor_df)
        assert fingerprint["start"] == "2025-01-01T00:00:00"
        assert fingerprint["end"] == "2025-01-10T23:50:00"
        assert fingerprint["n_rows"] == len(synthetic_tremor_df)
        assert fingerprint["columns"] == sorted(synthetic_tremor_df.columns)

    def test_column_order_does_not_matter(self, synthetic_tremor_df) -> None:
        reordered = synthetic_tremor_df[synthetic_tremor_df.columns[::-1]]
        assert BaseModel.tremor_fingerprint(
            reordered
        ) == BaseModel.tremor_fingerprint(synthetic_tremor_df)

    def test_dropping_a_row_changes_fingerprint(self, synthetic_tremor_df) -> None:
        assert BaseModel.tremor_fingerprint(
            synthetic_tremor_df.iloc[:-1]
        ) != BaseModel.tremor_fingerprint(synthetic_tremor_df)

    def test_dropping_a_column_changes_fingerprint(self, synthetic_tremor_df) -> None:
        assert BaseModel.tremor_fingerprint(
            synthetic_tremor_df.drop(columns=["entropy"])
        ) != BaseModel.tremor_fingerprint(synthetic_tremor_df)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_cache_round_trip(self, stub_model: _StubModel) -> None:
        identity = {"start": "2025-01-02", "classifiers": ["rf"]}
        path = stub_model.save(identity=identity)

        assert path == _StubModel.cache_path(stub_model.stage_dir, identity)
        assert os.path.isfile(path)

        restored = _StubModel.load(stub_model.stage_dir, identity)
        assert isinstance(restored, _StubModel)
        assert restored.start_date_str == stub_model.start_date_str

    def test_cache_writes_params_sidecar(self, stub_model: _StubModel) -> None:
        identity = {"b": np.int64(2), "a": datetime(2025, 1, 1)}
        path = stub_model.save(identity=identity)
        sidecar = path[: -len(".pkl")] + ".params.json"

        with open(sidecar, encoding="utf-8") as f:
            assert json.load(f) == {"a": "2025-01-01T00:00:00", "b": 2}

    def test_load_miss_returns_none(self, stub_model: _StubModel) -> None:
        stub_model.save(identity={"a": 1})
        assert _StubModel.load(stub_model.stage_dir, {"a": 2}) is None

    def test_load_truncated_pickle_returns_none(self, stub_model: _StubModel) -> None:
        # Simulates a run killed mid-write: the cache must fall through to a
        # rebuild instead of crashing.
        identity = {"a": 1}
        path = stub_model.save(identity=identity)
        with open(path, "rb") as f:
            payload = f.read()
        with open(path, "wb") as f:
            f.write(payload[: len(payload) // 2])

        assert _StubModel.load(stub_model.stage_dir, identity) is None

    def test_load_garbage_bytes_returns_none(self, stub_model: _StubModel) -> None:
        identity = {"a": 1}
        path = _StubModel.cache_path(stub_model.stage_dir, identity)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"not a pickle")

        assert _StubModel.load(stub_model.stage_dir, identity) is None

    def test_legacy_save_default_path(self, stub_model: _StubModel) -> None:
        path = stub_model.save()
        assert path == os.path.join(stub_model.output_dir, "_StubModel.pkl")
        assert os.path.isfile(path)

    def test_legacy_save_uses_basename(self, stub_model: _StubModel) -> None:
        stub_model.basename = "2025-01-02_2025-01-08"
        path = stub_model.save()
        assert os.path.basename(path) == "_StubModel_2025-01-02_2025-01-08.pkl"

    def test_legacy_save_explicit_path(self, stub_model: _StubModel, tmp_path) -> None:
        target = tmp_path / "nested" / "model.pkl"
        assert stub_model.save(path=str(target)) == str(target)
        assert target.is_file()
