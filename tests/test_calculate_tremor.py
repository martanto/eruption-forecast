"""Tests for ``CalculateTremor``.

A two-day synthetic SDS archive (40 Hz so every default band sits below
Nyquist) is written once per module. The full RSAM + DSAR + entropy run then
takes a few seconds; the cheaper checks reuse the same archive.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from tests.conftest import TREMOR_COLUMNS, STREAM_AMPLITUDE, build_stream
from eruption_forecast.sources.sds import SDS
from eruption_forecast.tremor.calculate_tremor import CalculateTremor


SAMPLING_RATE = 40.0
DAYS = ["2025-01-01", "2025-01-02"]


@pytest.fixture(scope="module")
def sds_dir(tmp_path_factory) -> str:
    """Write a two-day 3 Hz miniSEED archive and return its root."""
    root = tmp_path_factory.mktemp("sds")
    sds = SDS(str(root), station="OJN", channel="EHZ", network="VG", location="00")
    for day in DAYS:
        stream = build_stream(
            starttime=f"{day}T00:00:00", sampling_rate=SAMPLING_RATE, noise=50.0
        )
        sds.save(stream, datetime.fromisoformat(day))
    return str(root)


def _calculator(output_dir, **kwargs) -> CalculateTremor:
    params = {
        "start_date": DAYS[0],
        "end_date": DAYS[-1],
        "station": "OJN",
        "channel": "EHZ",
        "network": "VG",
        "location": "00",
        "output_dir": str(output_dir),
        **kwargs,
    }
    return CalculateTremor(**params)


@pytest.fixture(scope="module")
def full_run(sds_dir, tmp_path_factory) -> CalculateTremor:
    """Run every metric once over the archive."""
    output_dir = tmp_path_factory.mktemp("tremor-out")
    return _calculator(output_dir, overwrite=True).from_sds(sds_dir).run()


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------


class TestRun:
    def test_columns_and_rows(self, full_run: CalculateTremor) -> None:
        assert sorted(full_run.df.columns) == sorted(TREMOR_COLUMNS)
        assert len(full_run.df) == 144 * len(DAYS)
        assert full_run.df.index.min() == pd.Timestamp("2025-01-01 00:00")
        assert full_run.df.index.max() == pd.Timestamp("2025-01-02 23:50")

    def test_no_missing_values(self, full_run: CalculateTremor) -> None:
        assert not full_run.df.isna().any().any()

    def test_signal_band_dominates(self, full_run: CalculateTremor) -> None:
        # The 3 Hz signal lives in f2 (2-5 Hz); every other band only sees noise.
        rsam = full_run.df[[f"rsam_f{i}" for i in range(5)]].median()
        assert rsam.idxmax() == "rsam_f2"
        assert rsam["rsam_f2"] == pytest.approx(2 * STREAM_AMPLITUDE / np.pi, rel=0.05)

    def test_writes_interpolated_and_raw_csvs(self, full_run: CalculateTremor) -> None:
        files = os.listdir(full_run.tremor_dir)
        assert os.path.isfile(full_run.csv)
        assert any(f.startswith("tremor-interpolated_") for f in files)
        assert any(f.startswith("tremor-non-interpolated_") for f in files)

    def test_csv_round_trips(self, full_run: CalculateTremor) -> None:
        df = pd.read_csv(full_run.csv, index_col=0, parse_dates=True)
        pd.testing.assert_frame_equal(
            df[sorted(df.columns)],
            full_run.df[sorted(full_run.df.columns)],
            check_freq=False,
            check_names=False,
            check_exact=False,
        )


class TestRunVariants:
    def test_methods_subset(self, sds_dir, tmp_path) -> None:
        ct = _calculator(tmp_path, methods=["rsam"], overwrite=True).from_sds(sds_dir).run()
        assert ct.df.columns.tolist() == [f"rsam_f{i}" for i in range(5)]

    def test_missing_day_is_dropped(self, sds_dir, tmp_path) -> None:
        ct = _calculator(
            tmp_path, end_date="2025-01-03", methods=["rsam"], overwrite=True
        ).from_sds(sds_dir).run()
        assert ct.df.index.max() == pd.Timestamp("2025-01-02 23:50")

    def test_rerun_produces_identical_data(self, sds_dir, tmp_path) -> None:
        first = _calculator(tmp_path, methods=["rsam"], overwrite=True).from_sds(sds_dir).run()
        second = _calculator(tmp_path, methods=["rsam"]).from_sds(sds_dir).run()
        pd.testing.assert_frame_equal(second.df, first.df, check_freq=False, check_exact=False)

    def test_rerun_loads_merged_csv_without_rewriting(self, sds_dir, tmp_path) -> None:
        first = _calculator(tmp_path, methods=["rsam"], overwrite=True).from_sds(sds_dir).run()
        mtime = os.path.getmtime(first.csv)

        second = _calculator(tmp_path, methods=["rsam"]).from_sds(sds_dir)
        assert second.csv == first.csv
        second.run()

        assert os.path.getmtime(first.csv) == mtime
        # Only the merged CSV is read back; the daily pipeline never runs.
        assert second.daily_files == []
        assert len(second.df) == len(first.df)

    def test_custom_freq_bands(self, sds_dir, tmp_path) -> None:
        ct = (
            _calculator(tmp_path, methods=["rsam"], overwrite=True)
            .from_sds(sds_dir)
            .change_freq_bands([(1.0, 2.5), (2.5, 4.0)])
            .run()
        )
        assert ct.df.columns.tolist() == ["rsam_f0", "rsam_f1"]
        assert ct.df["rsam_f1"].median() > ct.df["rsam_f0"].median()


# ---------------------------------------------------------------------------
# Configuration / validation
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_paths_and_defaults(self, tmp_path) -> None:
        ct = _calculator(tmp_path)
        assert ct.nslc == "VG.OJN.00.EHZ"
        assert ct.station_dir == os.path.join(str(tmp_path), "VG.OJN.00.EHZ")
        assert ct.methods == ["rsam", "dsar", "entropy"]
        assert ct.n_days == 2
        assert os.path.isdir(ct.daily_dir)
        assert ct.filename.startswith("tremor-interpolated_")

    def test_buffer_days_extends_dates(self, tmp_path) -> None:
        ct = _calculator(tmp_path, buffer_days=2)
        assert ct.start_date_buffered == datetime(2024, 12, 30)
        assert ct.n_days == 4

    def test_filename_prefix(self, tmp_path) -> None:
        assert _calculator(tmp_path, filename_prefix="custom").filename.startswith("custom_")

    def test_freq_bands_alias(self, tmp_path) -> None:
        aliases = _calculator(tmp_path).freq_bands_alias
        assert list(aliases) == ["f0", "f1", "f2", "f3", "f4"]
        assert aliases["f2"] == (2, 5)

    @pytest.mark.parametrize(
        ("bands", "error"),
        [
            ([(5.0, 1.0)], ValueError),
            ([(1.0, 1.0)], ValueError),
            ([("a", 1.0)], TypeError),
            ([(1.0, None)], TypeError),
        ],
    )
    def test_change_freq_bands_validation(self, tmp_path, bands, error) -> None:
        with pytest.raises(error):
            _calculator(tmp_path).change_freq_bands(bands)

    def test_zero_jobs_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            _calculator(tmp_path, n_jobs=0)

    def test_start_not_before_end_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            _calculator(tmp_path, start_date="2025-01-02", end_date="2025-01-02")

    def test_unknown_method_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            _calculator(tmp_path, methods=["kurtosis"])

    def test_get_stream_without_source_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            _calculator(tmp_path).get_stream(datetime(2025, 1, 1))

    def test_from_sds_missing_dir_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            _calculator(tmp_path).from_sds(str(tmp_path / "missing"))

    def test_get_stream_reads_sds(self, sds_dir, tmp_path) -> None:
        stream = _calculator(tmp_path).from_sds(sds_dir).get_stream(datetime(2025, 1, 1))
        assert len(stream) == 1
        assert stream[0].stats.sampling_rate == SAMPLING_RATE
