"""Shared fixtures for the ``eruption-forecast`` test suite.

Every fixture here is synthetic and in-memory (or written under ``tmp_path``)
so the suite never depends on a real SDS archive, a network connection, or
files left behind by a previous run.

Fixtures:
    synthetic_tremor_df: 10-minute tremor frame with every default column.
    synthetic_tremor_csv: The same frame written as a tremor CSV.
    synthetic_stream: One-day ObsPy ``Stream`` carrying a known sinusoid.
    make_seed_ensemble: Factory building a fitted ``SeedEnsemble``.
    features_and_labels: Small id-indexed ``(X, y)`` pair.
    tiny_grids: Module-scoped one-point hyperparameter grids for fast fits.

Helpers:
    build_stream: Build a synthetic ``Stream`` with a known sinusoid.
    eruption_ramp_tremor_df: Tremor frame that ramps up before an eruption.
    seed_tremor_csv: Write the merged tremor CSV ``ForecastModel.calculate``
        reuses, so pipeline tests skip waveform processing.
"""

import os
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import matplotlib
from obspy import Trace, Stream, UTCDateTime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from eruption_forecast.ensemble.seed_ensemble import SeedEnsemble
from eruption_forecast.model.classifier_model import ClassifierModel


# Force a non-interactive backend before any test imports ``pyplot`` so plot
# code never tries to open a window on a developer machine or CI runner.
matplotlib.use("Agg")


TREMOR_COLUMNS: list[str] = [
    "rsam_f0",
    "rsam_f1",
    "rsam_f2",
    "rsam_f3",
    "rsam_f4",
    "dsar_f0-f1",
    "dsar_f1-f2",
    "dsar_f2-f3",
    "dsar_f3-f4",
    "entropy",
]

FEATURE_NAMES: list[str] = [f"feat_{i}" for i in range(4)]

STREAM_SAMPLING_RATE: float = 20.0
STREAM_SIGNAL_FREQUENCY: float = 3.0
STREAM_AMPLITUDE: float = 1000.0


@pytest.fixture()
def synthetic_tremor_df() -> pd.DataFrame:
    """Build a ten-day, 10-minute tremor frame with every default column.

    Values are strictly positive and carry a gentle upward ramp so that
    downstream code which divides, logs, or z-scores the data behaves.

    Returns:
        pd.DataFrame: ``DatetimeIndex`` named ``datetime`` spanning
            ``2025-01-01 00:00`` to ``2025-01-10 23:50`` with the
            ``rsam_*``, ``dsar_*`` and ``entropy`` columns.
    """
    index = pd.date_range("2025-01-01", "2025-01-10 23:50", freq="10min")
    index.name = "datetime"
    rng = np.random.default_rng(0)
    ramp = np.linspace(1.0, 2.0, len(index))
    data = {
        column: (i + 1) * 10.0 * ramp + rng.uniform(0.0, 1.0, len(index))
        for i, column in enumerate(TREMOR_COLUMNS)
    }
    return pd.DataFrame(data, index=index)


@pytest.fixture()
def synthetic_tremor_csv(tmp_path, synthetic_tremor_df: pd.DataFrame) -> str:
    """Write ``synthetic_tremor_df`` to disk in the tremor CSV format.

    Args:
        tmp_path: pytest-provided temporary directory.
        synthetic_tremor_df (pd.DataFrame): Frame to persist.

    Returns:
        str: Path to the written CSV.
    """
    path = tmp_path / "VG.OJN.00.EHZ_2025-01-01_2025-01-10.csv"
    synthetic_tremor_df.to_csv(path, index=True)
    return str(path)


def build_stream(
    starttime: str = "2025-01-01T00:00:00",
    frequency: float = STREAM_SIGNAL_FREQUENCY,
    amplitude: float = STREAM_AMPLITUDE,
    sampling_rate: float = STREAM_SAMPLING_RATE,
    duration_seconds: int = 86400,
    noise: float = 0.0,
    seed: int = 0,
) -> Stream:
    """Build a single-trace ``VG.OJN.00.EHZ`` stream holding a pure sinusoid.

    Args:
        starttime (str): ISO start time. Defaults to ``2025-01-01T00:00:00``.
        frequency (float): Sinusoid frequency in Hz. Defaults to ``3.0``.
        amplitude (float): Sinusoid peak amplitude. Defaults to ``1000.0``.
        sampling_rate (float): Samples per second. Defaults to ``20.0``.
        duration_seconds (int): Trace length in seconds. Defaults to one day.
        noise (float): Standard deviation of additive Gaussian noise.
            Defaults to ``0.0``.
        seed (int): Random seed for the noise. Defaults to ``0``.

    Returns:
        Stream: Stream holding one ``float64`` trace.
    """
    npts = int(duration_seconds * sampling_rate)
    t = np.arange(npts) / sampling_rate
    data = amplitude * np.sin(2 * np.pi * frequency * t)
    if noise:
        data = data + np.random.default_rng(seed).normal(0.0, noise, npts)
    trace = Trace(
        data=data.astype(np.float64),
        header={
            "network": "VG",
            "station": "OJN",
            "location": "00",
            "channel": "EHZ",
            "sampling_rate": sampling_rate,
            "starttime": UTCDateTime(starttime),
        },
    )
    return Stream([trace])


@pytest.fixture()
def synthetic_stream() -> Stream:
    """Return a one-day 20 Hz stream carrying a 3 Hz, 1000-count sinusoid.

    Returns:
        Stream: See :func:`build_stream` for the exact construction.
    """
    return build_stream()


@pytest.fixture()
def features_and_labels() -> tuple[pd.DataFrame, pd.Series]:
    """Return a small, separable, id-indexed ``(X, y)`` pair.

    ``feat_0`` is shifted for the positive class so tree and linear models
    learn a non-trivial decision boundary.

    Returns:
        tuple[pd.DataFrame, pd.Series]: 60 rows, 4 features, 15 positives.
    """
    rng = np.random.default_rng(1)
    n_samples = 60
    y = np.zeros(n_samples, dtype=int)
    y[-15:] = 1
    X = rng.standard_normal((n_samples, len(FEATURE_NAMES)))
    X[:, 0] += y * 3.0
    index = pd.RangeIndex(n_samples, name="id")
    features = pd.DataFrame(X, columns=FEATURE_NAMES, index=index)
    labels = pd.Series(y, index=index, name="is_erupted")
    return features, labels


@pytest.fixture()
def make_seed_ensemble(
    features_and_labels: tuple[pd.DataFrame, pd.Series],
) -> Callable[..., SeedEnsemble]:
    """Return a factory that builds a fitted ``SeedEnsemble`` in memory.

    The factory fits one model per seed on ``features_and_labels`` using a
    DataFrame, so ``predict_proba`` on DataFrames never warns about missing
    feature names.

    Args:
        features_and_labels (tuple[pd.DataFrame, pd.Series]): Training data.

    Returns:
        Callable[..., SeedEnsemble]: ``factory(n_seeds=3, kind="lr",
            seed_offset=0, feature_names=None)``. ``kind`` is ``"lr"`` for
            ``LogisticRegression`` or ``"rf"`` for a tiny
            ``RandomForestClassifier``.
    """
    features, labels = features_and_labels

    def factory(
        n_seeds: int = 3,
        kind: str = "lr",
        seed_offset: int = 0,
        feature_names: list[str] | None = None,
    ) -> SeedEnsemble:
        names = feature_names or FEATURE_NAMES
        classifier_name = (
            "RandomForestClassifier" if kind == "rf" else "LogisticRegression"
        )
        ensemble = SeedEnsemble(classifier_name=classifier_name)
        for i in range(n_seeds):
            random_state = i + seed_offset
            if kind == "rf":
                model = RandomForestClassifier(
                    n_estimators=5, max_depth=3, random_state=random_state
                )
            else:
                model = LogisticRegression(max_iter=200, random_state=random_state)
            model.fit(features[names], labels)
            ensemble.seeds.append(
                {
                    "random_state": random_state,
                    "model": model,
                    "feature_names": names,
                }
            )
        ensemble.features = sorted(names)
        return ensemble

    return factory


# One grid point per classifier turns each GridSearchCV into a single fit, so
# a full train -> predict -> evaluate -> explain chain finishes in about a
# minute instead of tens of minutes with the production grids.
TINY_GRIDS: dict[str, dict[str, list]] = {
    "lite-rf": {"n_estimators": [5], "max_depth": [3]},
    "rf": {"n_estimators": [5], "max_depth": [3]},
    "nb": {"var_smoothing": [1e-9]},
}


@pytest.fixture(scope="module")
def tiny_grids():
    """Swap the production hyperparameter grids for one-point grids.

    Module-scoped so it can back module-scoped pipeline fixtures. The
    registry is restored when the module finishes.

    Yields:
        dict[str, dict[str, list]]: The grids that were installed.
    """
    with pytest.MonkeyPatch.context() as mp:
        for name, grid in TINY_GRIDS.items():
            mp.setitem(ClassifierModel._GRID_REGISTRY, name, grid)
        yield TINY_GRIDS


def eruption_ramp_tremor_df(
    start: str = "2025-01-01",
    end: str = "2025-01-25",
    eruption_date: str = "2025-01-20",
    columns: tuple[str, ...] = ("rsam_f2", "entropy"),
    seed: int = 0,
) -> pd.DataFrame:
    """Build a 10-minute tremor frame whose first column ramps up pre-eruption.

    The first column climbs linearly over the three days before
    ``eruption_date`` and drops back afterwards; every other column is pure
    noise. That gives tsfresh a real signal to select and the classifiers
    something to learn.

    Args:
        start (str): First day (inclusive). Defaults to ``"2025-01-01"``.
        end (str): Last day (inclusive). Defaults to ``"2025-01-25"``.
        eruption_date (str): Eruption day. Defaults to ``"2025-01-20"``.
        columns (tuple[str, ...]): Tremor columns to create. Defaults to
            ``("rsam_f2", "entropy")``.
        seed (int): Noise seed. Defaults to ``0``.

    Returns:
        pd.DataFrame: ``DatetimeIndex`` named ``datetime``.
    """
    index = pd.date_range(start, f"{end} 23:50", freq="10min", name="datetime")
    eruption = pd.Timestamp(eruption_date)
    days_before = (eruption - index) / pd.Timedelta("3D")
    ramp = np.clip(1 - days_before, 0, None) * (index < eruption)
    rng = np.random.default_rng(seed)
    data = {
        column: 10 + rng.normal(0, 1, len(index)) + (20 * ramp if i == 0 else 0)
        for i, column in enumerate(columns)
    }
    return pd.DataFrame(data, index=index)


def seed_tremor_csv(
    output_dir: str,
    df: pd.DataFrame,
    nslc: str = "VG.OJN.00.EHZ",
) -> str:
    """Write ``df`` where ``CalculateTremor.run()`` looks for its merged CSV.

    With ``overwrite=False``, ``CalculateTremor.run()`` loads an existing
    ``tremor-interpolated_{nslc}_{start}-{end}.csv`` instead of reading
    waveforms, so pipeline tests can start from synthetic tremor.

    Args:
        output_dir (str): ``ForecastModel`` output directory.
        df (pd.DataFrame): Tremor frame to persist.
        nslc (str): Station identifier. Defaults to ``"VG.OJN.00.EHZ"``.

    Returns:
        str: Path to the written CSV.
    """
    start = df.index.min().strftime("%Y-%m-%d")
    end = df.index.max().strftime("%Y-%m-%d")
    tremor_dir = os.path.join(output_dir, nslc, "tremor")
    os.makedirs(tremor_dir, exist_ok=True)
    path = os.path.join(tremor_dir, f"tremor-interpolated_{nslc}_{start}-{end}.csv")
    df.to_csv(path, index=True)
    return path
