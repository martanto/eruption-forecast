import os
import json
import hashlib
import multiprocessing
from abc import ABC, abstractmethod
from typing import Any, Self, Literal
from pathlib import Path
from datetime import datetime
from functools import cached_property

import numpy as np
import joblib
import pandas as pd

from eruption_forecast.utils import validate_date_ranges
from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import (
    ensure_dir,
    load_pickle,
    resolve_output_dir,
)
from eruption_forecast.utils.date_utils import sort_dates, to_datetime
from eruption_forecast.tremor.tremor_data import TremorData
from eruption_forecast.features.features_builder import FeaturesBuilder
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble
from eruption_forecast.features.tremor_matrix_builder import TremorMatrixBuilder


class BaseModel(ABC):
    """Abstract base class for eruption forecast model stages.

    Concrete subclasses (``TrainingModel``, ``PredictionModel``,
    ``EvaluationModel``, ``ExplanationModel``) inherit construction,
    parameter validation, tremor-data loading, output-directory resolution,
    content-addressable cache identity hashing, and joblib-based persistence
    from this class. Each subclass implements its own ``set_directories()``,
    ``create_directories()``, ``validate()``, ``describe()``, ``to_dict()``,
    ``to_prompt()``, ``build_label()``, and ``extract_features()``.

    Cache-using subclasses additionally override ``stage_dir`` so cache
    artefacts land next to their other outputs, and ``build_identity`` to
    declare the parameters that uniquely identify a cache entry. Identity
    dicts are canonicalised and SHA-256-hashed so two calls with identical
    parameters resolve to the same on-disk pickle.

    Attributes:
        start_date (datetime): Inclusive start of the modelling period.
        end_date (datetime): Inclusive end of the modelling period.
        start_date_str (str): ``start_date`` formatted as ``"YYYY-MM-DD"``.
        end_date_str (str): ``end_date`` formatted as ``"YYYY-MM-DD"``.
        window_size (int): Sliding window size in days.
        eruption_dates (list[str] | None): Sorted eruption dates in
            ``"YYYY-MM-DD"`` format, or ``None`` when none are supplied.
        overwrite (bool): Whether downstream stages should overwrite
            existing artefacts.
        output_dir (str): Resolved path to the output directory.
        root_dir (str | None): Optional project root used to resolve
            ``output_dir``.
        n_jobs (int): Number of parallel workers, clamped to
            ``total_cpu - 2`` when the caller passes a value at or above
            ``total_cpu``.
        verbose (bool): Emit verbose log messages.
        n_days (int): Inclusive number of days in the modelling period
            (``(end_date - start_date).days + 1``).
        total_cpu (int): Number of logical CPUs on the current machine.
        use_relevant_features (bool): Enable tsfresh relevance filtering
            during feature extraction. Defaults to ``True``.
        features_df (pd.DataFrame): Extracted features. Empty until
            ``extract_features()`` has been called.
        features_path (str | None): Path to the merged features matrix Parquet on disk.
            ``None`` until ``extract_features()`` has been called.
        window_step (int | None): Step size between consecutive windows.
            Set by ``build_label()``.
        window_step_unit (Literal["minutes", "hours"] | None): Unit for
            ``window_step``. Set by ``build_label()``.
        ClassifierEnsemble (ClassifierEnsemble | None): Loaded classifier
            ensemble. ``None`` until populated, typically by
            ``EvaluationModel``.
    """

    def __init__(
        self,
        tremor_data: str | pd.DataFrame,
        start_date: str | datetime,
        end_date: str | datetime,
        window_size: int = 2,
        eruption_dates: list[str] | None = None,
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Initialise the base model.

        Resolves the output directory, stores all configuration on the
        instance, computes derived values (``start_date_str``, ``n_days``,
        ``total_cpu``, etc.), then runs ``_validate()`` to check date order
        and clamp ``n_jobs``. Heavy work (CSV loading, label building,
        feature extraction) is deferred until the corresponding method is
        explicitly called.

        Args:
            tremor_data (str | pd.DataFrame): Path to a tremor CSV file or a
                pre-loaded DataFrame.
            start_date (str | datetime): Inclusive start of the modelling
                period.
            end_date (str | datetime): Inclusive end of the modelling period.
            window_size (int, optional): Sliding window size in days.
                Defaults to ``2``.
            eruption_dates (list[str] | None, optional): Known eruption dates
                in ``"YYYY-MM-DD"`` format. Defaults to ``None``.
            overwrite (bool, optional): Overwrite existing output files when
                ``True``. Defaults to ``False``.
            output_dir (str | None, optional): Output directory path. Resolved
                relative to ``root_dir`` when omitted. Defaults to ``None``.
            root_dir (str | None, optional): Project root directory used to
                resolve ``output_dir``. Defaults to ``None``.
            n_jobs (int, optional): Number of parallel workers. Clamped to
                ``cpu_count - 2`` when the caller passes a value at or above
                ``cpu_count``. Defaults to ``1``.
            verbose (bool, optional): Emit verbose log messages when ``True``.
                Defaults to ``False``.

        Raises:
            ValueError: If ``start_date >= end_date`` (raised by
                ``validate_date_ranges`` inside ``_validate()``).

        Example:
            >>> # Concrete subclasses call super().__init__ with these args:
            >>> model = TrainingModel(
            ...     tremor_data="output/VG.OJN.00.EHZ/tremor/tremor.csv",
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ... )
        """

        # Set properties
        output_dir = resolve_output_dir(
            output_dir=output_dir,
            root_dir=root_dir,
            default_subpath="output",
        )

        # Set default properties
        self._tremor_data = tremor_data
        self.start_date = to_datetime(start_date)
        self.end_date = to_datetime(end_date)
        self.window_size = window_size
        self.eruption_dates = sort_dates(eruption_dates) if eruption_dates else None
        self.overwrite = overwrite
        self.output_dir = output_dir
        self.root_dir = root_dir
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Set additional properties
        self.start_date_str = self.start_date.strftime("%Y-%m-%d")
        self.end_date_str = self.end_date.strftime("%Y-%m-%d")
        self.use_relevant_features: bool = True
        self.n_days = (self.end_date - self.start_date).days + 1
        self.total_cpu = multiprocessing.cpu_count()

        # WIll be set after extract_features() called
        self.features_df: pd.DataFrame = pd.DataFrame()
        self.features_path: str | None = None

        # Will be set after build_label() called
        self.window_step: int | None = None
        self.window_step_unit: Literal["minutes", "hours"] | None = None
        self.basename: str | None = None

        # Use for evaluation model
        self.ClassifierEnsemble: ClassifierEnsemble | None = None

        self._validate()

    def _validate(self) -> Self:
        """Validate base model parameters and clamp ``n_jobs`` to a safe range.

        Delegates date-range validation to ``validate_date_ranges`` and
        clamps ``n_jobs`` to ``total_cpu - 2`` when the caller passes a value
        at or above the logical CPU count, leaving at least two cores free
        for the OS and other processes.

        Returns:
            Self: The current instance, for method chaining.

        Raises:
            ValueError: If ``start_date >= end_date`` (raised by
                ``validate_date_ranges``).
        """
        validate_date_ranges(self.start_date, self.end_date)

        if self.n_jobs >= self.total_cpu:
            n_jobs = self.total_cpu - 2
            if self.verbose:
                logger.warning(
                    f"Value of n_jobs ({self.n_jobs}) is more than {self.total_cpu} CPU. "
                    f"Update n_jobs to {n_jobs} CPU."
                )
            self.n_jobs = n_jobs

        return self

    def _sync_dates_to_tremor(self) -> None:
        """Clamp ``start_date`` and ``end_date`` to the tremor data range.

        Mutates ``start_date`` and ``end_date`` in place when they fall
        outside the span of the loaded tremor CSV, so downstream labelling
        and feature extraction never reference dates without tremor
        coverage. Called lazily from ``build_label()`` to avoid triggering
        a CSV read at construction time. Emits an INFO-level log message
        for each side that gets adjusted.
        """
        tremor_data: TremorData = self.tremor_data
        tremor_start_date = tremor_data.start_date
        tremor_end_date = tremor_data.end_date
        if self.start_date < tremor_start_date:
            self.start_date = tremor_start_date
            logger.info(
                f"Training start date adjusted to tremor start date: "
                f"{tremor_start_date.strftime('%Y-%m-%d')}"
            )
        if self.end_date > tremor_end_date:
            self.end_date = tremor_end_date
            logger.info(
                f"Training end date adjusted to tremor end date: "
                f"{tremor_end_date.strftime('%Y-%m-%d')}"
            )

    @cached_property
    def tremor_df(self) -> pd.DataFrame:
        """Tremor DataFrame loaded from ``tremor_data``.

        Convenience accessor for ``self.tremor_data.df``. Cached after the
        first access so repeated reads are free.

        Returns:
            pd.DataFrame: Tremor DataFrame with a ``DatetimeIndex`` and the
                usual tremor columns (``rsam_f0``..``rsam_f4``,
                ``dsar_f0-f1``..``dsar_f3-f4``, ``entropy``).
        """
        return self.tremor_data.df

    @cached_property
    def tremor_data(self) -> TremorData:
        """Validated ``TremorData`` instance loaded from the configured source.

        Lazily loads the tremor data from the value stored at
        ``self._tremor_data``: a ``str`` path is read via
        ``TremorData.from_csv()``, while a pre-loaded ``pd.DataFrame`` is
        wrapped directly. Cached after the first access so subsequent reads
        are free.

        Returns:
            TremorData: Validated tremor data container.

        Raises:
            TypeError: If ``_tremor_data`` is neither a ``str`` nor a
                ``pd.DataFrame``.
        """
        if not isinstance(self._tremor_data, str | pd.DataFrame):
            raise TypeError(
                f"tremor_data should have an instance of `str` or `pd.DataFrame` "
                f"instead of {type(self._tremor_data)}"
            )

        if isinstance(self._tremor_data, str):
            return TremorData.from_csv(self._tremor_data)

        return TremorData(self._tremor_data)

    @abstractmethod
    def set_directories(self) -> tuple:
        """Build and return all directory paths needed for this model.

        Subclasses compute their stage-specific subdirectories from
        ``self.output_dir`` and return them as a tuple. Called by
        ``create_directories()`` to materialise the paths on disk.

        Returns:
            tuple: Stage-specific directory paths. The shape and meaning of
                the tuple are defined by the concrete subclass.
        """

    @abstractmethod
    def create_directories(self) -> None:
        """Create the output directories required by this model.

        Subclasses iterate over the paths returned by ``set_directories()``
        and call ``ensure_dir()`` on each one so that downstream pipeline
        stages can write into them without manual setup.
        """

    def validate(self) -> Self:
        """Validate model-specific parameters.

        Concrete subclasses that need stage-specific checks (e.g. that
        ``eruption_dates`` is provided when required, or that a pre-fitted
        ensemble is available before prediction) override this hook. Stages
        whose invariants are already enforced inline in ``__init__`` (such
        as ``ExplanationModel``) inherit this no-op.

        Returns:
            Self: The current instance, for method chaining.

        Raises:
            ValueError: When stage-specific validation fails.
        """
        return self

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable prose description of this instance.

        Suitable for CLI output and human reports. For raw data use
        ``to_dict()``; for an MCP/LLM-friendly block use ``to_prompt()``.

        Returns:
            str: One- to four-sentence description of the model's
                configuration and, when applicable, its build state.
        """

    @abstractmethod
    def to_dict(self) -> dict:
        """Return a structured dictionary of all configuration and derived fields.

        Produces a stable, JSON-serialisable schema suitable for MCP tool
        responses and downstream LLM consumption. Subclasses define the
        exact set of keys.

        Returns:
            dict: Mapping of field name to its serialised value.
        """

    @abstractmethod
    def to_prompt(self) -> str:
        """Return a structured text block suitable for LLM and MCP prompt input.

        Builds a deterministic, template-driven bullet-list block from
        ``to_dict()`` that is stable across calls. Paths are reduced to
        filenames only to avoid leaking absolute system paths into prompts.
        For human-readable prose use ``describe()``. For raw data use
        ``to_dict()``.

        Returns:
            str: Formatted bullet-list block ready for MCP tool responses
                or LLM context.
        """

    @abstractmethod
    def save_config(
        self,
        path: str | None = None,
        fmt: Literal["yaml", "json"] = "yaml",
    ) -> str:
        """Persist the captured init configuration to disk.

        Concrete subclasses snapshot their constructor surface into a
        dedicated config dataclass on ``self._config`` during ``__init__``
        and delegate to that snapshot's ``save(path, fmt)``. The default
        path lives under the subclass's stage directory so configs from
        different stages never collide.

        Args:
            path (str | None): Destination file path. ``None`` resolves to
                the subclass-specific default. Defaults to ``None``.
            fmt (Literal["yaml", "json"]): Output format. Defaults to
                ``"yaml"``.

        Returns:
            str: The absolute path the configuration was written to.
        """

    @property
    def stage_dir(self) -> str:
        """Directory where cache artefacts are written.

        Defaults to ``self.output_dir`` so non-cache stages and the
        legacy-mode save still work. Cache-using subclasses override to point
        at their stage-specific directory (e.g. ``self.training_dir``).

        Returns:
            str: Absolute path used as the destination for cache writes.
        """
        return self.output_dir

    @classmethod
    def build_identity(cls, **kwargs: Any) -> dict:
        """Return the canonical identity dict that defines a cache entry.

        Cache-using subclasses override this to enumerate every parameter that
        materially affects the produced artefact. Runtime knobs that do not
        change output (``n_jobs``, ``n_grids``, ``verbose``, ``overwrite``,
        ``output_dir``, ``root_dir``) must be excluded.

        Called from two places with the same kwargs shape: ``ForecastModel``
        before a stage instance exists (so the cache lookup can short-circuit
        the expensive ``build_label`` / ``extract_features`` chain on a hit),
        and from inside each stage's main method (``fit`` / ``forecast`` /
        ``explain``) once instance state is populated.

        Args:
            **kwargs (Any): Subclass-specific identity inputs. Each subclass
                documents its accepted keyword arguments.

        Returns:
            dict: Canonical, JSON-serialisable identity dict ready for hashing.

        Raises:
            NotImplementedError: When the subclass does not participate in the
                content-addressable cache layer.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement build_identity. "
            "Override this classmethod on cache-using subclasses."
        )

    @classmethod
    def cache_path(cls, stage_dir: str, identity: dict) -> str:
        """Return the absolute ``.pkl`` path for ``identity`` under ``stage_dir``.

        Args:
            stage_dir (str): Directory where the cache file lives.
            identity (dict): Identity dict produced by :meth:`build_identity`.

        Returns:
            str: Path of the form ``{stage_dir}/{hash}.{ClassName}.pkl``.
        """
        return os.path.join(
            stage_dir, f"{cls.compute_hash(identity)}.{cls.__name__}.pkl"
        )

    @classmethod
    def compute_hash(cls, identity: dict, length: int = 12) -> str:
        """Hash the canonical form of ``identity`` to a truncated hex digest.

        Args:
            identity (dict): Identity dict produced by :meth:`build_identity`.
            length (int): Number of hex characters to keep from the SHA-256
                digest. Defaults to ``12`` (~48 bits — ample at this scale).

        Returns:
            str: Hex-encoded hash prefix.
        """
        canon = cls._canonicalize(identity)
        payload = json.dumps(canon, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:length]

    @staticmethod
    def _canonicalize(value: Any) -> Any:
        """Recursively normalise ``value`` into a JSON-stable representation.

        Ensures two equivalent inputs produce identical JSON, regardless of the
        original Python types or dict-key ordering. Handles ``datetime``,
        ``pd.Timestamp``, ``Path``, NumPy scalars, sets, and tuples in addition
        to plain containers.

        Args:
            value (Any): Arbitrary nested structure to normalise.

        Returns:
            Any: A structure built from ``dict``, ``list``, ``str``, ``int``,
                ``float``, ``bool``, and ``None`` only.
        """
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.isoformat()
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return [BaseModel._canonicalize(v) for v in value.tolist()]
        if isinstance(value, dict):
            return {
                str(k): BaseModel._canonicalize(value[k])
                for k in sorted(value, key=str)
            }
        if isinstance(value, (set, frozenset)):
            return sorted(
                (BaseModel._canonicalize(v) for v in value),
                key=lambda v: json.dumps(v, sort_keys=True, default=str),
            )
        if isinstance(value, (list, tuple)):
            return [BaseModel._canonicalize(v) for v in value]
        return str(value)

    @staticmethod
    def tremor_fingerprint(df: pd.DataFrame) -> dict:
        """Return a compact, hash-stable fingerprint of a tremor DataFrame.

        Captures the row count, index span, and sorted column names — enough
        to detect tremor recalculations (different outlier method, value
        multiplier, frequency bands, etc.) without hashing the full numerical
        payload.

        Args:
            df (pd.DataFrame): Tremor DataFrame with a ``DatetimeIndex``.

        Returns:
            dict: ``{"start", "end", "n_rows", "columns"}`` where ``start`` and
                ``end`` are ISO strings (or ``None`` for an empty frame) and
                ``columns`` is sorted.
        """
        if df.empty:
            return {"start": None, "end": None, "n_rows": 0, "columns": []}

        return {
            "start": pd.Timestamp(df.index.min()).isoformat(),
            "end": pd.Timestamp(df.index.max()).isoformat(),
            "n_rows": int(len(df)),
            "columns": sorted(str(c) for c in df.columns),
        }

    def save(
        self, identity: dict | None = None, path: str | None = None
    ) -> str:
        """Serialise this model instance to a ``.pkl`` file via joblib.

        Two modes:

        - **Cache mode** (``identity`` is set) — writes
          ``{stage_dir}/{hash}.{ClassName}.pkl`` plus a
          ``{hash}.{ClassName}.params.json`` sidecar capturing the
          canonicalised identity. Stage methods call this mode automatically
          so a successful run is replayable from disk by hash.
        - **Legacy mode** (``identity`` is ``None``) — writes
          ``{output_dir}/{ClassName}_{basename}.pkl`` (or
          ``{output_dir}/{ClassName}.pkl`` when ``self.basename`` is unset).
          Kept for standalone manual dumps; ``path`` overrides the default
          destination.

        Args:
            identity (dict | None): Identity dict produced by
                :meth:`build_identity`. When set, enables cache mode.
                Defaults to ``None``.
            path (str | None): Explicit destination path. Ignored in cache
                mode. When ``None`` in legacy mode the default basename path
                is used. Defaults to ``None``.

        Returns:
            str: Absolute path to the written ``.pkl`` file.

        Example:
            >>> # Cache mode — content-addressable artefact
            >>> path = model.save(TrainingModel.build_identity(...))
            >>> # Legacy mode — basename pkl for manual use
            >>> path = model.save()
        """
        if identity is not None:
            cache_path = type(self).cache_path(self.stage_dir, identity)
            ensure_dir(self.stage_dir)
            joblib.dump(self, cache_path)

            sidecar = cache_path[: -len(".pkl")] + ".params.json"
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump(
                    type(self)._canonicalize(identity), f, indent=2, default=str
                )

            logger.info(f"[{type(self).__name__}] Cached at: {cache_path}")
            return cache_path

        if path is None:
            basename: str | None = getattr(self, "basename", None)
            suffix = f"_{basename}" if basename else ""
            path = os.path.join(self.output_dir, f"{type(self).__name__}{suffix}.pkl")

        ensure_dir(str(Path(path).resolve().parent))
        joblib.dump(self, path)
        logger.info(f"[{type(self).__name__}] Saved to: {path}")
        return path

    @classmethod
    def load(cls, stage_dir: str, identity: dict) -> Self | None:
        """Return the cached instance for ``identity`` if it exists on disk.

        Resolves the cache path via :meth:`cache_path` and joblib-loads the
        pickle. Returns ``None`` on a cache miss so callers can fall through
        to the regular build path. Path-based restores live outside this
        method — use :func:`eruption_forecast.utils.pathutils.load_pickle`
        directly when a ``.pkl`` location is known.

        Args:
            stage_dir (str): Stage directory where the cache file lives.
            identity (dict): Identity dict produced by :meth:`build_identity`.

        Returns:
            Self | None: The restored instance on a cache hit, otherwise
                ``None``.
        """
        path = cls.cache_path(stage_dir, identity)
        if not os.path.isfile(path):
            return None

        try:
            obj: Self = load_pickle(path)
            logger.info(f"[{cls.__name__}] Loaded from cache: {path}")
            return obj
        except RuntimeError as e:
            logger.warning(f"[{cls.__name__}] Failed to load from cache: {path}. {e}")
            return None

    def build_label(
        self, window_step: int, window_step_unit: Literal["minutes", "hours"]
    ) -> Self:
        """Build the label set for this model stage.

        Concrete subclasses that need to construct a stage-specific label
        DataFrame (training labels, forecast-grid labels, or
        evaluation-window labels) override this hook and store the supplied
        window configuration on the instance. Reuse stages such as
        ``ExplanationModel`` inherit this no-op so the same kwargs can be
        passed uniformly across mixed pipelines.

        Args:
            window_step (int): Step size between consecutive windows.
                Ignored by the no-op default; consumed by overriding
                subclasses.
            window_step_unit (Literal["minutes", "hours"]): Unit for
                ``window_step``. Ignored by the no-op default; consumed by
                overriding subclasses.

        Returns:
            Self: The current instance, for method chaining.
        """
        return self

    def _build_features(
        self,
        label_df: pd.DataFrame,
        output_dir: str,
        features_dir: str,
        features_label: pd.DataFrame | None = None,
        select_tremor_columns: list[str] | None = None,
        save_tremor_matrix_per_method: bool = False,
        save_tremor_matrix_per_id: bool = False,
        minimum_completion: float = 1.0,
        overwrite: bool = False,
        n_jobs: int | None = None,
        verbose: bool | None = None,
        select_features: list[str] | None = None,
    ) -> FeaturesBuilder:
        """Build a tremor matrix and return a configured ``FeaturesBuilder``.

        Constructs the windowed tremor matrix aligned to ``label_df`` via
        ``TremorMatrixBuilder``, then wraps it in a ``FeaturesBuilder``
        ready for tsfresh extraction. The returned builder is not run; the
        caller decides when and how to extract.

        Args:
            label_df (pd.DataFrame): Label DataFrame used to align tremor
                windows. Drives the row count and time index of the
                resulting matrix.
            output_dir (str): Base output directory for tremor matrix files
                (the matrix is written under ``output_dir/tremor/``).
            features_dir (str): Output directory for extracted features.
            features_label (pd.DataFrame | None, optional): Label DataFrame
                passed through to ``FeaturesBuilder`` as the supervised
                target. ``None`` enables prediction (unsupervised) mode and
                disables tsfresh relevance filtering. Defaults to ``None``.
            select_tremor_columns (list[str] | None, optional): Tremor
                column names to include. All columns are used when ``None``.
                Defaults to ``None``.
            save_tremor_matrix_per_method (bool, optional): Save per-column
                tremor matrices under ``output_dir/tremor/per_method/`` when
                ``True``. Defaults to ``False``.
            save_tremor_matrix_per_id (bool, optional): Save per-ID tremor
                matrices when ``True``. Defaults to ``False``.
            minimum_completion (float, optional): Minimum data-completeness
                ratio in the range ``0.0`` to ``1.0``. Tremor windows whose
                sample count falls below this fraction of the expected
                count are skipped. Defaults to ``1.0`` (no gaps tolerated).
            overwrite (bool, optional): Overwrite existing matrix files when
                ``True``. OR-combined with ``self.overwrite``. Defaults to
                ``False``.
            n_jobs (int | None, optional): Number of parallel workers for
                ``FeaturesBuilder``. Falls back to ``self.n_jobs`` when
                ``None``. Defaults to ``None``.
            verbose (bool | None, optional): Emit verbose log messages.
                Falls back to ``self.verbose`` when ``None``. Defaults to
                ``None``.
            select_features (list[str] | None, optional): Pre-filter tsfresh to
                this list of fully-qualified feature names; forwarded straight
                to :class:`FeaturesBuilder`. ``None`` keeps the default
                ``ComprehensiveFCParameters`` behaviour. Defaults to ``None``.

        Returns:
            FeaturesBuilder: Configured builder ready to run feature
                extraction.
        """
        tremor_matrix_df = (
            TremorMatrixBuilder(
                tremor_df=self.tremor_df,
                label_df=label_df,
                output_dir=os.path.join(output_dir, "tremor"),
                window_size=self.window_size,
                overwrite=overwrite or self.overwrite,
                minimum_completion=minimum_completion,
                verbose=verbose if verbose else self.verbose,
            )
            .build(
                select_tremor_columns=select_tremor_columns,
                save_tremor_matrix_per_method=save_tremor_matrix_per_method,
                save_tremor_matrix_per_id=save_tremor_matrix_per_id,
            )
            .df
        )

        features_builder = FeaturesBuilder(
            tremor_matrix_df=tremor_matrix_df,
            label_df=features_label,
            select_features=select_features,
            output_dir=features_dir,
            overwrite=overwrite or self.overwrite,
            n_jobs=n_jobs if n_jobs is not None else self.n_jobs,
            verbose=self.verbose,
        )

        return features_builder

    def extract_features(self) -> Self:
        """Extract tsfresh features from the tremor data.

        Concrete subclasses that own a feature-extraction pass override this
        hook to build a tremor matrix aligned to their labels (typically via
        ``_build_features()``), run tsfresh extraction, and populate
        ``features_df`` and ``features_path`` on the instance. Reuse stages
        such as ``ExplanationModel`` inherit this no-op so they can still be
        dropped into chains that call ``extract_features()``.

        Returns:
            Self: The current instance, for method chaining.
        """
        return self
