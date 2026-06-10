import os
import multiprocessing
from abc import ABC, abstractmethod
from typing import Self, Literal
from pathlib import Path
from datetime import datetime
from functools import cached_property

import joblib
import pandas as pd

from eruption_forecast.utils import validate_date_ranges
from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import ensure_dir, resolve_output_dir
from eruption_forecast.utils.date_utils import sort_dates, to_datetime
from eruption_forecast.tremor.tremor_data import TremorData
from eruption_forecast.features.features_builder import FeaturesBuilder
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble
from eruption_forecast.features.tremor_matrix_builder import TremorMatrixBuilder


class BaseModel(ABC):
    """Abstract base class for eruption forecast model stages.

    Concrete subclasses (``TrainingModel``, ``PredictionModel``,
    ``EvaluationModel``) inherit construction, parameter validation,
    tremor-data loading, output-directory resolution, and joblib-based
    persistence from this class. Each subclass implements its own
    ``set_directories()``, ``create_directories()``, ``validate()``,
    ``describe()``, ``to_dict()``, ``to_prompt()``, ``build_label()``,
    and ``extract_features()``.

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
        features_csv (str | None): Path to the features CSV on disk.
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
        self.features_csv: str | None = None

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

    @abstractmethod
    def validate(self) -> Self:
        """Validate model-specific parameters.

        Subclasses extend ``_validate()`` with stage-specific checks
        (e.g. that ``eruption_dates`` is provided when required, or that
        a pre-fitted ensemble is available before prediction).

        Returns:
            Self: The current instance, for method chaining.

        Raises:
            ValueError: When stage-specific validation fails.
        """

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

    def save(self, path: str | None = None) -> str:
        """Serialise this model instance to a ``.pkl`` file via joblib.

        Persists the entire object — including cached tremor data,
        extracted features, the loaded ``ClassifierEnsemble``, and window-grid
        metadata — so that a later run can restore it with :meth:`load` and
        pass it directly to ``EvaluationModel`` without re-training or
        re-extracting features.

        The default path is
        ``{output_dir}/{ClassName}_{basename}.pkl`` when ``self.basename``
        is set and non-empty, and ``{output_dir}/{ClassName}.pkl`` otherwise.

        Args:
            path (str | None): Explicit destination path. When ``None`` the
                default path is used. Defaults to ``None``.

        Returns:
            str: Absolute path to the written ``.pkl`` file.

        Example:
            >>> path = model.save()
            >>> restored = TrainingModel.load(path)
        """
        if path is None:
            basename: str | None = getattr(self, "basename", None)
            suffix = f"_{basename}" if basename else ""
            path = os.path.join(self.output_dir, f"{type(self).__name__}{suffix}.pkl")

        ensure_dir(str(Path(path).resolve().parent))
        joblib.dump(self, path)
        logger.info(f"[{type(self).__name__}] Saved to: {path}")
        return path

    @classmethod
    def load(cls, path: str) -> Self:
        """Restore a model instance previously saved with :meth:`save`.

        Args:
            path (str): Path to a ``.pkl`` file produced by :meth:`save`.

        Returns:
            Self: The restored model instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist on disk.

        Example:
            >>> model = TrainingModel.load("output/TrainingModel_2025-01-01_2025-12-31.pkl")
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{cls.__name__}: file not found: {path}")
        obj = joblib.load(path)
        logger.info(f"[{cls.__name__}] Loaded from: {path}")
        return obj

    @abstractmethod
    def build_label(
        self, window_step: int, window_step_unit: Literal["minutes", "hours"]
    ) -> Self:
        """Build the label set for this model stage.

        Concrete subclasses construct a label DataFrame appropriate to
        their stage (training labels, forecast-grid labels, or
        evaluation-window labels) and store the supplied window
        configuration on the instance.

        Args:
            window_step (int): Step size between consecutive windows.
            window_step_unit (Literal["minutes", "hours"]): Unit for
                ``window_step``.

        Returns:
            Self: The current instance, for method chaining.
        """

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
            output_dir=features_dir,
            overwrite=overwrite or self.overwrite,
            n_jobs=n_jobs if n_jobs is not None else self.n_jobs,
            verbose=self.verbose,
            select_features=select_features,
        )

        return features_builder

    @abstractmethod
    def extract_features(self) -> Self:
        """Extract tsfresh features from the tremor data.

        Concrete subclasses build a tremor matrix aligned to their labels
        (typically via ``_build_features()``), run tsfresh extraction, and
        populate ``features_df`` and ``features_csv`` on the instance.

        Returns:
            Self: The current instance, for method chaining.
        """
