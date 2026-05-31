"""Content-addressable cache for model pipeline stages.

This module provides :class:`CacheModel`, an abstract base for model classes
whose expensive build artefacts (trained ensembles, extracted feature matrices,
forecast results) should be reusable across runs when their identity-defining
parameters match exactly.

Subclasses declare which parameters define a unique cache entry via the
abstract :meth:`CacheModel.build_cache_identity` classmethod. The base provides
the canonical-JSON hashing and the on-disk read/write layer underneath
``{output_dir}/cache/{ClassName}/``.
"""

import os
import json
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Self
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib
import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import ensure_dir


class CacheModel(ABC):
    """Abstract base for content-addressable model caches.

    Concrete subclasses implement :meth:`build_cache_identity` to declare the
    parameters that define a unique cache entry. Identity dicts are
    canonicalised and SHA-256-hashed so that two calls with identical
    parameters resolve to the same on-disk pickle.

    The cache layer is intentionally decoupled from
    :class:`~eruption_forecast.model.base_model.BaseModel`: it relies only on
    ``self.output_dir`` being set on the instance and uses ``joblib`` directly
    for persistence. Legacy ``BaseModel.save()`` / ``BaseModel.load()`` paths
    remain untouched so that existing call sites continue to work.

    Attributes:
        output_dir (str): Base directory under which the ``cache/`` subtree is
            created. Concrete subclasses are expected to set this during
            construction (``BaseModel`` already does so).
    """

    output_dir: str

    @classmethod
    @abstractmethod
    def build_cache_identity(cls, **kwargs: Any) -> dict:
        """Return the canonical identity dict that defines a cache entry.

        Implementations should include every parameter that materially affects
        the produced artefact (constructor args plus the args passed to the
        pipeline methods that mutate the instance). Runtime knobs that do not
        affect output — ``n_jobs``, ``n_grids``, ``verbose``, ``overwrite``,
        ``output_dir``, ``root_dir`` — must be excluded.

        Args:
            **kwargs (Any): Subclass-specific identity inputs. Each subclass
                documents its accepted keyword arguments.

        Returns:
            dict: Canonical, JSON-serialisable identity dict ready for hashing.
        """

    @classmethod
    def cache_dir(cls, output_dir: str) -> str:
        """Return the cache subdirectory for this class.

        Args:
            output_dir (str): Base output directory.

        Returns:
            str: ``{output_dir}/cache/{ClassName}``.
        """
        return os.path.join(output_dir, "cache", cls.__name__)

    @classmethod
    def cache_path_for(cls, output_dir: str, identity: dict) -> str:
        """Return the absolute ``.pkl`` path for the supplied identity.

        Args:
            output_dir (str): Base output directory.
            identity (dict): Identity dict produced by
                :meth:`build_cache_identity`.

        Returns:
            str: Absolute path of the form
                ``{output_dir}/cache/{ClassName}/{hash}.pkl``.
        """
        key = cls.compute_hash(identity)
        return os.path.join(cls.cache_dir(output_dir), f"{key}.pkl")

    def save_to_cache(self, identity: dict) -> str:
        """Persist this instance and the identity dict to the cache directory.

        Writes the pickled instance to ``{hash}.pkl`` and the canonicalised
        identity to ``{hash}.params.json`` alongside it. The JSON sidecar is
        diagnostic only — it lets a human map a cache file back to the call
        that produced it.

        Args:
            identity (dict): Identity dict produced by
                :meth:`build_cache_identity`.

        Returns:
            str: Absolute path of the written ``.pkl``.
        """
        path = type(self).cache_path_for(self.output_dir, identity)
        ensure_dir(os.path.dirname(path))
        joblib.dump(self, path)

        sidecar = path[: -len(".pkl")] + ".params.json"
        with open(sidecar, "w") as f:
            json.dump(self._canonicalize(identity), f, indent=2, default=str)

        logger.info(f"[{type(self).__name__}] Cached at: {path}")
        return path

    @classmethod
    def load_from_cache(cls, output_dir: str, identity: dict) -> Self | None:
        """Return the cached instance for ``identity`` if it exists on disk.

        Args:
            output_dir (str): Base output directory used to resolve the cache
                path.
            identity (dict): Identity dict produced by
                :meth:`build_cache_identity`.

        Returns:
            Self | None: The restored instance on a cache hit, otherwise
                ``None``.
        """
        path = cls.cache_path_for(output_dir, identity)
        if not os.path.isfile(path):
            return None
        obj: Self = joblib.load(path)
        logger.info(f"[{cls.__name__}] Loaded from cache: {path}")
        return obj

    @classmethod
    def compute_hash(cls, identity: dict, length: int = 12) -> str:
        """Hash the canonical form of ``identity`` to a truncated hex digest.

        Args:
            identity (dict): Identity dict produced by
                :meth:`build_cache_identity`.
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

        Ensures two equivalent inputs produce identical JSON, regardless of
        the original Python types or dict-key ordering. Handles ``datetime``,
        ``pd.Timestamp``, ``Path``, NumPy scalars, sets, and tuples in
        addition to plain containers.

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
            return [CacheModel._canonicalize(v) for v in value.tolist()]
        if isinstance(value, dict):
            return {
                str(k): CacheModel._canonicalize(value[k])
                for k in sorted(value, key=str)
            }
        if isinstance(value, (set, frozenset)):
            return sorted(
                (CacheModel._canonicalize(v) for v in value),
                key=lambda v: json.dumps(v, sort_keys=True, default=str),
            )
        if isinstance(value, (list, tuple)):
            return [CacheModel._canonicalize(v) for v in value]
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
