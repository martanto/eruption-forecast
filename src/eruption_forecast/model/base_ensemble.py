"""BaseEnsemble: shared save/load logic for ensemble classes.

Provides :class:`BaseEnsemble`, a mixin that implements ``save`` and ``load``
via joblib so that :class:`SeedEnsemble` and :class:`ClassifierEnsemble` do not
duplicate the persistence boilerplate.
"""

import os

import joblib

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import ensure_dir


class BaseEnsemble:
    """Mixin that provides joblib-based save/load for ensemble objects.

    Subclasses gain :meth:`save` and :meth:`load` without repeating the
    boilerplate.  Override :meth:`_load_log_msg` to append class-specific
    information (e.g. seed count, classifier count) to the load log line.
    """

    def save(self, path: str) -> None:
        """Dump the ensemble to a single ``.pkl`` file via joblib.

        Creates any missing parent directories, serialises the entire object
        with ``joblib.dump``, and logs the destination path.  Reload with
        :meth:`load`.

        Args:
            path (str): Destination file path (should end with ``.pkl``).
        """
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        joblib.dump(self, path)
        logger.info(f"[{type(self).__name__}] Saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "BaseEnsemble":
        """Load a previously saved ensemble from a ``.pkl`` file.

        Restores the full object from a file written by :meth:`save`.

        Args:
            path (str): Path to the ``.pkl`` file.

        Returns:
            BaseEnsemble: The restored ensemble instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{cls.__name__} file not found: {path}")
        obj = joblib.load(path)
        extra = cls._load_log_msg(obj)
        suffix = f" — {extra}" if extra else ""
        logger.info(f"[{cls.__name__}] Loaded from: {path}{suffix}")
        return obj

    @classmethod
    def _load_log_msg(cls, obj: "BaseEnsemble") -> str:
        """Return an optional info string appended to the load log message.

        Subclasses override this to report ensemble-specific counts (e.g.
        number of seeds or classifiers).  The default returns an empty string.

        Args:
            obj (BaseEnsemble): The just-loaded ensemble instance.

        Returns:
            str: Info string to append, or empty string for no suffix.
        """
        return ""
