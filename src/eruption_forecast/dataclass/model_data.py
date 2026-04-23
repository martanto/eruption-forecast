import os
from datetime import datetime
from dataclasses import field, dataclass

import pandas as pd

from eruption_forecast import TremorData
from eruption_forecast.utils.pathutils import resolve_output_dir
from eruption_forecast.utils.date_utils import normalize_dates


@dataclass
class ModelData:
    _tremor_data: str | pd.DataFrame
    _start_date: str | datetime
    _end_date: str | datetime
    _sub_dir: str
    _output_dir: str | None = field(default=None)
    _root_dir: str | None = field(default=None)

    tremor_data: pd.DataFrame = field(init=False, repr=False)
    start_date: datetime = field(init=False, repr=False)
    end_date: datetime = field(init=False, repr=False)
    start_date_str: str = field(init=False, repr=False)
    end_date_str: str = field(init=False, repr=False)
    output_dir: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self._tremor_data, str):
            self.tremor_data = TremorData.from_csv(self._tremor_data).df
        elif isinstance(self._tremor_data, pd.DataFrame):
            self.tremor_data = TremorData(self._tremor_data).df
        else:
            raise TypeError(
                f"tremor_data must be ``str`` or ``pd.DataFrame``, found {type(self._tremor_data)}"
            )

        self.start_date, self.end_date, self.start_date_str, self.end_date_str = (
            normalize_dates(self._start_date, self._end_date)
        )

        output_dir = resolve_output_dir(
            self._output_dir,
            self._root_dir,
            os.path.join("output"),
        )
        self.output_dir: str = os.path.join(output_dir, self._sub_dir)
