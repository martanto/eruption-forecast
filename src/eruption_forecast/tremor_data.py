# Third party imports
import numpy as np
import pandas as pd


class TremorData:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def load(self, file: str) -> pd.DataFrame:
        self.df = pd.read_csv(file)
        return self.df
