import numpy as np
import pandas as pd


class TremorData:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def load(self, file: str) -> pd.DataFrame:
        self.data = pd.read_csv(file)
        return self.data
