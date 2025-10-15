import os
from datetime import datetime
from obspy import read, Stream

class SDS:
    def __init__(self, sds_dir: str, nslc: str):
        self.sds_dir = sds_dir
        self.nslc = nslc

        assert len(nslc.split(".")) == 3, ValueError("❌ NSLC format incorrect")

        self.network, self.station, self.location, self.channel = nslc.split('.')

    def get(self, date: datetime) -> Stream:
        return Stream()
