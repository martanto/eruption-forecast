"""Tests for the ``SDS`` reader and the ``FDSN`` adapter.

``SDS`` runs against a real miniSEED archive written under ``tmp_path``.
``FDSN`` swaps the ObsPy client for ``_FakeClient`` so no request ever leaves
the machine; the tests check the local SDS cache behaviour around it.
"""

import os
import importlib
from datetime import datetime

import numpy as np
import pytest
from obspy import Stream
from obspy.clients.fdsn.header import FDSNException

from tests.conftest import build_stream
from eruption_forecast.sources.sds import SDS


fdsn_module = importlib.import_module("eruption_forecast.sources.fdsn")

DAY = datetime(2025, 1, 15)


def _short_stream(starttime: str = "2025-01-15T00:00:00") -> Stream:
    """One hour at 20 Hz keeps miniSEED reads and writes fast."""
    return build_stream(starttime=starttime, duration_seconds=3600)


@pytest.fixture()
def sds(tmp_path) -> SDS:
    return SDS(str(tmp_path), station="ojn", channel="ehz", network="vg", location="00")


# ---------------------------------------------------------------------------
# SDS
# ---------------------------------------------------------------------------


class TestSDSInit:
    def test_uppercases_codes(self, sds: SDS) -> None:
        assert sds.nslc == "VG.OJN.00.EHZ"
        assert sds.channel_type == "D"

    def test_none_location_is_empty(self, tmp_path) -> None:
        assert SDS(str(tmp_path), "OJN", "EHZ", "VG", location=None).nslc == "VG.OJN..EHZ"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"station": ""},
            {"channel": ""},
            {"network": ""},
            {"location": 0},
        ],
    )
    def test_invalid_codes_raise(self, tmp_path, kwargs) -> None:
        params = {"station": "OJN", "channel": "EHZ", "network": "VG", **kwargs}
        with pytest.raises(ValueError):
            SDS(str(tmp_path), **params)

    def test_missing_dir_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            SDS(str(tmp_path / "missing"), "OJN", "EHZ", "VG")

    def test_file_instead_of_dir_raises(self, tmp_path) -> None:
        path = tmp_path / "file.txt"
        path.write_text("x")
        with pytest.raises(NotADirectoryError):
            SDS(str(path), "OJN", "EHZ", "VG")


class TestSDSFilepath:
    def test_follows_sds_layout(self, sds: SDS, tmp_path) -> None:
        expected = os.path.join(
            str(tmp_path.resolve()), "2025", "VG", "OJN", "EHZ.D", "VG.OJN.00.EHZ.D.2025.015"
        )
        assert sds.get_filepath(DAY) == expected

    def test_julian_day_zero_padded(self, sds: SDS) -> None:
        assert sds.get_filepath(datetime(2025, 1, 1)).endswith(".2025.001")

    def test_rejects_non_datetime(self, sds: SDS) -> None:
        with pytest.raises(TypeError):
            sds.get_filepath("2025-01-15")  # type: ignore[arg-type]


class TestSDSReadWrite:
    def test_save_and_get_round_trip(self, sds: SDS) -> None:
        stream = _short_stream()
        path = sds.save(stream, DAY)

        assert path == sds.get_filepath(DAY)
        loaded = sds.get(DAY)
        assert len(loaded) == 1
        np.testing.assert_allclose(loaded[0].data, stream[0].data)
        assert sds.files[-1]["date"] == "2025-01-15"

    def test_get_trace(self, sds: SDS) -> None:
        sds.save(_short_stream(), DAY)
        trace = sds.get_trace(DAY)
        assert trace is not None
        assert trace.stats.station == "OJN"

    def test_missing_day_returns_empty(self, sds: SDS) -> None:
        assert len(sds.get(DAY)) == 0
        assert sds.get_trace(DAY) is None

    def test_save_empty_stream_returns_none(self, sds: SDS) -> None:
        assert sds.save(Stream(), DAY) is None

    def test_gappy_file_is_merged(self, sds: SDS) -> None:
        first = _short_stream("2025-01-15T00:00:00")
        second = _short_stream("2025-01-15T02:00:00")
        sds.save(first + second, DAY)

        stream = sds.get(DAY)

        assert len(stream) == 1
        assert not np.isnan(stream[0].data).any()  # interpolate=True by default

    def test_corrupt_file_returns_empty(self, sds: SDS) -> None:
        path = sds.get_filepath(DAY)
        os.makedirs(os.path.dirname(path))
        with open(path, "wb") as f:
            f.write(b"definitely not miniseed")
        assert len(sds.get(DAY)) == 0

    def test_get_rejects_non_datetime(self, sds: SDS) -> None:
        with pytest.raises(TypeError):
            sds.get("2025-01-15")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FDSN
# ---------------------------------------------------------------------------


class _FakeClient:
    """Stand-in for ``obspy.clients.fdsn.Client`` that records every request."""

    instances: list["_FakeClient"] = []
    response: Stream | Exception = Stream()

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.calls: list[dict] = []
        _FakeClient.instances.append(self)

    def get_waveforms(self, **kwargs) -> Stream:
        self.calls.append(kwargs)
        if isinstance(_FakeClient.response, Exception):
            raise _FakeClient.response
        return _FakeClient.response.copy()


@pytest.fixture()
def fake_client(monkeypatch) -> type[_FakeClient]:
    _FakeClient.instances = []
    _FakeClient.response = _short_stream()
    monkeypatch.setattr(fdsn_module, "FDSNClient", _FakeClient)
    return _FakeClient


def _fdsn(tmp_path, **kwargs):
    return fdsn_module.FDSN(
        station="OJN",
        channel="EHZ",
        network="VG",
        location="00",
        download_dir=str(tmp_path / "downloads"),
        **kwargs,
    )


class TestFDSN:
    def test_init_creates_cache_and_defaults_url(self, tmp_path, fake_client) -> None:
        fdsn = _fdsn(tmp_path)
        assert os.path.isdir(tmp_path / "downloads")
        assert fake_client.instances[0].base_url == "https://service.iris.edu"
        assert fdsn.nslc == "VG.OJN.00.EHZ"

    def test_downloads_full_day_and_caches(self, tmp_path, fake_client) -> None:
        fdsn = _fdsn(tmp_path, client_url="https://example.org")

        stream = fdsn.get(DAY)

        call = fake_client.instances[0].calls[0]
        assert str(call["starttime"]) == "2025-01-15T00:00:00.000000Z"
        assert str(call["endtime"]) == "2025-01-15T23:59:59.000000Z"
        assert len(stream) == 1
        assert os.path.isfile(fdsn.SDS.get_filepath(DAY))

    def test_second_call_served_from_cache(self, tmp_path, fake_client) -> None:
        fdsn = _fdsn(tmp_path)
        fdsn.get(DAY)
        fdsn.get(DAY)
        assert len(fake_client.instances[0].calls) == 1

    def test_overwrite_redownloads(self, tmp_path, fake_client) -> None:
        fdsn = _fdsn(tmp_path, overwrite=True)
        fdsn.get(DAY)
        fdsn.get(DAY)
        assert len(fake_client.instances[0].calls) == 2

    def test_empty_response_not_cached(self, tmp_path, fake_client) -> None:
        fake_client.response = Stream()
        fdsn = _fdsn(tmp_path)
        assert len(fdsn.get(DAY)) == 0
        assert not os.path.exists(fdsn.SDS.get_filepath(DAY))

    def test_service_error_returns_empty(self, tmp_path, fake_client) -> None:
        fake_client.response = FDSNException("service down")
        assert len(_fdsn(tmp_path).get(DAY)) == 0

    def test_rejects_non_datetime(self, tmp_path, fake_client) -> None:
        with pytest.raises(TypeError):
            _fdsn(tmp_path).get("2025-01-15")  # type: ignore[arg-type]
