"""Tests for StationData dataclass."""

import pytest

from eruption_forecast.dataclass.station_data import StationData


class TestStationDataConstruction:
    def test_basic_construction(self):
        sd = StationData(station="OJN", channel="EHZ", network="VG", location="00")
        assert sd.station == "OJN"
        assert sd.channel == "EHZ"
        assert sd.network == "VG"
        assert sd.location == "00"

    def test_uppercase_normalisation(self):
        sd = StationData(station="ojn", channel="ehz", network="vg", location="00")
        assert sd.station == "OJN"
        assert sd.channel == "EHZ"
        assert sd.network == "VG"

    def test_nslc_derived(self):
        sd = StationData(station="OJN", channel="EHZ", network="VG", location="00")
        assert sd.nslc == "VG.OJN.00.EHZ"

    def test_nslct_derived(self):
        sd = StationData(station="OJN", channel="EHZ", network="VG", location="00")
        assert sd.nslct == "VG.OJN.00.EHZ.D"

    def test_location_defaults_to_empty_string(self):
        sd = StationData(station="OJN", channel="EHZ", network="VG")
        assert sd.location == ""
        assert sd.nslc == "VG.OJN..EHZ"

    def test_location_none_coerced_to_empty(self):
        sd = StationData(station="OJN", channel="EHZ", network="VG", location=None)
        assert sd.location == ""

    def test_channel_type_default(self):
        sd = StationData(station="OJN", channel="EHZ", network="VG")
        assert sd.channel_type == "D"

    def test_channel_type_custom(self):
        sd = StationData(station="OJN", channel="EHZ", network="VG", channel_type="r")
        assert sd.channel_type == "R"


class TestStationDataValidation:
    def test_empty_station_raises(self):
        with pytest.raises(ValueError):
            StationData(station="", channel="EHZ", network="VG")

    def test_empty_channel_raises(self):
        with pytest.raises(ValueError):
            StationData(station="OJN", channel="", network="VG")

    def test_empty_network_raises(self):
        with pytest.raises(ValueError):
            StationData(station="OJN", channel="EHZ", network="")

    def test_non_string_station_raises(self):
        with pytest.raises((ValueError, TypeError)):
            StationData(station=123, channel="EHZ", network="VG")

    def test_non_string_location_raises(self):
        with pytest.raises(ValueError):
            StationData(station="OJN", channel="EHZ", network="VG", location=123)
