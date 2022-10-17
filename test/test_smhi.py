from app.smhi import fetch_smhi_temperatures

import requests_mock
import json
from pathlib import Path

THIS_DIR = Path(__file__).parent


def test_fetch_smhi_temperatures(pinned):

    with requests_mock.Mocker() as mock:
        mock.get(
            "https://opendata-download-metfcst.smhi.se/api/category/pmp3g/version/2/geotype/point/lon/11/lat/56/data.json",
            json=json.load((THIS_DIR / "smhi_mocked_response.json").open()),
        )
        temps = fetch_smhi_temperatures(11, 56)
        assert temps == pinned
