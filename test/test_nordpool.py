from app.nordpool import fetch_nordpool_data

from datetime import datetime, timezone
import requests_mock
import json
from pathlib import Path
from unittest.mock import patch


THIS_DIR = Path(__file__).parent


def test_fetch_nordpool_data(pinned):

    with requests_mock.Mocker() as mock, patch(
        "app.nordpool._get_break_point_time"
    ) as time_wrapper:
        mock.get(
            "https://spot.utilitarian.io/electricity/SE3/latest/",
            json=json.load((THIS_DIR / "nordpool_mocked_response.json").open()),
        )
        time_wrapper.return_value = datetime(2022, 10, 17, 11, 32, tzinfo=timezone.utc)
        prices = fetch_nordpool_data("SE3")
        assert prices == pinned
