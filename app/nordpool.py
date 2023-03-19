from datetime import datetime, timezone, timedelta
import logging

import requests
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

NORDPOOL_API_URL = "https://spot.utilitarian.io/electricity/{price_area}/latest/"


def _get_break_point_time():
    now = datetime.now(timezone.utc)
    return now - timedelta(hours=1)


def fetch_nordpool_data(price_area: str) -> np.ndarray:
    url = NORDPOOL_API_URL.format(price_area=price_area)

    response = requests.get(url)

    response.raise_for_status()

    data: pd.DataFrame = pd.json_normalize(response.json())

    data["timestamp"] = pd.to_datetime(data["timestamp"], infer_datetime_format=True)
    data.set_index("timestamp", inplace=True)
    data.sort_index(inplace=True)

    LOGGER.info("Fetched new data from %s", url)
    LOGGER.debug(data)

    return np.asarray(data[_get_break_point_time() :]["value"], dtype=float)
