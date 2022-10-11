from datetime import date, datetime

import requests
import numpy as np
import pandas as pd

NORDPOOL_API_URL = (
    "https://www.vattenfall.se/api/price/spot/pricearea/{start}/{end}/{price_area}"
)


def fetch_nordpool_data(
    price_area: str, start: date = None, end: date = None
) -> np.ndarray:

    only_future_times = True if start is None else False

    start = start or date.today()
    end = end or date.fromisoformat("2099-12-31")

    url = NORDPOOL_API_URL.format(
        start=start.isoformat(), end=end.isoformat(), price_area=price_area
    )

    response = requests.get(url)

    response.raise_for_status()

    data = pd.json_normalize(response.json())

    now = datetime.now()

    if only_future_times:
        data = data[now.hour :]

    return np.asarray(data["Value"])
