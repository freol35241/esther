from datetime import date

import requests
import numpy as np

SMHI_API_URL = "https://opendata-download-metfcst.smhi.se/api/category/pmp3g/version/2/geotype/point/lon/{longitude}/lat/{latitude}/data.json"


def fetch_smhi_temperatures(longitude: float, latitude: float) -> list:
    url = SMHI_API_URL.format(longitude=longitude, latitude=latitude)

    response = requests.get(url)

    response.raise_for_status()

    data = response.json()

    forecasts = data["timeSeries"]

    out = []

    for forecast in forecasts:
        for param in forecast["parameters"]:
            if param["name"] == "t":
                out.append(param["values"][0])

    return np.asarray(out)
