import json
import logging
import argparse
import time
from typing import Tuple
from dataclasses import dataclass
from functools import partial
from datetime import datetime

import numpy as np
from streamz import Stream
from paho.mqtt.client import MQTTMessage
from jsonpointer import JsonPointer

from app import dynamic_model, opt, streamz_nodes
from app.nordpool import fetch_nordpool_data
from app.smhi import fetch_smhi_temperatures
from app.heat_sources import IVT490


@dataclass
class Config:
    T_indoor_requested: float = 20.0
    T_indoor_bounds: Tuple[float] = (-2, 2)
    T_feed_maximum: float = 60.0
    sensor_timeout: int = 600


config = Config()


def run(cmd_args: argparse.Namespace):
    """Main function setting up and running the gateway functionality

    Args:
        config (argparse.Namespace): argparse command line config
    """

    # Update config
    config.T_indoor_requested = cmd_args.T_indoor_requested
    config.T_indoor_bounds = cmd_args.T_indoor_bounds
    config.T_feed_maximum = cmd_args.T_feed_maximum
    config.sensor_timeout = cmd_args.sensor_timeout

    def resolve_jsonpointer(pointer: JsonPointer, x: MQTTMessage):
        obj = json.loads(x.payload)
        return pointer.resolve(obj)

    def run_optimization(x):
        T_outdoor_current, T_indoor_current = x
        prices = fetch_nordpool_data(cmd_args.nordpool_price_area)
        outdoor_temperatures = fetch_smhi_temperatures(
            cmd_args.longitude, cmd_args.latitude
        )

        # Add current outdoor temperature to the list of prognosticised temperatures and adjust length in accordance to the list of prices
        outdoor_temperatures = np.insert(outdoor_temperatures, 0, T_outdoor_current)
        outdoor_temperatures = outdoor_temperatures[: len(prices)]

        # Generate times to be simulated
        delta_t = np.ones_like(prices) * 3600
        now = datetime.now()
        delta_t[0] -= now.minute * 60 + now.second

        # Create model parameters from heat pump settings
        model = IVT490.create_model_from_slope(cmd_args.heating_curve_slope)

        problem = opt.prepare_optimization_problem(
            model,
            prices,
            outdoor_temperatures,
            delta_t,
            T_indoor_current,
            config.T_indoor_requested,
            config.T_indoor_requested + config.T_indoor_bounds[0],
            config.T_indoor_requested + config.T_indoor_bounds[1],
            config.T_feed_maximum,
        )

        res = opt.solve_problem(
            problem,
            IVT490.make_initial_guess(
                cmd_args.heating_curve_slope, outdoor_temperatures
            ),
            maxiter=500,
        )

        logging.debug(res)

        if res.success:
            logging.info("New target feed temperature: %s", res.x[0])
            logging.debug(
                f"Indoor temperature forecast: {dynamic_model.simulate(model, T_indoor_current, res.x, outdoor_temperatures, delta_t)}"
            )
            return res.x[0]

        logging.error("Solver failed: %s", res.message)

    # Build pipeline

    source_outdoor_temperature = Stream.from_secured_mqtt(
        cmd_args.host,
        cmd_args.port,
        cmd_args.T_outdoor_topic,
        username=cmd_args.username,
        password=cmd_args.password,
    ).map(partial(resolve_jsonpointer, cmd_args.T_outdoor_jsonpointer))

    source_indoor_temperature = Stream.from_secured_mqtt(
        cmd_args.host,
        cmd_args.port,
        cmd_args.T_indoor_topic,
        username=cmd_args.username,
        password=cmd_args.password,
    ).map(partial(resolve_jsonpointer, cmd_args.T_indoor_jsonpointer))

    # Exception handling
    source_outdoor_temperature.on_exception(exception=ValueError).sink(print)
    source_indoor_temperature.on_exception(exception=ValueError).sink(print)

    opt_output = (
        streamz_nodes.combine_latest_with_timeout(
            source_outdoor_temperature,
            source_indoor_temperature,
            timeout=config.sensor_timeout,
        )
        .latest()
        .rate_limit(60)
        .map(run_optimization)
    )

    # Exception handling
    opt_output.on_exception().sink(print)

    sink = opt_output.to_secured_mqtt(
        cmd_args.host,
        cmd_args.port,
        cmd_args.T_feed_target_topic,
        username=cmd_args.username,
        password=cmd_args.password,
    )

    # Lets get this show on the road!
    sink.start()

    logging.info("Pipeline started!")

    while True:
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Esther - an economically smart thermostat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("host", type=str, help="Hostname of MQTT broker")
    parser.add_argument("port", type=int, help="Port number of MQTT broker")
    parser.add_argument(
        "-u",
        "--username",
        type=str,
        default=None,
        help="Username to use for accessing the MQTT broker",
    )
    parser.add_argument(
        "-p",
        "--password",
        type=str,
        default=None,
        help="Password to use for accessing the MQTT broker",
    )
    parser.add_argument(
        "--T-outdoor-topic",
        type=str,
        required=True,
        help="Topic on which to listen for outdoor temperature sensor values",
    )
    parser.add_argument(
        "--T-outdoor-jsonpointer",
        type=JsonPointer,
        default=JsonPointer(""),
        help="JsonPointer for resolving the value in the payload on T-outdoor-topic",
    )
    parser.add_argument(
        "--T-indoor-topic",
        type=str,
        required=True,
        help="Topic on which to listen for indoor temperature sensor values",
    )
    parser.add_argument(
        "--T-indoor-jsonpointer",
        type=JsonPointer,
        default=JsonPointer(""),
        help="JsonPointer for resolving the value in the payload on T-indoor-topic",
    )
    parser.add_argument(
        "--T-feed-target-topic",
        type=str,
        required=True,
        help="Topic on which to publish new optimal target values for the feed temperature",
    )
    parser.add_argument(
        "--sensor-timeout",
        type=float,
        default=config.sensor_timeout,
        help="Maximum allowed time (s) between sensor readings (T-indoor and T-outdoor). If exceeded, no new optimal feed temperature will be calculated and outputted until sensor readings are within the given timeout again.",
    )
    parser.add_argument(
        "--T-indoor-requested",
        type=float,
        default=config.T_indoor_requested,
        help="Requested indoor temperature",
    )
    parser.add_argument(
        "--T-indoor-bounds",
        type=tuple,
        default=config.T_indoor_bounds,
        help="Allowed bounds of indoor temperature relative to T-indoor-requested.",
    )
    parser.add_argument(
        "--T-feed-maximum",
        type=float,
        default=config.T_feed_maximum,
        help="Maximum allowable feed temperature",
    )
    parser.add_argument(
        "--nordpool-price-area",
        type=str,
        required=True,
        help="Nordpool price area, eg: SE3",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        required=True,
        help="Longitude for SMHI weather forecasts",
    )
    parser.add_argument(
        "--latitude",
        type=float,
        required=True,
        help="Latitude for SMHI weather forecasts",
    )
    parser.add_argument(
        "--heating-curve-slope",
        type=float,
        required=True,
        help="Heating curve slope value, IVT490-style.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    conf = parser.parse_args()

    logging.basicConfig(level=conf.loglevel)
    run(conf)
