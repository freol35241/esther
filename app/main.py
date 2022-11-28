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
    T_indoor_bounds: Tuple[float] = (-1, 1)
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
    config.T_indoor_bounds = (
        cmd_args.T_indoor_bound_lower,
        cmd_args.T_indoor_bound_upper,
    )
    config.T_feed_maximum = cmd_args.T_feed_maximum
    config.sensor_timeout = cmd_args.sensor_timeout

    # Configure heating system model
    if cmd_args.T_feed_time_constant:
        heating_system_model = dynamic_model.ModelParameters(
            cmd_args.T_outdoor_time_constant, cmd_args.T_feed_time_constant
        )
        initial_guess_func = lambda x: np.ones_like(x) * 25
    elif cmd_args.T_feed_time_constant_from_IVT490_heating_curve_slope:
        heating_system_model = IVT490.create_model_from_slope(
            cmd_args.T_feed_time_constant_from_IVT490_heating_curve_slope,
            cmd_args.T_outdoor_time_constant,
        )
        initial_guess_func = partial(
            IVT490.make_initial_guess,
            cmd_args.T_feed_time_constant_from_IVT490_heating_curve_slope,
        )
    else:
        # Should not happen
        raise ValueError("Dont know how to setup heating system model!")

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

        problem = opt.prepare_optimization_problem(
            heating_system_model,
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
            initial_guess_func(outdoor_temperatures),
            maxiter=cmd_args.max_iter,
        )

        logging.debug(res)
        logging.debug(
            f"Indoor temperature forecast: {dynamic_model.simulate(heating_system_model, T_indoor_current, res.x, outdoor_temperatures, delta_t)}"
        )

        if not res.success:
            logging.error("Solver failed: %s", res.message)
            if not cmd_args.allow_failing_solutions:
                return

        logging.info("New target feed temperature: %s", res.x[0])
        return res.x[0]

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

    general_config_group = parser.add_argument_group(title="General configuration")
    general_config_group.add_argument(
        "--allow-failing-solutions",
        action="store_true",
        help="Allow failing solutions to the optimization problem to publish target feed temperatures.",
    )
    general_config_group.add_argument(
        "--sensor-timeout",
        type=float,
        default=config.sensor_timeout,
        help="Maximum allowed time (s) between sensor readings (T-indoor and T-outdoor). If exceeded, no new optimal feed temperature will be calculated and outputted until sensor readings are within the given timeout again.",
    )
    general_config_group.add_argument(
        "--T-indoor-requested",
        type=float,
        default=config.T_indoor_requested,
        help="Requested indoor temperature",
    )
    general_config_group.add_argument(
        "--T-indoor-bound-lower",
        type=float,
        default=config.T_indoor_bounds[0],
        help="Allowed bounds of indoor temperature relative to T-indoor-requested.",
    )
    general_config_group.add_argument(
        "--T-indoor-bound-upper",
        type=float,
        default=config.T_indoor_bounds[1],
        help="Allowed bounds of indoor temperature relative to T-indoor-requested.",
    )
    general_config_group.add_argument(
        "--T-feed-maximum",
        type=float,
        default=config.T_feed_maximum,
        help="Maximum allowable feed temperature",
    )
    general_config_group.add_argument(
        "--nordpool-price-area",
        type=str,
        required=True,
        help="Nordpool price area, eg: SE3",
    )
    general_config_group.add_argument(
        "--longitude",
        type=float,
        required=True,
        help="Longitude for SMHI weather forecasts",
    )
    general_config_group.add_argument(
        "--latitude",
        type=float,
        required=True,
        help="Latitude for SMHI weather forecasts",
    )

    mqtt_group = parser.add_argument_group(title="MQTT connection configuration")
    mqtt_group.add_argument(
        "--host", type=str, required=True, help="Hostname of MQTT broker"
    )
    mqtt_group.add_argument(
        "--port", type=int, default=1883, help="Port number of MQTT broker"
    )
    mqtt_group.add_argument(
        "--username",
        type=str,
        default=None,
        help="Username to use for accessing the MQTT broker",
    )
    mqtt_group.add_argument(
        "--password",
        type=str,
        default=None,
        help="Password to use for accessing the MQTT broker",
    )

    mqtt_api_group = parser.add_argument_group(title="MQTT API configuration")
    mqtt_api_group.add_argument(
        "--T-outdoor-topic",
        type=str,
        required=True,
        help="Topic on which to listen for outdoor temperature sensor values",
    )
    mqtt_api_group.add_argument(
        "--T-outdoor-jsonpointer",
        type=JsonPointer,
        default=JsonPointer(""),
        help="JsonPointer for resolving the value in the payload on T-outdoor-topic",
    )
    mqtt_api_group.add_argument(
        "--T-indoor-topic",
        type=str,
        required=True,
        help="Topic on which to listen for indoor temperature sensor values",
    )
    mqtt_api_group.add_argument(
        "--T-indoor-jsonpointer",
        type=JsonPointer,
        default=JsonPointer(""),
        help="JsonPointer for resolving the value in the payload on T-indoor-topic",
    )
    mqtt_api_group.add_argument(
        "--T-feed-target-topic",
        type=str,
        required=True,
        help="Topic on which to publish new optimal target values for the feed temperature",
    )

    thermal_group = parser.add_argument_group(title="Heating system configuration")
    thermal_group.add_argument(
        "--T-outdoor-time-constant",
        type=float,
        default=dynamic_model.DEFAULT_HOUSE_COOLDOWN_TIME_CONSTANT,
        help="Time constant (hours) of changes in indoor temperature subject to changes in outdoor temperature",
    )
    thermal_exclusive_group = thermal_group.add_mutually_exclusive_group(required=True)
    thermal_exclusive_group.add_argument(
        "--T-feed-time-constant",
        type=float,
        help="Time constant (hours) of changes in indoor temperature subject to changes in feed temperature",
    )
    thermal_exclusive_group.add_argument(
        "--T-feed-time-constant-from-IVT490-heating-curve-slope",
        type=float,
        help="Heating curve slope value, IVT490-style.",
    )

    opt_group = parser.add_argument_group(title="Optimization algorithm configuration")
    opt_group.add_argument(
        "--max-iter",
        type=float,
        default=500,
        help="Maximum number of iterations allowed in the optimization algorithms",
    )

    conf = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s", level=conf.loglevel
    )
    logging.debug("Parsed command line arguments:")
    logging.debug(conf)
    run(conf)
