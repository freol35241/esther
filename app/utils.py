def T_eff(T_out: float, wind_speed: float) -> float:
    return (
        13.12
        + 0.61215 * T_out
        - 13.956 * wind_speed**0.16
        + 0.48669 * T_out * wind_speed**0.16
    )
