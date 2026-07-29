from typing import Dict, Union

from pydantic import BaseModel, Field


# Fog-free night for BEV/recording: dark (no sun) but no CARLA night fog.
FOG_FREE_NIGHT_WEATHER: Dict[str, float] = {
    "cloudiness": 10.0,
    "precipitation": 0.0,
    "precipitation_deposits": 0.0,
    "wind_intensity": 0.0,
    "sun_azimuth_angle": 0.0,
    "sun_altitude_angle": -90.0,
    "fog_density": 0.0,
    "fog_distance": 0.0,
    "wetness": 0.0,
    "fog_falloff": 0.0,
}

# Sentinel the VLM / pipeline may emit; expanded to FOG_FREE_NIGHT_WEATHER.
CUSTOM_NIGHT_WEATHER = "CustomNight"

WeatherParam = Union[str, Dict[str, float]]


class HeaderSetting(BaseModel):
    """
    The header setting for the scenario.
    weather may be a CARLA preset name (e.g. ClearNoon) or a param dict
    (used for fog-free CustomNight).
    """
    carla_map: str = "Town05"
    map_file_path: str = "../../maps/Town05.xodr"
    blueprint: str = "vehicle.lincoln.mkz_2017"
    weather: WeatherParam = Field(default="ClearNoon")
