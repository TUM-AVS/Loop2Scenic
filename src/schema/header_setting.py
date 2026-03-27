from pydantic import BaseModel


class HeaderSetting(BaseModel):
    """
    The header setting for the scenario.
    """
    carla_map: str = "Town05"
    map_file_path: str = "../../maps/Town05.xodr"
    blueprint: str = "vehicle.lincoln.mkz_2017"
    weather: str = "ClearNoon"