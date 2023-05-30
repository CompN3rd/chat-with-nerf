from attrs import define


@define
class SceneConfig:
    scene_name: str
    load_lerf_config: str
    load_h5_config: str
    camera_path: str
