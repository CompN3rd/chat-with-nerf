import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from attrs import define


from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel

from chat_with_nerf import logger
from chat_with_nerf.model.scene_config import SceneConfig
from chat_with_nerf.settings import Settings
from chat_with_nerf.visual_grounder.captioner import (  # Blip2Captioner,
    BaseCaptioner,
)
from chat_with_nerf.visual_grounder.picture_taker import (
    PictureTaker,
    PictureTakerFactory,
)

from chat_with_nerf.settings import Settings


@define
class ModelContext:
    scene_configs: dict[str, SceneConfig]
    picture_takers: dict[str, PictureTaker]
    captioner: BaseCaptioner


class ModelContextManager:
    model_context: Optional[ModelContext] = None

    @classmethod
    def get_model_context(cls, scene_name) -> ModelContext:
        return ModelContextManager.initialize_model_context(scene_name)

    @classmethod
    def get_model_no_gpt_context(cls, scene_name) -> ModelContext:
        if Settings.IS_EVALUATION:
            return ModelContextManager.initialize_model_no_gpt_context(scene_name)
        elif cls.model_context is None:
            cls.model_context = ModelContextManager.initialize_model_no_gpt_context(
                scene_name
            )
        return cls.model_context

    @classmethod
    def get_model_no_visual_feedback_context(cls, scene_name) -> ModelContext:
        if Settings.IS_EVALUATION:
            return ModelContextManager.initialize_model_no_visual_feedback_context(
                scene_name
            )
        elif cls.model_context is None:
            cls.model_context = (
                ModelContextManager.initialize_model_no_visual_feedback_context(
                    scene_name
                )
            )
        return cls.model_context

    @classmethod
    def get_model_context_with_gpt(cls) -> ModelContext:
        if Settings.IS_EVALUATION:
            return (
                ModelContextManager.initialize_model_no_visual_feedback_openscene_context()
            )
        elif cls.model_context is None:
            cls.model_context = (
                ModelContextManager.initialize_model_no_visual_feedback_openscene_context()
            )
        return cls.model_context

    @classmethod
    def initialize_model_no_visual_feedback_openscene_context(
        cls,
    ) -> ModelContext:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.append(project_root)
        logger.info("Search for all Scenes and Set the current Scene")
        scene_configs = ModelContextManager.search_scenes(Settings.data_path)
        picture_taker_dict = (
            PictureTakerFactory.get_picture_takers_no_visual_feedback_openscene(
                scene_configs
            )
        )
        return ModelContext(scene_configs, picture_taker_dict, None)

    @staticmethod
    def initialize_model_no_gpt_context(scene_name: str) -> ModelContext:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.append(project_root)
        logger.info("Search for all Scenes and Set the current Scene")
        scene_configs = ModelContextManager.search_scenes(
            Settings.data_path, scene_name
        )
        picture_taker_dict = PictureTakerFactory.get_picture_takers_no_gpt(
            scene_configs
        )
        return ModelContext(scene_configs, picture_taker_dict, None)

    @staticmethod
    def initialize_model_no_visual_feedback_context(scene_name: str) -> ModelContext:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.append(project_root)
        logger.info("Search for all Scenes and Set the current Scene")
        scene_configs = ModelContextManager.search_scenes(
            Settings.data_path, scene_name
        )
        picture_taker_dict = PictureTakerFactory.get_picture_takers_no_visual_feedback(
            scene_configs
        )
        return ModelContext(scene_configs, picture_taker_dict, None)

    @staticmethod
    def initialize_model_context(scene_name: str) -> ModelContext:
        # Get the absolute path of the project's root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Add the project's root directory to sys.path
        sys.path.append(project_root)

        logger.info("Search for all Scenes and Set the current Scene")
        scene_configs = ModelContextManager.search_scenes(
            Settings.data_path, scene_name
        )

        logger.info("Initialize picture_taker for all scenes")
        picture_taker_dict = PictureTakerFactory.get_picture_takers(scene_configs)

        return ModelContext(scene_configs, picture_taker_dict, None)

    @staticmethod
    def search_scenes(path: str) -> dict[str, SceneConfig]:
        scenes = {}
        subdirectories = [
            name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
        ]
        for subdir in subdirectories:
            try:
                scene_path = (Path(path) / subdir / subdir).with_suffix(".yaml")
                logger.info(f"scene_path: {scene_path}")
                with open(scene_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                scene = SceneConfig(
                    subdir,
                    data["load_lerf_config"],
                    data["load_embedding"],
                    data["camera_path"],
                    data["nerf_exported_mesh_path"],
                    data["load_openscene"],
                    data["load_mesh"],
                    data["load_metadata"],
                )
                scenes[subdir] = scene
            except FileNotFoundError:
                raise ValueError(f"Scene {subdir} not found in {path}")

        return scenes

