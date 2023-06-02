import datetime
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import open3d as o3d
import h5py
import mediapy as media
import numpy as np
import open_clip
import torch
import trimesh
from attrs import define
from nerfstudio.cameras.camera_paths import get_path_from_json
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import install_checks

# from nerfstudio.cameras.cameras import CameraType
from nerfstudio.utils.eval_utils import eval_setup
from sklearn.cluster import DBSCAN
from torch import Tensor
from transformers import AutoTokenizer, CLIPVisionModel

from chat_with_nerf import logger
from chat_with_nerf.model.scene_config import SceneConfig
from chat_with_nerf.settings import Settings
from chat_with_nerf.visual_grounder.camera_pose import CameraPose
from chat_with_nerf.visual_grounder.image_ref import ImageRef


@define
class PictureTaker:
    scene: str
    lerf_pipeline: Pipeline
    h5_dict: dict
    clip_model: CLIPVisionModel
    tokenizer: AutoTokenizer
    neg_embeds: Tensor
    negative_words_length: int
    thread_pool_executor: ThreadPoolExecutor

    @staticmethod
    def render_picture(
        lerf_pipeline: Pipeline, camera_pose: dict, session_id: str
    ) -> ImageRef:
        logger.info("Picture Taking...")
        install_checks.check_ffmpeg_installed()
        camera = get_path_from_json(camera_pose)
        # camera_type = CameraType.PESPECTIVE
        camera.rescale_output_resolution(1.0)
        camera = camera.to(lerf_pipeline.device)
        output_filepath_path = Path(Settings.output_path) / session_id / "images"
        rgb_image_dir = output_filepath_path / "rgb"
        rgb_image_dir.mkdir(parents=True, exist_ok=True)

        result = {}
        camera_idx = 0
        aabb_box = None
        camera_ray_bundle = camera.generate_rays(
            camera_indices=camera_idx, aabb_box=aabb_box
        )
        with torch.no_grad():
            outputs = lerf_pipeline.model.get_outputs_for_camera_ray_bundle(
                camera_ray_bundle.to(lerf_pipeline.device)
            )

        output_image = outputs["rgb"].cpu().numpy()
        print(output_image.shape)

        if output_image.shape[-1] == 1:
            output_image = np.concatenate((output_image,) * 3, axis=-1)

        # Get the current timestamp
        now = datetime.datetime.now()
        # Format the timestamp as a string
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        # saving rgb
        rgb = "rgb" + str(camera_idx)
        # create file name
        rgb_filename = rgb + "_" + timestamp_str + ".png"
        result[rgb] = str(rgb_image_dir) + "/" + rgb_filename
        media.write_image(result[rgb], output_image)

        imageRef = ImageRef(result[rgb], output_image)

        return imageRef

    def take_picture(self, query: str, session_id: str) -> list[ImageRef]:
        positives = [query]
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in positives]
            ).to("cuda")
            pos_embeds = self.clip_model.encode_text(tok_phrases)
        pos_embeds /= pos_embeds.norm(dim=-1, keepdim=True)
        # use query to dot product with the point cloud -> centroids
        scales_list = torch.linspace(0.0, 1.5, 30)

        n_phrases = len(positives)
        n_phrases_maxs: list[None | Tensor] = [None for _ in range(n_phrases)]
        n_phrases_sims: list[None | Tensor] = [None for _ in range(n_phrases)]
        for i, scale in enumerate(scales_list):
            clip_output = torch.from_numpy(
                self.h5_dict["clip_embeddings_per_scale"][i]
            ).to("cuda")
            for i in range(n_phrases):
                probs = self.get_relevancy(
                    embed=clip_output,
                    positive_id=i,
                    pos_embeds=pos_embeds,
                    neg_embeds=self.neg_embeds,
                    positive_words_length=1,
                )
                pos_prob = probs[..., 0:1]
                if (
                    n_phrases_maxs[i] is None
                    or pos_prob.max() > n_phrases_sims[i].max()  # type: ignore
                ):
                    n_phrases_maxs[i] = scale
                    n_phrases_sims[i] = pos_prob

        possibility_array = n_phrases_sims[0].detach().cpu().numpy()  # type: ignore
        num_points = possibility_array.shape[0]
        percentage_points = int(num_points * 0.005)
        flattened_values = possibility_array.flatten()

        # # Find the indices of the top 5% values
        top_indices = np.argpartition(flattened_values, -percentage_points)[
            -percentage_points:
        ]
        logger.info(
            "Export RGB GLB files highlighted the top 5% values with color of red..."
        )

        points = self.h5_dict["points"]
        values = self.h5_dict["values"]
        colors = self.h5_dict["rgb"]
        origins = self.h5_dict["origins"]
        # directions = self.h5_dict["directions"]

        # Assuming you have the indices of the selected points
        selected_indices = top_indices

        # Set the color of selected points to red and make them fully opaque
        red_color = np.tile(np.array([1.0, 0.0, 0.0]), (len(selected_indices), 1))
        # opacity = 1.0  # Fully opaque
        colors[selected_indices, :] = red_color

        mesh = o3d.io.read_triangle_mesh(
            "/workspace/dev/home_1_nerfacto_poisson_export/poisson_mesh.ply"
        )
        top_positions = points[top_indices]
        top_values = values[top_indices]
        top_origins = origins[top_indices]
        points = top_positions  # Random points in [0, 1)
        # positions = points  # Replace with your positions
        colors = red_color  # Replace with your colors

        # Create a PointCloud object
        pcd = o3d.geometry.PointCloud()

        # Set the positions and colors
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        radius = 0.02  # You may need to adjust this
        bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([radius, radius * 2])
        )
        combined_mesh = bpa_mesh + mesh

        o3d.io.write_triangle_mesh("mesh.glb", combined_mesh, write_vertex_colors=True)
        logger.info("Clustering...")

        # Apply DBSCAN clustering
        epsilon = 0.05  # Radius of the neighborhood
        min_samples = 100  # Minimum number of samples in a cluster
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        clusters = dbscan.fit(top_positions)

        labels = clusters.labels_

        centroids = []
        top_values_in_clusters = []
        top_points_in_clusters = []
        max_index = []
        # Iterate over each cluster ID
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise
                continue
            else:
                # Compute the centroid of each cluster
                members = top_positions[
                    labels == cluster_id
                ]  # Get all members of the cluster
                centroids.append(members.mean(axis=0))  # Compute centroid

                # Find the top value and corresponding point in the cluster
                member_values = top_values[labels == cluster_id]
                max_value_index = np.argmax(member_values)
                max_index.append(max_value_index)
                top_values_in_clusters.append(member_values[max_value_index])
                top_points_in_clusters.append(members[max_value_index])

        top_origins_list = []

        for index in max_index:
            top_origins_list.append(top_origins[max_index, :])
        # centroids -> camera poses
        assert n_phrases_maxs[0] is not None
        c2w_list = [
            self.compute_camera_to_world_matrix(
                centroid, origin, n_phrases_maxs[0].item()
            )
            for centroid, origin in zip(centroids, top_origins_list)
        ]
        camera_pose_instance = CameraPose()
        camera_poses = [
            camera_pose_instance.construct_camera_pose(c2w) for c2w in c2w_list
        ]

        lerf_pipelines = [self.lerf_pipeline] * len(camera_poses)
        session_id_list = [session_id] * len(camera_poses)
        # camera pose -> render pictures
        # put it in a 'with' block so that we will wait on all threads to finish
        # see https://stackoverflow.com/a/70003564
        with self.thread_pool_executor as exe:
            picture_paths: Iterator[ImageRef] = exe.map(
                lambda tup: PictureTaker.render_picture(*tup),
                (
                    (lerf_pipeline, camera_pose, session_id)
                    for lerf_pipeline, camera_pose, session_id in zip(
                        lerf_pipelines, camera_poses, session_id_list
                    )
                ),
            )

        return list(picture_paths)

    def get_relevancy(
        self,
        embed: torch.Tensor,
        positive_id: int,
        pos_embeds: Tensor,
        neg_embeds: Tensor,
        positive_words_length: int,
    ) -> torch.Tensor:
        phrases_embeds = torch.cat([pos_embeds, neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # noqa E501
        negative_vals = output[..., positive_words_length:]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(
            1, self.negative_words_length
        )  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(
            softmax,
            1,
            best_id[..., None, None].expand(
                best_id.shape[0], self.negative_words_length, 2
            ),
        )[:, 0, :]

    def compute_camera_to_world_matrix(
        self, point: np.ndarray, origin: np.ndarray, k: float
    ) -> np.ndarray:
        epsilon = 1e-6
        direction = point - origin
        direction = direction / np.linalg.norm(direction)

        camera_position = point - (k * direction)

        up = np.array([0, 1, 0])

        right = np.cross(direction, up)
        right /= np.linalg.norm(right) + epsilon

        new_up = np.cross(right, direction)
        new_up /= np.linalg.norm(new_up) + epsilon

        camera_to_world = np.eye(4)
        camera_to_world[:3, 0] = right
        camera_to_world[:3, 1] = new_up
        camera_to_world[:3, 2] = -direction
        camera_to_world[:3, 3] = camera_position

        return camera_to_world.flatten()


class PictureTakerFactory:
    picture_taker_dict: Optional[dict[str, PictureTaker]] = None

    @classmethod
    def get_picture_takers(
        cls, scene_configs: dict[str, SceneConfig]
    ) -> dict[str, PictureTaker]:
        if cls.picture_taker_dict is None:
            cls.picture_taker_dict = PictureTakerFactory.initialize_picture_takers(
                scene_configs
            )
        return cls.picture_taker_dict

    @staticmethod
    def initialize_picture_takers(
        scene_configs: dict[str, SceneConfig],
    ) -> dict[str, PictureTaker]:
        picture_taker_dict = {}
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16",  # e.g., ViT-B-16
            pretrained="laion2b_s34b_b88k",  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        model = model.to("cuda")
        tokenizer = open_clip.get_tokenizer("ViT-B-16")

        negatives = ["object", "things", "stuff", "texture"]
        with torch.no_grad():
            tok_phrases = torch.cat([tokenizer(phrase) for phrase in negatives]).to(
                "cuda"
            )
            neg_embeds = model.encode_text(tok_phrases)
        neg_embeds /= neg_embeds.norm(dim=-1, keepdim=True)

        for scene_name, scene_config in scene_configs.items():
            lerf_pipeline = PictureTakerFactory.initialize_lerf_pipeline(
                scene_config.load_lerf_config, scene_name
            )
            h5_dict = PictureTakerFactory.load_h5_file(scene_config.load_h5_config)
            thread_pool_executor = ThreadPoolExecutor(max_workers=Settings.MAX_WORKERS)
            picture_taker_dict[scene_name] = PictureTaker(
                scene=scene_config.scene_name,
                lerf_pipeline=lerf_pipeline,
                h5_dict=h5_dict,
                clip_model=model,
                tokenizer=tokenizer,
                neg_embeds=neg_embeds,
                negative_words_length=len(negatives),
                thread_pool_executor=thread_pool_executor,
            )

        return picture_taker_dict

    @staticmethod
    def initialize_lerf_pipeline(load_config: str, scene_name: str) -> Pipeline:
        initial_dir = os.getcwd()
        os.chdir(Settings.data_path + "/" + scene_name)
        _, lerf_pipeline, _, _ = eval_setup(
            Path(load_config),
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )
        os.chdir(initial_dir)
        return lerf_pipeline

    @staticmethod
    def load_h5_file(load_config: str) -> dict:
        hdf5_file = h5py.File(load_config, "r")
        # batch_idx = 5
        points = hdf5_file["points"]["points"][:]
        origins = hdf5_file["origins"]["origins"][:]
        directions = hdf5_file["directions"]["directions"][:]

        clip_embeddings_per_scale = []

        clips_group = hdf5_file["clip"]
        for i in range(30):
            clip_embeddings_per_scale.append(clips_group[f"scale_{i}"][:])

        hdf5_file.close()
        h5_dict = {
            "points": points,
            "origins": origins,
            "directions": directions,
            "clip_embeddings_per_scale": clip_embeddings_per_scale,
        }
        return h5_dict
