from collections import OrderedDict
from typing import List, Optional

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.common import random_choice
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import vectorize_pose

from .base_env import CustomOtherObjectsInSceneEnv, CustomSceneEnv


class GraspMultipleInSceneEnv(CustomSceneEnv):
    objects: List[sapien.Actor]  # target objects to grasp
    num_objects: int
    model_ids: List[str]
    model_scales: List[float]
    model_bbox_sizes: List[Optional[np.ndarray]]

    def __init__(
        self,
        require_lifting_obj_for_success: bool = True,
        success_from_episode_stats: bool = True,
        num_objects: int = 3,
        **kwargs,
    ):
        self.objects = []
        self.num_objects = num_objects
        self.model_ids = []
        self.model_scales = []
        self.model_bbox_sizes = []

        self.obj_init_options = {}

        self.require_lifting_obj_for_success = require_lifting_obj_for_success
        self.success_from_episode_stats = success_from_episode_stats
        self.consecutive_grasps = [0] * num_objects
        self.lifted_objs = [False] * num_objects
        self.obj_heights_after_settle = [None] * num_objects

        self.episode_stats = None

        super().__init__(**kwargs)

    def _load_actors(self):
        self._load_arena_helper()
        self._load_models()
        for obj in self.objects:
            obj.set_damping(0.1, 0.1)

    def _load_models(self):
        """Load multiple target objects."""
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.obj_init_options = options.get("obj_init_options", {})

        self.set_episode_rng(seed)
        self._set_models(options.get("model_ids"), options.get("model_scales"))

        self.consecutive_grasps = [0] * self.num_objects
        self.lifted_objs = [False] * self.num_objects
        self.obj_heights_after_settle = [None] * self.num_objects
        self._initialize_episode_stats()

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update(
            {
                "model_ids": self.model_ids,
                "model_scales": self.model_scales,
                "obj_init_poses_wrt_robot_base": [
                    self.agent.robot.pose.inv() * obj.pose for obj in self.objects
                ],
            }
        )
        return obs, info

    def _set_models(self, model_ids, model_scales):
        if model_ids is None:
            model_ids = [
                random_choice(self.model_ids, self._episode_rng)
                for _ in range(self.num_objects)
            ]
        self.model_ids = model_ids

        if model_scales is None:
            model_scales = []
            for model_id in model_ids:
                scales = self.model_db[model_id].get("scales")
                scale = (
                    1.0 if scales is None else random_choice(scales, self._episode_rng)
                )
                model_scales.append(scale)
        self.model_scales = model_scales

        self.model_bbox_sizes = []
        for model_id, model_scale in zip(self.model_ids, self.model_scales):
            model_info = self.model_db[model_id]
            if "bbox" in model_info:
                bbox = model_info["bbox"]
                bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
                self.model_bbox_sizes.append(bbox_size * model_scale)
            else:
                self.model_bbox_sizes.append(None)

    def _initialize_actors(self):
        for i, (obj, model_id, model_scale) in enumerate(
            zip(self.objects, self.model_ids, self.model_scales)
        ):
            obj_init_xy = self.obj_init_options.get(f"init_xy_{i}", None)
            if obj_init_xy is None:
                obj_init_xy = self._episode_rng.uniform(
                    [-0.35, -0.02], [-0.12, 0.42], [2]
                )
            obj_init_z = self.obj_init_options.get(
                f"init_z_{i}", self.scene_table_height
            )
            obj_init_z = obj_init_z + 0.5  # let object fall onto the table
            obj_init_rot_quat = self.obj_init_options.get(
                f"init_rot_quat_{i}", [1, 0, 0, 0]
            )
            p = np.hstack([obj_init_xy, obj_init_z])
            q = obj_init_rot_quat

            # Rotate along z-axis
            if self.obj_init_options.get(f"init_rand_rot_z_{i}", False):
                ori = self._episode_rng.uniform(0, 2 * np.pi)
                q = qmult(euler2quat(0, 0, ori), q)

            # Rotate along a random axis by a small angle
            if (
                init_rand_axis_rot_range := self.obj_init_options.get(
                    f"init_rand_axis_rot_range_{i}", 0.0
                )
            ) > 0:
                axis = self._episode_rng.uniform(-1, 1, 3)
                axis = axis / max(np.linalg.norm(axis), 1e-6)
                ori = self._episode_rng.uniform(0, init_rand_axis_rot_range)
                q = qmult(q, axangle2quat(axis, ori, True))
            obj.set_pose(Pose(p, q))

        # Move the robot far away to avoid collision
        self.agent.robot.set_pose(Pose([-10, 0, 0]))

        # Let objects fall and settle
        for obj in self.objects:
            obj.lock_motion(0, 0, 0, 1, 1, 0)
        self._settle(0.5)

        for obj in self.objects:
            obj.lock_motion(0, 0, 0, 0, 0, 0)
            obj.set_pose(obj.pose)
            obj.set_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

        # Record object heights after settling
        self.obj_heights_after_settle = [obj.pose.p[2] for obj in self.objects]

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_poses=[vectorize_pose(obj.pose) for obj in self.objects],
                tcp_to_obj_poses=[obj.pose.p - self.tcp.pose.p for obj in self.objects],
            )
        return obs

    def evaluate(self, **kwargs):
        successes = []
        for i, obj in enumerate(self.objects):
            is_grasped = self.agent.check_grasp(obj, max_angle=80)
            if is_grasped:
                self.consecutive_grasps[i] += 1
            else:
                self.consecutive_grasps[i] = 0
                self.lifted_objs[i] = False

            contacts = self._scene.get_contacts()
            flag = True
            robot_link_names = [x.name for x in self.agent.robot.get_links()]
            for contact in contacts:
                actor_0, actor_1 = contact.actor0, contact.actor1
                if actor_0.name == obj.name or actor_1.name == obj.name:
                    other_actor = actor_1 if actor_0.name == obj.name else actor_0
                    if other_actor.name not in robot_link_names:
                        contact_impulse = np.sum(
                            [point.impulse for point in contact.points], axis=0
                        )
                        if np.linalg.norm(contact_impulse) > 1e-6:
                            flag = False
                            break

            consecutive_grasp = self.consecutive_grasps[i] >= 5
            diff_obj_height = obj.pose.p[2] - self.obj_heights_after_settle[i]
            self.lifted_objs[i] = self.lifted_objs[i] or (
                flag and (diff_obj_height > 0.01)
            )
            lifted_object_significantly = self.lifted_objs[i] and (
                diff_obj_height > 0.02
            )

            if self.require_lifting_obj_for_success:
                success = self.lifted_objs[i]
            else:
                success = consecutive_grasp

            self.episode_stats[f"n_lift_significant_{i}"] += int(
                lifted_object_significantly
            )
            self.episode_stats[f"consec_grasp_{i}"] = (
                self.episode_stats[f"consec_grasp_{i}"] or consecutive_grasp
            )
            self.episode_stats[f"grasped_{i}"] = (
                self.episode_stats[f"grasped_{i}"] or is_grasped
            )

            if self.success_from_episode_stats:
                success = success or (
                    self.episode_stats[f"n_lift_significant_{i}"] >= 5
                )

            successes.append(success)

        return dict(
            is_grasped=[
                self.agent.check_grasp(obj, max_angle=80) for obj in self.objects
            ],
            consecutive_grasp=self.consecutive_grasps,
            lifted_object=self.lifted_objs,
            lifted_object_significantly=[
                self.lifted_objs[i]
                and (obj.pose.p[2] - self.obj_heights_after_settle[i] > 0.02)
                for i, obj in enumerate(self.objects)
            ],
            success=all(successes),
            individual_successes=successes,
            episode_stats=self.episode_stats,
        )

    def _initialize_episode_stats(self):
        self.episode_stats = OrderedDict()
        for i in range(self.num_objects):
            self.episode_stats.update(
                {
                    f"n_lift_significant_{i}": 0,
                    f"consec_grasp_{i}": False,
                    f"grasped_{i}": False,
                }
            )


@register_env("GraspMultipleCustomInScene-v0", max_episode_steps=120)
class GraspMultipleCustomInSceneEnv(
    GraspMultipleInSceneEnv, CustomOtherObjectsInSceneEnv
):
    def _load_models(self):
        for model_id, model_scale in zip(self.model_ids, self.model_scales):
            # NOTE: it is a hard code now, to lower the density of objects
            density = 0.01 * self.model_db[model_id].get("density", 1)
            print(model_id, model_scale, density)
            obj = self._build_actor_helper(
                model_id,
                self._scene,
                scale=model_scale,
                density=density,
                physical_material=self._scene.create_physical_material(
                    static_friction=self.obj_static_friction,
                    dynamic_friction=self.obj_dynamic_friction,
                    restitution=0.0,
                ),
                root_dir=self.asset_root,
            )
            obj.name = f"{model_id}_{len(self.objects)}"
            self.objects.append(obj)

    def _get_init_z(self):
        return [
            -self.model_db[model_id]["bbox"]["min"][2] * model_scale + 0.05
            for model_id, model_scale in zip(self.model_ids, self.model_scales)
        ]

    def get_language_instruction(self, **kwargs):
        obj_names = [self._get_instruction_obj_name(obj.name) for obj in self.objects]
        task_description = f"pick {', '.join(obj_names[:-1])}, and {obj_names[-1]}"
        return task_description


@register_env("GraspMultipleCustomOrientationInScene-v0", max_episode_steps=120)
class GraspMultipleCustomOrientationInSceneEnv(GraspMultipleCustomInSceneEnv):
    def __init__(
        self,
        num_objects: int = 3,
        default_orientation: str = None,
        **kwargs,
    ):
        self.default_orientation = default_orientation
        self.orientations = [None] * num_objects
        self.orientations_dict = {
            "upright": euler2quat(np.pi / 2, 0, 0),
            "laid_vertically": euler2quat(0, 0, np.pi / 2),
            "lr_switch": euler2quat(0, 0, np.pi),
        }
        super().__init__(num_objects=num_objects, **kwargs)

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()
        self.set_episode_rng(seed)

        obj_init_options = options.get("obj_init_options", None)
        if obj_init_options is None:
            obj_init_options = dict()
        obj_init_options = (
            obj_init_options.copy()
        )  # avoid modifying the original options

        for i in range(self.num_objects):
            orientation = None
            if obj_init_options.get(f"init_rot_quat_{i}", None) is None:
                if obj_init_options.get(f"orientation_{i}", None) is not None:
                    orientation = obj_init_options[f"orientation_{i}"]
                else:
                    orientation = self.default_orientation

                if orientation is not None:
                    try:
                        obj_init_options[f"init_rot_quat_{i}"] = self.orientations_dict[
                            orientation
                        ]
                    except KeyError as e:
                        if "standing" in orientation:
                            obj_init_options[f"init_rot_quat_{i}"] = (
                                self.orientations_dict["upright"]
                            )
                        elif "horizontal" in orientation:
                            obj_init_options[f"init_rot_quat_{i}"] = (
                                self.orientations_dict["lr_switch"]
                            )
                        else:
                            raise e
                else:
                    orientation = self._episode_rng.choice(
                        list(self.orientations_dict.keys())
                    )
                    obj_init_options[f"init_rot_quat_{i}"] = self.orientations_dict[
                        orientation
                    ]

            self.orientations[i] = orientation

        options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update({"orientations": self.orientations})
        return obs, info

    def get_language_instruction(self, **kwargs):
        obj_descriptions = []
        for i, obj in enumerate(self.objects):
            obj_name = self._get_instruction_obj_name(obj.name)
            orientation = self.orientations[i]
            if orientation == "upright":
                obj_descriptions.append(f"upright {obj_name}")
            elif orientation == "laid_vertically":
                obj_descriptions.append(f"vertically laid {obj_name}")
            elif orientation == "lr_switch":
                obj_descriptions.append(f"horizontally laid {obj_name}")
            else:
                obj_descriptions.append(obj_name)

        if len(obj_descriptions) == 1:
            task_description = f"pick the {obj_descriptions[0]}"
        elif len(obj_descriptions) == 2:
            task_description = (
                f"pick the {obj_descriptions[0]} and the {obj_descriptions[1]}"
            )
        else:
            task_description = f"pick the {', '.join(obj_descriptions[:-1])}, and the {obj_descriptions[-1]}"

        return task_description


"""
0,-1,0,0
0,0,-1,0
"""


@register_env("GraspMultipleRandomObjectsInScene-v0", max_episode_steps=120)
class GraspMultipleRandomObjectsInSceneEnv(GraspMultipleCustomOrientationInSceneEnv):
    def __init__(self, num_objects=3, **kwargs):
        self.available_model_ids = [
            "opened_pepsi_can",
            "opened_coke_can",
            "opened_sprite_can",
            "opened_fanta_can",
            "opened_redbull_can",
            "blue_plastic_bottle",
            "apple",
            "orange",
            "sponge",
            "bridge_spoon_generated_modified",
            "bridge_carrot_generated_modified",
            "bridge_spoon_generated_modified",
            "green_cube_3cm",
            "yellow_cube_3cm",
            "eggplant",
        ]
        kwargs.pop("model_ids", None)
        super().__init__(num_objects=num_objects, **kwargs)

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        # Ensure the episode RNG is set before using it
        self.set_episode_rng(seed)

        selected_model_ids = self._episode_rng.choice(
            self.available_model_ids, size=self.num_objects, replace=False
        )
        options["model_ids"] = selected_model_ids

        return super().reset(seed=seed, options=options)


@register_env("GraspMultipleSameObjectsInScene-v0", max_episode_steps=120)
class GraspMultipleSameObjectsInSceneEnv(GraspMultipleCustomOrientationInSceneEnv):
    def __init__(self, num_objects=8, model_id="opened_coke_can", **kwargs):
        self.fixed_model_id = model_id
        super().__init__(num_objects=num_objects, **kwargs)
        # self._scene.set_ambient_light([1.5, 1.5, 1.5])

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        # Ensure the episode RNG is set before using it
        self.set_episode_rng(seed)

        # Create a list of the same model ID repeated num_objects times
        selected_model_ids = [self.fixed_model_id] * self.num_objects
        options["model_ids"] = selected_model_ids

        # If we want to vary scales, we can do that here
        if "model_scales" not in options:
            scales = self.model_db[self.fixed_model_id].get("scales")
            if scales:
                selected_scales = self._episode_rng.choice(
                    scales, size=self.num_objects
                )
                options["model_scales"] = selected_scales

        # Handle custom positions and orientations
        obj_init_options = options.get("obj_init_options", {})
        for i in range(self.num_objects):
            # Set custom position if provided
            if f"init_xy_{i}" in obj_init_options:
                options.setdefault("obj_init_options", {})[f"init_xy_{i}"] = (
                    obj_init_options[f"init_xy_{i}"]
                )

            # Set custom orientation if provided
            if f"orientation_{i}" in obj_init_options:
                options.setdefault("obj_init_options", {})[f"orientation_{i}"] = (
                    obj_init_options[f"orientation_{i}"]
                )
            elif f"init_rot_quat_{i}" in obj_init_options:
                options.setdefault("obj_init_options", {})[f"init_rot_quat_{i}"] = (
                    obj_init_options[f"init_rot_quat_{i}"]
                )

        return super().reset(seed=seed, options=options)

    def get_language_instruction(self, **kwargs):
        obj_name = self._get_instruction_obj_name(self.fixed_model_id)
        if self.num_objects == 1:
            task_description = f"pick the {obj_name}"
        else:
            task_description = f"pick {self.num_objects} {obj_name}s"
        return task_description


@register_env("GraspMultipleDifferentObjectsInScene-v0", max_episode_steps=120)
class GraspMultipleDifferentObjectsInSceneEnv(GraspMultipleCustomOrientationInSceneEnv):
    def __init__(
        self,
        model_ids=[
            "eggplant",
            "bridge_spoon_generated_modified",
            "apple",
            "bridge_carrot_generated_modified",
        ],
        **kwargs,
    ):
        self.custom_model_ids = model_ids
        super().__init__(num_objects=len(model_ids) if model_ids else 3, **kwargs)

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        # Ensure the episode RNG is set before using it
        self.set_episode_rng(seed)

        # Use the provided model_ids or the custom_model_ids set during initialization
        model_ids = options.get("model_ids", self.custom_model_ids)
        if model_ids is None:
            raise ValueError(
                "model_ids must be provided either during initialization or in reset options"
            )

        options["model_ids"] = model_ids
        self.num_objects = len(model_ids)

        # Handle custom positions and orientations
        obj_init_options = options.get("obj_init_options", {})
        for i in range(self.num_objects):
            # Set custom position if provided
            if f"init_xy_{i}" in obj_init_options:
                options.setdefault("obj_init_options", {})[f"init_xy_{i}"] = (
                    obj_init_options[f"init_xy_{i}"]
                )

            # Set custom orientation if provided
            if f"orientation_{i}" in obj_init_options:
                options.setdefault("obj_init_options", {})[f"orientation_{i}"] = (
                    obj_init_options[f"orientation_{i}"]
                )
            elif f"init_rot_quat_{i}" in obj_init_options:
                options.setdefault("obj_init_options", {})[f"init_rot_quat_{i}"] = (
                    obj_init_options[f"init_rot_quat_{i}"]
                )

        return super().reset(seed=seed, options=options)

    def get_language_instruction(self, **kwargs):
        obj_descriptions = [
            self._get_instruction_obj_name(obj.name) for obj in self.objects
        ]
        if len(obj_descriptions) == 1:
            task_description = f"pick the {obj_descriptions[0]}"
        elif len(obj_descriptions) == 2:
            task_description = (
                f"pick the {obj_descriptions[0]} and the {obj_descriptions[1]}"
            )
        else:
            task_description = f"pick the {', '.join(obj_descriptions[:-1])}, and the {obj_descriptions[-1]}"
        return task_description


@register_env("PushMultipleDifferentObjectsInScene-v0", max_episode_steps=120)
class PushMultipleDifferentObjectsInSceneEnv(GraspMultipleCustomOrientationInSceneEnv):
    def __init__(
        self,
        model_ids=[
            "green_cube_3cm",
            "yellow_cube_3cm",
            "yellow_cube_3cm",
            "green_cube_3cm",
        ],
        **kwargs,
    ):
        self.custom_model_ids = model_ids
        super().__init__(num_objects=len(model_ids) if model_ids else 3, **kwargs)

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        # Ensure the episode RNG is set before using it
        self.set_episode_rng(seed)

        # Use the provided model_ids or the custom_model_ids set during initialization
        model_ids = options.get("model_ids", self.custom_model_ids)
        if model_ids is None:
            raise ValueError(
                "model_ids must be provided either during initialization or in reset options"
            )

        options["model_ids"] = model_ids
        self.num_objects = len(model_ids)

        # Handle custom positions and orientations
        obj_init_options = options.get("obj_init_options", {})
        for i in range(self.num_objects):
            # Set custom position if provided
            if f"init_xy_{i}" in obj_init_options:
                options.setdefault("obj_init_options", {})[f"init_xy_{i}"] = (
                    obj_init_options[f"init_xy_{i}"]
                )

            # Set custom orientation if provided
            if f"orientation_{i}" in obj_init_options:
                options.setdefault("obj_init_options", {})[f"orientation_{i}"] = (
                    obj_init_options[f"orientation_{i}"]
                )
            elif f"init_rot_quat_{i}" in obj_init_options:
                options.setdefault("obj_init_options", {})[f"init_rot_quat_{i}"] = (
                    obj_init_options[f"init_rot_quat_{i}"]
                )

        return super().reset(seed=seed, options=options)

    def get_language_instruction(self, **kwargs):
        obj_descriptions = [
            self._get_instruction_obj_name(obj.name) for obj in self.objects
        ]
        if len(obj_descriptions) == 1:
            task_description = f"pick the {obj_descriptions[0]}"
        elif len(obj_descriptions) == 2:
            task_description = (
                f"pick the {obj_descriptions[0]} and the {obj_descriptions[1]}"
            )
        else:
            task_description = f"pick the {', '.join(obj_descriptions[:-1])}, and the {obj_descriptions[-1]}"
        return task_description
