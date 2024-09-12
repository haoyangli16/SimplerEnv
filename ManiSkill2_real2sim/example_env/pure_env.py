import argparse
import numpy as np
import gymnasium as gym
from transforms3d.euler import euler2quat
from mani_skill2_real2sim.envs.sapien_env import BaseEnv
from mani_skill2_real2sim.utils.visualization.cv2_utils import OpenCVViewer
from sapien.core import Pose
from mani_skill2_real2sim.utils.sapien_utils import look_at, normalize_vector

MS1_ENV_IDS = [
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "PushChair-v1",
    "MoveBucket-v1",
]
"""
python example_env/manual_control.py -e GraspMultipleDifferentObjectsInScene-v0 \
    -c arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner \
    -o rgbd \
    --enable-sapien-viewer \
    robot google_robot_static

python example_env/manual_control.py -e GraspMultipleSameObjectsInScene-v0 \
    -c arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner \
    -o rgbd \
    --enable-sapien-viewer \
    robot google_robot_static


python example_env/manual_control.py -e PushMultipleDifferentObjectsInScene-v0 \
-c arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos -o rgbd robot widowx sim_freq @500 control_freq @5 \
scene_name bridge_table_1_v1  rgb_overlay_mode debug rgb_overlay_path data/real_inpainting/bridge_real_eval_1.png rgb_overlay_cameras 3rd_view_camera

"""

import numpy as np
from enum import Enum
from scipy.spatial.transform import Rotation
from sapien.core import Pose
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Robot Control Visualization Script")
    parser.add_argument(
        "-e", "--env-id", type=str, required=True, help="Environment ID"
    )
    parser.add_argument("-o", "--obs-mode", type=str, help="Observation mode")
    parser.add_argument("--reward-mode", type=str, help="Reward mode")
    parser.add_argument(
        "-c",
        "--control-mode",
        type=str,
        default="pd_ee_delta_pose",
        help="Control mode",
    )
    parser.add_argument(
        "--render-mode", type=str, default="cameras", help="Render mode"
    )
    parser.add_argument(
        "--add-segmentation",
        action="store_true",
        help="Add segmentation to observation",
    )
    parser.add_argument(
        "--enable-sapien-viewer", action="store_true", help="Enable SAPIEN viewer"
    )

    args, opts = parser.parse_known_args()
    args.env_kwargs = parse_env_kwargs(opts)
    return args


def parse_env_kwargs(opts):
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    return dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))


def get_env_reset_options(env, args):
    if hasattr(env, "prepackaged_config") and env.prepackaged_config:
        return {}

    reset_options = {}
    if "GraspSingle" in args.env_id:
        reset_options = setup_grasp_single_options()
    elif "MoveNear" in args.env_id:
        reset_options = setup_move_near_options()
    elif "Drawer" in args.env_id:
        reset_options = setup_drawer_options(args)
    elif "GraspMultipleSameObjectsInScene" in args.env_id:
        reset_options = setup_grasp_multi_options()
    if args.env_id == "GraspMultipleDifferentObjectsInScene-v0":
        reset_options = setup_grasp_multi_different_options()
    elif "PushMultipleDifferentObjectsInScene" in args.env_id:
        reset_options = setup_push_multi_different_options()
    elif any(
        task in args.env_id
        for task in [
            "PutSpoonOnTableCloth",
            "PutCarrotOnPlate",
            "StackGreenCubeOnYellowCube",
            "PutEggplantInBasket",
        ]
    ):
        reset_options = setup_task_options(env)

    return reset_options


def setup_grasp_multi_options():
    custom_options = {
        "obj_init_options": {
            "init_xy_0": [-0.2, 0.2],  # Custom position for the first can
            "orientation_0": "upright",  # Custom orientation for the first can
            "init_xy_1": [-0.35, 0.2],  # Custom position for the second can
            "orientation_1": "upright",  # Custom orientation for the first can
            "init_xy_2": [-0.2, 0.35],  # Custom position for the third can
            "orientation_2": "upright",  # Custom orientation for the third can
            "init_xy_3": [-0.35, 0.35],  # Custom position for the fourth can
            "orientation_3": "upright",  # Custom orientation for the fourth can
            "init_xy_4": [-0.2, 0.09],  # Custom position for the fifth can
            "orientation_4": "upright",  # Custom orientation for the fifth can
            "init_xy_5": [-0.35, 0.05],  # Custom position for the sixth can
            "orientation_5": "laid_vertically",  # Custom orientation for the sixth can
            "init_xy_6": [-0.47, 0.15],  # Custom position for the seventh can
            "orientation_6": "upright",  # Custom orientation for the seventh can
            "init_xy_7": [-0.5, 0.3],  # Custom position for the eighth can
            "init_rot_quat_7": euler2quat(
                np.pi / 4, np.pi / 4, 0
            ),  # Custom orientation for the second can (45 degrees around y-axis)
            "init_xy_8": [-0.2, -0.15],  # Custom position for the ninth can
            "orientation_8": "upright",  # Custom orientation for the ninth can
            "init_xy_9": [-0.35, 0.65],  # Custom position for the tenth can
            "orientation_9": "laid_vertically",  # Custom orientation for the tenth can
        }
    }
    return custom_options


def setup_grasp_multi_different_options():
    custom_options = {
        "model_ids": [
            "bridge_spoon_generated_modified",
            "blue_plastic_bottle",
            "apple",
            "eggplant",
        ],
        "obj_init_options": {
            "init_xy_0": [-0.2, 0.2],
            "orientation_0": "upright",
            "init_xy_1": [-0.35, 0.02],
            "orientation_1": "laid_vertically",
            "init_xy_2": [-0.2, 0.35],
            # "init_rot_quat_2": euler2quat(np.pi / 2, 0, 0),
            "init_xy_3": [-0.35, 0.23],
            "init_rot_quat_3": euler2quat(np.pi / 4, 0, 0),
            "init_xy_4": [-0.2, 0.09],
            "orientation_4": "upright",
            "init_xy_5": [-0.35, 0.05],
            "orientation_5": "laid_vertically",
            "init_xy_6": [-0.47, 0.15],
            "orientation_6": "upright",
            "init_xy_7": [-0.5, 0.3],
            "init_rot_quat_7": euler2quat(np.pi / 4, np.pi / 4, 0),
            "init_xy_8": [-0.2, -0.15],
            "orientation_8": "upright",
            "init_xy_9": [-0.35, 0.65],
            "orientation_9": "laid_vertically",
        },
    }
    return custom_options


def setup_push_multi_different_options():
    custom_options = {
        "model_ids": [
            "bridge_spoon_generated_modified",
            "blue_plastic_bottle",
            "apple",
            "eggplant",
        ],
        "obj_init_options": {
            "init_xy_0": [-0.096, 0.011],
            "orientation_0": "upright",
            "init_xy_1": [-0.141, 0.193],
            "orientation_1": "laid_vertically",
            "init_xy_2": [-0.208, 0.084],
            # "init_rot_quat_2": euler2quat(np.pi / 2, 0, 0),
            "init_xy_3": [-0.2, 0.23],
            "init_xy_4": [-0.2, 0.09],
            "orientation_4": "upright",
            "init_xy_5": [-0.35, 0.05],
            "orientation_5": "laid_vertically",
            "init_xy_6": [-0.47, 0.15],
            "orientation_6": "upright",
            "init_xy_7": [-0.5, 0.3],
            "init_rot_quat_7": euler2quat(np.pi / 4, np.pi / 4, 0),
            "init_xy_8": [-0.2, -0.15],
            "orientation_8": "upright",
            "init_xy_9": [-0.35, 0.65],
            "orientation_9": "laid_vertically",
        },
    }
    return custom_options


def setup_grasp_single_options():
    init_rot_quat = Pose(q=[0, 0, 0, 1]).q
    return {
        "obj_init_options": {"init_xy": [-0.12, 0.2]},
        "robot_init_options": {
            "init_xy": [0.35, 0.20],
            "init_rot_quat": init_rot_quat,
        },
    }


def setup_move_near_options():
    init_rot_quat = (Pose(q=euler2quat(0, 0, -0.09)) * Pose(q=[0, 0, 0, 1])).q
    return {
        "obj_init_options": {"episode_id": 0},
        "robot_init_options": {
            "init_xy": [0.35, 0.21],
            "init_rot_quat": init_rot_quat,
        },
    }


def setup_drawer_options(args):
    init_rot_quat = [0, 0, 0, 1]
    init_xy = [0.652, 0.009] if "PlaceInClosedDrawer" in args.env_id else [0.851, 0.035]
    return {
        "obj_init_options": {"init_xy": [0.0, 0.0]},
        "robot_init_options": {
            "init_xy": init_xy,
            "init_rot_quat": init_rot_quat,
        },
    }


def setup_environment(args):
    if (
        args.env_id in MS1_ENV_IDS
        and args.control_mode is not None
        and not args.control_mode.startswith("base")
    ):
        args.control_mode = "base_pd_joint_vel_arm_" + args.control_mode

    if "robot" in args.env_kwargs:
        setup_camera_pose(args)

    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        camera_cfgs={"add_segmentation": args.add_segmentation},
        **args.env_kwargs,
    )

    return env


def setup_task_options(env):
    init_rot_quat = Pose(q=[0, 0, 0, 1]).q
    if env.robot_uid == "widowx":
        init_xy = [0.147, 0.028]
    elif env.robot_uid == "widowx_camera_setup2":
        init_xy = [0.147, 0.070]
    elif env.robot_uid == "widowx_sink_camera_setup":
        init_xy = [0.127, 0.060]
    else:
        init_xy = [0.147, 0.028]

    return {
        "obj_init_options": {"episode_id": 0},
        "robot_init_options": {
            "init_xy": init_xy,
            "init_rot_quat": init_rot_quat,
        },
    }


def setup_camera_pose(args):
    if (
        "google_robot" in args.env_kwargs["robot"]
        or "widowx" in args.env_kwargs["robot"]
    ):
        pose = look_at([1.0, 1.0, 2.0], [0.0, 0.0, 0.7])
        args.env_kwargs["render_camera_cfgs"] = {
            "render_camera": dict(p=pose.p, q=pose.q)
        }


def get_robot_info(env):
    has_base = "base" in env.agent.controller.configs
    has_gripper = any("gripper" in x for x in env.agent.controller.configs)
    is_google_robot = "google_robot" in env.agent.robot.name
    is_widowx = "wx250s" in env.agent.robot.name
    return has_base, has_gripper, is_google_robot, is_widowx


def create_action_dict(base_action, ee_action, gripper_action, has_gripper):
    action_dict = {"base": base_action, "arm": ee_action}
    if has_gripper:
        action_dict["gripper"] = gripper_action
    return action_dict


def main():
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()
    env = setup_environment(args)
    reset_options = get_env_reset_options(env, args)

    obs, info = env.reset(options=reset_options)
    print("Reset info:", info)
    print("Instruction:", env.unwrapped.get_language_instruction())
    print("Robot pose:", env.agent.robot.pose)
    print("Initial qpos:", env.agent.robot.get_qpos())

    while True:
        render_frame = env.render()
        env.render_human()

        if "GraspSingle" in args.env_id:
            print("Object pose:", env.obj.get_pose())
        elif "GraspMultipleSameObjectsInScene" in args.env_id:
            for obj in env.objects[:1]:
                print(f"Object {obj.name} pose: {obj.get_pose()}")

        print("TCP pose wrt world:", env.tcp.pose)

        # Determine object position based on environment type
        if "GraspSingle" in args.env_id:
            object_position = env.obj.get_pose().p
        elif "GraspMultipleSameObjectsInScene" in args.env_id:
            object_position = env.objects[0].get_pose().p
        elif "GraspMultipleDifferentObjectsInScene" in args.env_id:
            object_position = env.objects[0].get_pose().p
        elif "PushMultipleDifferentObjectsInScene" in args.env_id:
            object_position = env.objects[0].get_pose().p
        else:
            object_position = env.episode_objs[0].get_pose().p

        print("Object position:", object_position)

        # Simulate the environment without actions
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

        time.sleep(1.0 / 20)  # Limit to 20 FPS

        if terminated or truncated:
            obs, info = env.reset(options=reset_options)
            print("Environment reset")


if __name__ == "__main__":
    main()
