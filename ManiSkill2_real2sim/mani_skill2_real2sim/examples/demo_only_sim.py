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
env-id:
GraspSingleOpenedCokeCanInScene-v0
MoveNearGoogleBakedTexInScene-v1
OpenDrawerCustomInScene-v0
CloseDrawerCustomInScene-v0
PlaceIntoClosedDrawerCustomInScene-v0
PutCarrotOnPlateInScene-v0
PutSpoonOnTableClothInScene-v0
StackGreenCubeOnYellowCubeBakedTexInScene-v0
PutEggplantInBasketScene-v0
MoveNearGoogleInScene-v0
MoveNearGoogleBakedTexInScene-v0 (also mentioned with -v1)
StackGreenCubeOnYellowCubeInScene-v0

control mode:
arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner
arm_pd_ee_delta_pose_align2_gripper_pd_joint_pos
arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos
"""


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

    print_env_info(env)
    return env


def setup_camera_pose(args):
    if (
        "google_robot" in args.env_kwargs["robot"]
        or "widowx" in args.env_kwargs["robot"]
    ):
        pose = look_at([1.0, 1.0, 2.0], [0.0, 0.0, 0.7])
        args.env_kwargs["render_camera_cfgs"] = {
            "render_camera": dict(p=pose.p, q=pose.q)
        }


def print_env_info(env):
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Control mode:", env.control_mode)
    print("Reward mode:", env.reward_mode)


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


def setup_viewers(env, args):
    if args.enable_sapien_viewer:
        env.render_human()
    return OpenCVViewer(exit_on_esc=False)


def get_robot_info(env):
    has_base = "base" in env.agent.controller.configs
    num_arms = sum("arm" in x for x in env.agent.controller.configs)
    has_gripper = any("gripper" in x for x in env.agent.controller.configs)
    is_google_robot = "google_robot" in env.agent.robot.name
    is_widowx = "wx250s" in env.agent.robot.name
    is_gripper_delta_target_control = (
        env.agent.controller.controllers["gripper"].config.use_target
        and env.agent.controller.controllers["gripper"].config.use_delta
    )
    return (
        has_base,
        num_arms,
        has_gripper,
        is_google_robot,
        is_widowx,
        is_gripper_delta_target_control,
    )


def get_action_scale(is_google_robot, is_widowx):
    ee_action = 0.02 if (is_google_robot or is_widowx) else 0.1
    ee_rot_action = 0.1 if (is_google_robot or is_widowx) else 1.0
    return ee_action, ee_rot_action


def process_user_input(
    args,
    key,
    has_base,
    num_arms,
    has_gripper,
    is_google_robot,
    ee_action_scale,
    ee_rot_action_scale,
):
    base_action = np.zeros(4) if has_base else np.zeros(0)
    ee_action_dim = (
        6
        if "pd_ee_delta_pose" in args.control_mode
        or "pd_ee_target_delta_pose" in args.control_mode
        else 3
    )
    ee_action = np.zeros(ee_action_dim)
    gripper_action = 0

    if has_base:
        base_action = process_base_input(key, base_action)

    if num_arms > 0:
        ee_action = process_ee_input(
            key, ee_action, ee_action_scale, ee_rot_action_scale
        )

    if has_gripper:
        gripper_action = process_gripper_input(key, is_google_robot)

    return base_action, ee_action, gripper_action


def process_base_input(key, base_action):
    if key == "w":
        base_action[0] = 1
    elif key == "s":
        base_action[0] = -1
    elif key == "a":
        base_action[1] = 1
    elif key == "d":
        base_action[1] = -1
    elif key == "q" and len(base_action) > 2:
        base_action[2] = 1
    elif key == "e" and len(base_action) > 2:
        base_action[2] = -1
    elif key == "z" and len(base_action) > 2:
        base_action[3] = 1
    elif key == "x" and len(base_action) > 2:
        base_action[3] = -1
    return base_action


def process_ee_input(key, ee_action, ee_action_scale, ee_rot_action_scale):
    if key == "i":
        ee_action[0] = ee_action_scale
    elif key == "k":
        ee_action[0] = -ee_action_scale
    elif key == "j":
        ee_action[1] = ee_action_scale
    elif key == "l":
        ee_action[1] = -ee_action_scale
    elif key == "u":
        ee_action[2] = ee_action_scale
    elif key == "o":
        ee_action[2] = -ee_action_scale
    elif key == "1":
        ee_action[3:6] = [ee_rot_action_scale, 0, 0]
    elif key == "2":
        ee_action[3:6] = [-ee_rot_action_scale, 0, 0]
    elif key == "3":
        ee_action[3:6] = [0, ee_rot_action_scale, 0]
    elif key == "4":
        ee_action[3:6] = [0, -ee_rot_action_scale, 0]
    elif key == "5":
        ee_action[3:6] = [0, 0, ee_rot_action_scale]
    elif key == "6":
        ee_action[3:6] = [0, 0, -ee_rot_action_scale]
    return ee_action


def process_gripper_input(key, is_google_robot):
    if not is_google_robot:
        return 1 if key == "f" else -1 if key == "g" else 0
    else:
        return -1 if key == "f" else 1 if key == "g" else 0


def create_action_dict(base_action, ee_action, gripper_action, has_gripper):
    action_dict = {"base": base_action, "arm": ee_action}
    if has_gripper:
        action_dict["gripper"] = gripper_action
    return action_dict


def visualize_observation(obs, env):
    if "rgbd" in env.obs_mode:
        from itertools import chain
        from mani_skill2_real2sim.utils.visualization.misc import (
            observations_to_images,
            tile_images,
        )

        images = list(
            chain(*[observations_to_images(x) for x in obs["image"].values()])
        )
        render_frame = tile_images(images)
        opencv_viewer = OpenCVViewer(exit_on_esc=False)
        opencv_viewer.imshow(render_frame)
    elif "pointcloud" in env.obs_mode:
        import trimesh

        xyzw = obs["pointcloud"]["xyzw"]
        mask = xyzw[..., 3] > 0
        rgb = obs["pointcloud"]["rgb"]
        if "robot_seg" in obs["pointcloud"]:
            robot_seg = obs["pointcloud"]["robot_seg"]
            rgb = np.uint8(robot_seg * [11, 61, 127])
        trimesh.PointCloud(xyzw[mask, :3], rgb[mask]).show()


def main():
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()
    env = setup_environment(args)
    reset_options = get_env_reset_options(env, args)
    opencv_viewer = setup_viewers(env, args)

    (
        has_base,
        num_arms,
        has_gripper,
        is_google_robot,
        is_widowx,
        is_gripper_delta_target_control,
    ) = get_robot_info(env)
    ee_action_scale, ee_rot_action_scale = get_action_scale(is_google_robot, is_widowx)

    obs, info = env.reset(options=reset_options)
    print("Reset info:", info)
    print("Instruction:", env.unwrapped.get_language_instruction())
    print("Robot pose:", env.agent.robot.pose)
    print("Initial qpos:", env.agent.robot.get_qpos())

    def render_wait():
        if not args.enable_sapien_viewer:
            return
        while True:
            env.render_human()
            sapien_viewer = env.viewer
            if sapien_viewer.window.key_down("0"):
                break

    idx = 0
    while True:
        render_frame = env.render()
        key = opencv_viewer.imshow(render_frame)

        if key == "0":  # switch to SAPIEN viewer
            render_wait()
        elif key == "r":  # reset env
            obs, info = env.reset(options=reset_options)
            print("Reset info:", info)
            print("Instruction:", env.unwrapped.get_language_instruction())
            continue
        elif key is None:  # exit
            break

        base_action, ee_action, gripper_action = process_user_input(
            args,
            key,
            has_base,
            num_arms,
            has_gripper,
            is_google_robot,
            ee_action_scale,
            ee_rot_action_scale,
        )

        if key == "v":
            visualize_observation(obs, env)

        action_dict = create_action_dict(
            base_action, ee_action, gripper_action, has_gripper
        )
        action = env.agent.controller.from_action_dict(action_dict)

        print("Action:", action)
        obs, reward, terminated, truncated, info = env.step(action)

        print("TCP pose wrt robot base:", env.agent.robot.pose.inv() * env.tcp.pose)
        print("QPos:", env.agent.robot.get_qpos())
        print("Reward:", reward)
        print("Terminated:", terminated, "Truncated:", truncated)
        print("Info:", info)

        if is_gripper_delta_target_control:
            gripper_action = 0

        idx += 1

    env.close()


if __name__ == "__main__":
    main()
