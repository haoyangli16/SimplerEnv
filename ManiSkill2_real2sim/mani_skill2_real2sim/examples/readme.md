```bash
# grasp multiple same objects (Coke cans)
python mani_skill2_real2sim/examples/demo_grasp_motion_plan.py -e GraspMultipleSameObjectsInScene-v0 \
 -c arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner -o rgbd \
 --enable-sapien-viewer robot google_robot_static

# put carrot on plate
python mani_skill2_real2sim/examples/demo_grasp_motion_plan.py -e PutCarrotOnPlateInScene-v0 --enable-sapien-viewer \
 -c arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos -o rgbd robot widowx sim_freq @500 control_freq @5 \
 scene_name bridge_table_1_v1 rgb_overlay_mode debug rgb_overlay_path data/real_inpainting/bridge_real_eval_1.png rgb_overlay_cameras 3rd_view_camera
```
