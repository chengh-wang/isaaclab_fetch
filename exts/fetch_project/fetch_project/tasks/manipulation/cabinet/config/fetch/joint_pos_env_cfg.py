# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

# from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

# from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (  # isort: skip
#     FRAME_MARKER_SMALL_CFG,
#     CabinetEnvCfg,
# )

import fetch_project.tasks.manipulation.cabinet.mdp as mdp
from fetch_project.tasks.manipulation.cabinet.cabinet_env_cfg import (CabinetEnvCfg, FRAME_MARKER_SMALL_CFG)

# ##
# # Pre-defined configs
# ##
from fetch_project.robots.fetch import FETCH_CFG  # isort: skip


@configclass
class FetchCabinetEnvCfg(CabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set franka as robot
        self.scene.robot = FETCH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set Actions for the specific robot type (franka)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["torso_lift_joint", "shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"],
            # joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["r_gripper_finger_joint", "l_gripper_finger_joint"],
            open_command_expr={"r_gripper_finger_joint": 0.04, "l_gripper_finger_joint": 0.04},
            close_command_expr={"r_gripper_finger_joint": 0.0, "l_gripper_finger_joint": 0.0},
        )

        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/wrist_roll_link",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.183, 0.0, 0.0),
                        # rot=(0.707, 0.0, 0.0, -0.707),
                        rot=(0.0, 0.707, 0.707, 0.0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/l_gripper_finger_link",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.015, 0.01, 0.0),
                        # rot=(0.707, 0.0, 0.0, -0.707),
                        rot=(0.0, 0.707, 0.707, 0.0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/r_gripper_finger_link",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.015, -0.01, 0.0),
                        # rot=(0.707, 0.0, 0.0, -0.707),
                        rot=(0.0, 0.707, 0.707, 0.0),
                    ),
                ),
            ],
        )

        # override rewards
        self.rewards.approach_gripper_handle.params["offset"] = 0.04
        self.rewards.grasp_handle.params["open_joint_pos"] = 0.04
        self.rewards.grasp_handle.params["asset_cfg"].joint_names = ["r_gripper_finger_joint", "l_gripper_finger_joint"]


@configclass
class FetchCabinetEnvCfg_PLAY(FetchCabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
