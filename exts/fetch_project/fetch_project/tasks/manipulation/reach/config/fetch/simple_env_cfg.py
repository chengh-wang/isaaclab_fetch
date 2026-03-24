"""Fetch simple reach environment config."""

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import fetch_project.tasks.manipulation.reach.mdp as mdp
from fetch_project.tasks.manipulation.reach.reach_env_simple_cfg import ReachEnvSimpleCfg
from fetch_project.robots.fetch import FETCH_CFG, FETCH_FOLD_CFG, FETCH_CFG_PACE


FETCH_ARM_JOINTS = [
    "torso_lift_joint",
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]


@configclass
class FetchReachSimpleEnvCfg(ReachEnvSimpleCfg):

    def __post_init__(self):
        super().__post_init__()

        # Robot
        self.scene.robot = FETCH_CFG_PACE.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.events.reset_robot_joints.params["position_range"] = (-0.5, 0.5)

        # Arm action
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=FETCH_ARM_JOINTS,
            scale=0.5,
            use_default_offset=True,
        )

        # Command
        self.commands.ee_pose.body_name = "wrist_roll_link"
        # self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # Set arm joint names for weighted regularization rewards
        self.rewards.action_rate.params["arm_joint_names"] = FETCH_ARM_JOINTS
        self.rewards.joint_vel.params["arm_joint_names"] = FETCH_ARM_JOINTS

        # Contact sensor for self-collision detection
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=False,
            force_threshold=1.0,
        )


@configclass
class FetchReachSimpleEnvCfg_PLAY(FetchReachSimpleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 50
        self.observations.policy.enable_corruption = False