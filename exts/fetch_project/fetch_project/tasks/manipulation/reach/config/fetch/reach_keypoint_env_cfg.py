"""Fetch-specific keypoint reach environment config."""

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import fetch_project.tasks.manipulation.reach.mdp as mdp
from fetch_project.robots.fetch import FETCH_CFG

from fetch_project.tasks.manipulation.reach.reach_env_keypoint_cfg import ReachEnvKeypointCfg


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
class FetchReachKeypointEnvCfg(ReachEnvKeypointCfg):

    def __post_init__(self):
        super().__post_init__()

        # Robot
        self.scene.robot = FETCH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Wider joint reset range for exploration
        self.events.reset_robot_joints.params["position_range"] = (-0.5, 0.5)

        # Arm action
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=FETCH_ARM_JOINTS,
            scale=0.5,
            use_default_offset=True,
        )

        # Command body
        self.commands.ee_pose.body_name = "wrist_roll_link"

        # Arm joint names for weighted regularization
        self.rewards.action_rate.params["arm_joint_names"] = FETCH_ARM_JOINTS
        self.rewards.joint_vel.params["arm_joint_names"] = FETCH_ARM_JOINTS

        # Contact sensor
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=False,
            force_threshold=1.0,
        )


@configclass
class FetchReachKeypointEnvCfg_PLAY(FetchReachKeypointEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 50
        self.observations.policy.enable_corruption = False
