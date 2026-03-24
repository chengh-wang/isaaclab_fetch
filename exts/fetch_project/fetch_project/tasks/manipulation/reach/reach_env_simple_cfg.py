"""Standalone Simple Reach Env Config for Fetch Mobile Manipulator.

Self-contained — does NOT inherit from ReachEnvCfg.

Reward structure:
- EE position tracking: exp + tanh + L2
- EE orientation tracking: exp + L2
- Base approach + facing target
- Regularization: action rate, joint vel, contacts, flat orientation
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import fetch_project.tasks.manipulation.reach.mdp as mdp
from fetch_project.robots.fetch import FETCH_WHEEL_RADIUS_EFF, FETCH_WHEEL_SEPARATION_EFF


# ============================================================================
# Shared shorthand
# ============================================================================

_EE = {
    "command_name": "ee_pose",
    "asset_cfg": SceneEntityCfg("robot", body_names=["wrist_roll_link"]),
}


# ============================================================================
# Scene
# ============================================================================

@configclass
class SimpleReachSceneCfg(InteractiveSceneCfg):
    """Ground + robot + lights + contact sensor."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    robot: ArticulationCfg = MISSING  # Set in derived (Fetch-specific) config

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,
        debug_vis=True,
    )

    # contact_sensor = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*",
    #     update_period=0.0,
    #     history_length=1,
    #     debug_vis=False,
    # )


# ============================================================================
# Commands — world-frame pose targets
# ============================================================================

@configclass
class SimpleCommandsCfg:
    ee_pose = mdp.WorldPoseCommandCfg(
        asset_name="robot",
        body_name="wrist_roll_link",
        ranges=mdp.WorldPoseCommandCfg.Ranges(
            pos_x=(-0.5, 0.7),
            pos_y=(-0.5, 0.5),
            pos_z=(0.4, 1.1),
            roll=(-0.3, 0.3),
            pitch=(1.5707963267948966, 1.5707963267948966),
            yaw=(-1.57, 1.57),
        ),
        success_threshold=0.04,
        ori_threshold=0.25,
        settling_speed_threshold=0.08,
        hold_time_range=(1.0, 3.0),
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
    )



# ============================================================================
# Actions — arm position + diff-drive base
# ============================================================================

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
class SimpleActionsCfg:
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=FETCH_ARM_JOINTS,
        scale=0.5,
        use_default_offset=True,
    )

    base_action: ActionTerm = mdp.DifferentialDriveActionCfg(
        asset_name="robot",
        left_wheel_joint_name="l_wheel_joint",
        right_wheel_joint_name="r_wheel_joint",
        wheel_radius=FETCH_WHEEL_RADIUS_EFF,
        wheel_separation=FETCH_WHEEL_SEPARATION_EFF,
        max_linear_velocity=0.3,
        max_angular_velocity=0.5,
    )


# ============================================================================
# Observations — body-frame command for base invariance
# ============================================================================

@configclass
class SimpleObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command = ObsTerm(
            func=mdp.pose_command_in_body_frame,
            params={"command_name": "ee_pose", "asset_cfg": SceneEntityCfg("robot")},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ============================================================================
# Events
# ============================================================================

@configclass
class SimpleEventCfg:

    set_caster_friction = EventTerm(
        func=mdp.set_caster_friction,
        mode="startup",
        params={
            "static_friction": 0.0,
            "dynamic_friction": 0.0,
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1), 
            "velocity_range": (0.0, 0.0),
        },
    )
    
    # Reset robot base to origin on reset
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.,0.1), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )



# ============================================================================
# Rewards
# ============================================================================

@configclass
class SimpleRewardsCfg:

    # --- EE position tracking ---
    pos_exp = RewTerm(
        func=mdp.position_error_exp,
        weight=1.0,
        params={**_EE, "sigma": 0.15},
    )
    pos_fine = RewTerm(
        func=mdp.position_error_tanh,
        weight=2,
        params={**_EE, "sigma": 0.05},
    )
    pos_l2 = RewTerm(
        func=mdp.position_error_l2,
        weight=-1.0,
        params=_EE,
    )

    # --- EE orientation tracking ---
    ori_exp = RewTerm(
        func=mdp.orientation_error_exp,
        weight=1.0,
        params={**_EE, "sigma": 0.2},
    )
    ori_fine = RewTerm(
        func=mdp.orientation_error_exp,
        weight=2.0,
        params={**_EE, "sigma": 0.05},
    )
    ori_l2 = RewTerm(
        func=mdp.orientation_error_l2,
        weight=-0.5,
        params=_EE,
    )

    # --- Base: approach + face target ---
    base_move = RewTerm(
        func=mdp.base_approach_facing,
        weight=1.5,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot"),
            "approach_threshold": 0.5,
            "approach_sigma": 0.3,
        },
    )

    # --- Regularization ---
    action_rate = RewTerm(
        func=mdp.action_rate_weighted,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "arm_joint_names": FETCH_ARM_JOINTS,
            "base_joint_names": ["l_wheel_joint", "r_wheel_joint"],
            "arm_weight": 5e-1,
            "base_weight": 1e-3,
        },
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_weighted,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "arm_joint_names": FETCH_ARM_JOINTS,
            "base_joint_names": ["l_wheel_joint", "r_wheel_joint"],
            "arm_weight": 1e-2,
            "base_weight": 1e-7,
        },
    )
    base_vel = RewTerm(
        func=mdp.base_velocity_penalty,
        weight=-1e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    ".*(?<!l_wheel_link)(?<!r_wheel_link)"
                    "(?<!l_gripper_finger_link)(?<!r_gripper_finger_link)$"
                ],
            ),
            "threshold": 1.0,
        },
    )
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ============================================================================
# Terminations
# ============================================================================

@configclass
class SimpleTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    robot_flipped = DoneTerm(
        func=mdp.robot_flipped,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_tilt_angle": 0.5},
    )



# ============================================================================
# Top-level environment config
# ============================================================================

@configclass
class ReachEnvSimpleCfg(ManagerBasedRLEnvCfg):
    """Standalone simple reach env for Fetch mobile manipulator."""

    # Scene
    scene: SimpleReachSceneCfg = SimpleReachSceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP components
    commands: SimpleCommandsCfg = SimpleCommandsCfg()
    actions: SimpleActionsCfg = SimpleActionsCfg()
    observations: SimpleObservationsCfg = SimpleObservationsCfg()
    events: SimpleEventCfg = SimpleEventCfg()
    rewards: SimpleRewardsCfg = SimpleRewardsCfg()
    terminations: SimpleTerminationsCfg = SimpleTerminationsCfg()
    curriculum = None

    def __post_init__(self):
        # General
        self.decimation = 4
        self.episode_length_s = 12.0
        # Sim
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # Viewer
        self.viewer.eye = (3.5, 3.5, 3.5)