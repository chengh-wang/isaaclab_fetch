"""Keypoint-based Reach Env Config for mobile manipulators.

Self-contained — does NOT inherit from any other env config.

Key differences from the original reach_env_simple_cfg:
  - Observation: 9-dim keypoint delta replaces 7-dim pos+quat delta
  - Reward: 3 keypoint tracking terms replace 6 separate pos/ori terms
  - Progress reward for dense shaping signal
  - Orientation command range tightened to physically reachable space
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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import fetch_project.tasks.manipulation.reach.mdp as mdp


# =============================================================================
# Constants
# =============================================================================

CUBE_SIDE = 0.3  # Virtual cube side length — controls pos/ori trade-off

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

# Shorthand for keypoint reward params
_KP = {
    "command_name": "ee_pose",
    "asset_cfg": SceneEntityCfg("robot", body_names=["wrist_roll_link"]),
    "cube_side": CUBE_SIDE,
}


# =============================================================================
# Scene
# =============================================================================

@configclass
class KeypointReachSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    robot: ArticulationCfg = MISSING  # Set in Fetch-specific config

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,
    )


# =============================================================================
# Commands — world-frame pose targets (unchanged)
# =============================================================================

@configclass
class KeypointCommandsCfg:
    ee_pose = mdp.WorldPoseCommandCfg(
        asset_name="robot",
        body_name="wrist_roll_link",
        ranges=mdp.WorldPoseCommandCfg.Ranges(
            pos_x=(-0.5, 1.0),
            pos_y=(-.5, .5),
            pos_z=(0.4, 1.1),
            # Tightened to Fetch-reachable orientation ranges
            roll=(-0.5, 0.5),  
            pitch=(-1.0, 0.3),
            yaw=(-3.14, 3.14),
        ),
        success_threshold=0.001,
        ori_threshold=0.01,
        settling_speed_threshold=0.08,
        hold_time_range=(1.0, 3.0),
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
    )


# =============================================================================
# Actions — arm position + diff-drive base
# =============================================================================

@configclass
class KeypointActionsCfg:
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
        wheel_radius=0.0625,
        wheel_separation=0.372,
        max_linear_velocity=0.3,
        max_angular_velocity=0.5,
    )


# =============================================================================
# Observations — keypoint delta in body frame (9-dim)
# =============================================================================

@configclass
class KeypointObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=FETCH_ARM_JOINTS)},
        )
        # === KEYPOINT OBS: 9-dim replaces 7-dim pos+quat delta ===
        kp_command = ObsTerm(
            func=mdp.keypoint_command_in_body_frame,
            params={
                "command_name": "ee_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names=["wrist_roll_link"]),
                "cube_side": CUBE_SIDE,
            },
        )
        # === BASE STATE: critical for closed-loop base control ===
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# Events
# =============================================================================

@configclass
class KeypointEventCfg:

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

    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.1),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    # Reset keypoint progress tracker
    reset_kp_progress = EventTerm(
        func=mdp.keypoint_progress_reset,
        mode="reset",
    )


# =============================================================================
# Rewards — keypoint tracking + regularization
# =============================================================================

@configclass
class KeypointRewardsCfg:

    # --- Keypoint tracking (replaces all pos_*/ori_* terms) ---

    kp_exp = RewTerm(
        func=mdp.keypoint_tracking_exp,
        weight=3.0,
        params={**_KP, "sigma": 0.15},
    )

    kp_tanh = RewTerm(
        func=mdp.keypoint_tracking_tanh,
        weight=10,
        params={**_KP, "sigma": 0.05},
    )

    kp_l2 = RewTerm(
        func=mdp.keypoint_tracking_l2,
        weight=-1.0,
        params=_KP,
    )

    kp_progress = RewTerm(
        func=mdp.keypoint_progress,
        weight=5.0,
        params=_KP,
    )

    # --- Base: approach + face target (kept from original) ---

    base_move = RewTerm(
        func=mdp.base_approach_facing,
        weight=1.0,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot"),
            "approach_threshold": 0.5,
            "approach_sigma": 0.3,
        },
    )

    # --- Regularization (reduced arm_weight for more exploration) ---

    action_rate = RewTerm(
        func=mdp.action_rate_weighted,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "arm_joint_names": FETCH_ARM_JOINTS,
            "base_joint_names": ["l_wheel_joint", "r_wheel_joint"],
            "arm_weight": 1e-1,   # Reduced from 5e-1 to allow arm exploration
            "base_weight": 1e-1,
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
            "base_weight": 1e-2,
        },
    )

    base_vel = RewTerm(
        func=mdp.base_velocity_penalty,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    ".*(?<!l_wheel_link)(?<!r_wheel_link)"
                    "(?<!l_gripper_finger_link)(?<!r_gripper_finger_link)"
                    "(?<!base_link)"
                    "(?<!bl_caster_link)(?<!br_caster_link)"
                    "(?<!fl_caster_link)(?<!fr_caster_link)"
                    "(?<!caster_link)(?<!caster_bottom_link)$"
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

    # metrics = RewTerm(
    #     func=mdp.keypoint_metrics,
    #     weight=0.0,
    #     params=_KP,
    # )

# =============================================================================
# Terminations
# =============================================================================

@configclass
class KeypointTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    robot_flipped = DoneTerm(
        func=mdp.robot_flipped,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_tilt_angle": 0.5},
    )


# =============================================================================
# Curriculum
# =============================================================================

@configclass
class KeypointCurriculumCfg:
    kp_sigma = CurrTerm(
        func=mdp.keypoint_sigma_curriculum,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=["wrist_roll_link"]),
            "cube_side": CUBE_SIDE,
        },
    )


# =============================================================================
# Top-level env config
# =============================================================================

@configclass
class ReachEnvKeypointCfg(ManagerBasedRLEnvCfg):
    """Keypoint-based reach env for mobile manipulators."""

    scene: KeypointReachSceneCfg = KeypointReachSceneCfg(
        num_envs=4096, env_spacing=2.5,
    )
    commands: KeypointCommandsCfg = KeypointCommandsCfg()
    actions: KeypointActionsCfg = KeypointActionsCfg()
    observations: KeypointObservationsCfg = KeypointObservationsCfg()
    events: KeypointEventCfg = KeypointEventCfg()
    rewards: KeypointRewardsCfg = KeypointRewardsCfg()
    terminations: KeypointTerminationsCfg = KeypointTerminationsCfg()
    # curriculum: KeypointCurriculumCfg = KeypointCurriculumCfg()
    curriculum: KeypointCurriculumCfg = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.5, 3.5, 3.5)
