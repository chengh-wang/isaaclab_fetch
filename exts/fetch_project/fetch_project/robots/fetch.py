# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Fetch robots."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from pace_sim2real.utils import PaceDCMotorCfg
import torch
# -----------------------------------------------------------------------------
# Original USD-based Fetch config
# -----------------------------------------------------------------------------

FETCH_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/chengh-wang/Documents/git/isaaclab_fetch/assets/robots/fetch_description/fetch_isaaclab/fetch_isaaclab.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "r_wheel_joint": 0.0,
            "l_wheel_joint": 0.0,
            "head_pan_joint": 0.0,
            "head_tilt_joint": 0.0,
            "torso_lift_joint": 0.0,
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": 0.0,
            "upperarm_roll_joint": 0.0,
            "elbow_flex_joint": 0.0,
            "forearm_roll_joint": 0.0,
            "wrist_flex_joint": 0.0,
            "wrist_roll_joint": 0.0,
            "r_gripper_finger_joint": 0.01,
            "l_gripper_finger_joint": 0.01,
        },
    ),
    actuators={
        "fetch_wheels": ImplicitActuatorCfg(
            joint_names_expr=["l_wheel_joint", "r_wheel_joint"],
            effort_limit=18.5,
            velocity_limit=16.0,
            stiffness=0.0,
            damping=20.0,
        ),
        "fetch_torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_lift_joint"],
            effort_limit=450.0,
            velocity_limit=0.5,
            stiffness=400.0,
            damping=80.0,
        ),
        "fetch_shoulder_pan": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint"],
            effort_limit=33.82,
            velocity_limit=1.256,
            stiffness=400.0,
            damping=80.0,
        ),
        "fetch_shoulder_lift": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_lift_joint"],
            effort_limit=131.76,
            velocity_limit=1.454,
            stiffness=400.0,
            damping=80.0,
        ),
        "fetch_upperarm_roll": ImplicitActuatorCfg(
            joint_names_expr=["upperarm_roll_joint"],
            effort_limit=76.94,
            velocity_limit=1.571,
            stiffness=400.0,
            damping=80.0,
        ),
        "fetch_elbow_flex": ImplicitActuatorCfg(
            joint_names_expr=["elbow_flex_joint"],
            effort_limit=66.18,
            velocity_limit=1.521,
            stiffness=400.0,
            damping=80.0,
        ),
        "fetch_forearm_roll": ImplicitActuatorCfg(
            joint_names_expr=["forearm_roll_joint"],
            effort_limit=29.35,
            velocity_limit=1.571,
            stiffness=400.0,
            damping=80.0,
        ),
        "fetch_wrist_flex": ImplicitActuatorCfg(
            joint_names_expr=["wrist_flex_joint"],
            effort_limit=25.70,
            velocity_limit=2.268,
            stiffness=400.0,
            damping=80.0,
        ),
        "fetch_wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["wrist_roll_joint"],
            effort_limit=7.36,
            velocity_limit=2.268,
            stiffness=400.0,
            damping=80.0,
        ),
        "fetch_gripper": ImplicitActuatorCfg(
            joint_names_expr=["r_gripper_finger_joint", "l_gripper_finger_joint"],
            effort_limit=60.0,
            velocity_limit=0.05,
            stiffness=1000.0,
            damping=100.0,
        ),
        "fetch_head": ImplicitActuatorCfg(
            joint_names_expr=["head_pan_joint", "head_tilt_joint"],
            effort_limit=1.0,
            velocity_limit=1.57,
            stiffness=1000.0,
            damping=100.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FETCH_FOLD_CFG = FETCH_CFG.copy()
FETCH_FOLD_CFG.init_state = ArticulationCfg.InitialStateCfg(
    joint_pos={
        "r_wheel_joint": 0.0,
        "l_wheel_joint": 0.0,
        "head_pan_joint": 0.0,
        "head_tilt_joint": 0.0,
        "torso_lift_joint": 0.0,
        "shoulder_pan_joint": -1.28,
        "shoulder_lift_joint": 1.51,
        "upperarm_roll_joint": 0.35,
        "elbow_flex_joint": 1.81,
        "forearm_roll_joint": 0.0,
        "wrist_flex_joint": 1.47,
        "wrist_roll_joint": 0.0,
        "r_gripper_finger_joint": 0.01,
        "l_gripper_finger_joint": 0.01,
    },
)


FETCH_WHEEL_ARMATURE = {
    "l_wheel_joint": 1.9985375,
    "r_wheel_joint": 1.4232441,
}

FETCH_WHEEL_VISCOUS_FRICTION = {
    "l_wheel_joint": 3.1895635,
    "r_wheel_joint": 3.1029959,
}

FETCH_WHEEL_FRICTION = {
    "l_wheel_joint": 0.0,
    "r_wheel_joint": 0.0,
}

FETCH_WHEEL_ENCODER_BIAS = [0.0, 0.0]
FETCH_WHEEL_MAX_DELAY = 10

FETCH_WHEEL_RADIUS_EFF = 0.05502
FETCH_WHEEL_SEPARATION_EFF = 0.4197
FETCH_WHEEL_A_MAX = 4.210

# -----------------------------------------------------------------------------
# PACE identified params (2026-03-18, kp=[400,400,300,300,200,200,200])
# data: chirp_bias0_-0.5_..._120s_fmax5_200hz.pt
# log: 26_03_18_02-48-24/mean_199.pt
# NOTE: wrist_roll params replaced with wrist_flex (wrist_roll sys-id unreliable)
# -----------------------------------------------------------------------------

FETCH_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]

# FETCH_ARM_ARMATURE = {
#     "shoulder_pan_joint": 5.3262,
#     "shoulder_lift_joint": 2.0786,
#     "upperarm_roll_joint": 3.9381,
#     "elbow_flex_joint": 0.7278,
#     "forearm_roll_joint": 0.006014,
#     "wrist_flex_joint": 0.003366,
#     "wrist_roll_joint": 0.003366,  # copied from wrist_flex
# }

# FETCH_ARM_VISCOUS_FRICTION = {
#     "shoulder_pan_joint": 4.4546,
#     "shoulder_lift_joint": 4.0718,
#     "upperarm_roll_joint": 3.5383,
#     "elbow_flex_joint": 1.8606,
#     "forearm_roll_joint": 2.6730,
#     "wrist_flex_joint": 2.4211,
#     "wrist_roll_joint": 2.4211,  # copied from wrist_flex
# }

# FETCH_ARM_FRICTION = {
#     "shoulder_pan_joint": 0.2678,
#     "shoulder_lift_joint": 0.3921,
#     "upperarm_roll_joint": 0.2319,
#     "elbow_flex_joint": 0.1966,
#     "forearm_roll_joint": 0.1157,
#     "wrist_flex_joint": 0.1474,
#     "wrist_roll_joint": 0.1474,  # copied from wrist_flex
# }

# FETCH_ARM_ENCODER_BIAS = [
#     0.0251,
#     0.0796,
#     -0.0255,
#     -0.0800,
#     -0.0213,
#     -0.0550,
#     -0.0550,  # copied from wrist_flex
# ]
# FETCH_ARM_MAX_DELAY = 1


FETCH_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]
 
FETCH_ARM_ARMATURE = {
    "shoulder_pan_joint": 0.7610,
    "shoulder_lift_joint": 0.9946,
    "upperarm_roll_joint": 0.3076,
    "elbow_flex_joint": 0.5928,
    "forearm_roll_joint": 0.0035,
    "wrist_flex_joint": 0.0014,
    "wrist_roll_joint": 0.0014,   # copied from wrist_flex
}
 
FETCH_ARM_VISCOUS_FRICTION = {
    "shoulder_pan_joint": 4.2429,
    "shoulder_lift_joint": 3.3837,
    "upperarm_roll_joint": 3.1769,
    "elbow_flex_joint": 3.1387,
    "forearm_roll_joint": 3.1858,
    "wrist_flex_joint": 4.0530,
    "wrist_roll_joint": 4.0530,   # copied from wrist_flex
}
 
FETCH_ARM_FRICTION = {
    "shoulder_pan_joint": 0.2715,
    "shoulder_lift_joint": 0.3140,
    "upperarm_roll_joint": 0.2971,
    "elbow_flex_joint": 0.2059,
    "forearm_roll_joint": 0.2527,
    "wrist_flex_joint": 0.2037,
    "wrist_roll_joint": 0.2037,   # copied from wrist_flex
}
 
FETCH_ARM_ENCODER_BIAS = [
    -0.0069,
    -0.0025,
     0.0222,
     0.0483,
    -0.0477,
     0.0492,
     0.0,       # wrist_roll: no reliable estimate
]
 
FETCH_ARM_MAX_DELAY = 1   # ~0.5 physics steps identified
 
# Effort limits per joint (from URDF / real driver limits)
FETCH_ARM_EFFORT_LIMIT = {
    "shoulder_pan_joint": 33.82,
    "shoulder_lift_joint": 90.76,
    "upperarm_roll_joint": 56.94,
    "elbow_flex_joint": 46.18,
    "forearm_roll_joint": 29.35,
    "wrist_flex_joint": 25.70,
    "wrist_roll_joint": 7.36,
}
 
FETCH_ARM_STIFFNESS = {
    "shoulder_pan_joint": 300.0,
    "shoulder_lift_joint": 300.0,
    "upperarm_roll_joint": 250.0,
    "elbow_flex_joint": 250.0,
    "forearm_roll_joint": 200.0,
    "wrist_flex_joint": 200.0,
    "wrist_roll_joint": 200.0,
}
 
FETCH_ARM_DAMPING = {
    "shoulder_pan_joint": 40.0,
    "shoulder_lift_joint": 40.0,
    "upperarm_roll_joint": 25.0,
    "elbow_flex_joint": 25.0,
    "forearm_roll_joint": 10.0,
    "wrist_flex_joint": 10.0,
    "wrist_roll_joint": 10.0,
}
 
 
# =============================================================================
#  Shared spawn / init-state config
# =============================================================================
 
_FETCH_USD_PATH = (
    "/home/chengh-wang/Documents/git/isaaclab_fetch/"
    "assets/robots/fetch_description/fetch_isaaclab/fetch_isaaclab.usd"
)
 
_SPAWN_CFG = sim_utils.UsdFileCfg(
    usd_path=_FETCH_USD_PATH,
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
        fix_root_link=False,
    ),
)
 
_INIT_STATE_CFG = ArticulationCfg.InitialStateCfg(
    joint_pos={
        "r_wheel_joint": 0.0,
        "l_wheel_joint": 0.0,
        "head_pan_joint": 0.0,
        "head_tilt_joint": 0.0,
        "torso_lift_joint": 0.0,
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": 0.0,
        "upperarm_roll_joint": 0.0,
        "elbow_flex_joint": 0.0,
        "forearm_roll_joint": 0.0,
        "wrist_flex_joint": 0.0,
        "wrist_roll_joint": 0.0,
        "r_gripper_finger_joint": 0.01,
        "l_gripper_finger_joint": 0.01,
    },
)
 
# Non-arm actuators (shared between both configs)
_TORSO_CFG = ImplicitActuatorCfg(
    joint_names_expr=["torso_lift_joint"],
    effort_limit=450.0,
    velocity_limit=0.5,
    stiffness=400.0,
    damping=80.0,
)
 
_GRIPPER_CFG = ImplicitActuatorCfg(
    joint_names_expr=["r_gripper_finger_joint", "l_gripper_finger_joint"],
    effort_limit=60.0,
    velocity_limit=0.05,
    stiffness=1000.0,
    damping=100.0,
)
 
_HEAD_CFG = ImplicitActuatorCfg(
    joint_names_expr=["head_pan_joint", "head_tilt_joint"],
    effort_limit=1.0,
    velocity_limit=1.57,
    stiffness=1000.0,
    damping=100.0,
)
 
 
# =============================================================================
#  Config 1: PACE DC Motor  (explicit PD + saturation + delay + encoder bias)
#  - Pros: models communication delay, DC motor torque-speed saturation,
#          encoder bias; closest to PACE paper setup
#  - Cons: discrete-time PD (less accurate at large dt)
# =============================================================================
 
FETCH_CFG_PACE = ArticulationCfg(
    spawn=_SPAWN_CFG,
    init_state=_INIT_STATE_CFG,
    actuators={
        "fetch_torso": _TORSO_CFG,
        "fetch_arm": PaceDCMotorCfg(
            joint_names_expr=FETCH_ARM_JOINTS,
            saturation_effort=torch.tensor(
                [33.82, 90.76, 56.94, 46.18, 29.35, 25.70, 7.36],
                device="cuda:0",
            ),
            effort_limit=FETCH_ARM_EFFORT_LIMIT,
            velocity_limit=2.268,
            stiffness=FETCH_ARM_STIFFNESS,
            damping=FETCH_ARM_DAMPING,
            armature=FETCH_ARM_ARMATURE,
            friction=FETCH_ARM_FRICTION,
            viscous_friction=FETCH_ARM_VISCOUS_FRICTION,
            encoder_bias=FETCH_ARM_ENCODER_BIAS,
            max_delay=FETCH_ARM_MAX_DELAY,
        ),
        "fetch_gripper": _GRIPPER_CFG,
        "fetch_head": _HEAD_CFG,
    },
    soft_joint_pos_limit_factor=1.0,
)
 
 
# =============================================================================
#  Config 2: Implicit Actuator  (PhysX continuous-time PD)
#  - Pros: more accurate PD integration; simpler; faster simulation
#  - Cons: no delay, no DC saturation, no encoder bias
#  - Note: armature, friction, viscous_friction are set into PhysX solver
#          directly as physics engine parameters
# =============================================================================
 
FETCH_CFG_IMPLICIT = ArticulationCfg(
    spawn=_SPAWN_CFG,
    init_state=_INIT_STATE_CFG,
    actuators={
        "fetch_torso": _TORSO_CFG,
        "fetch_arm": ImplicitActuatorCfg(
            joint_names_expr=FETCH_ARM_JOINTS,
            effort_limit=FETCH_ARM_EFFORT_LIMIT,
            velocity_limit=2.268,
            stiffness=FETCH_ARM_STIFFNESS,
            damping=FETCH_ARM_DAMPING,
            armature=FETCH_ARM_ARMATURE,
            friction=FETCH_ARM_FRICTION,
            viscous_friction=FETCH_ARM_VISCOUS_FRICTION,
        ),
        "fetch_gripper": _GRIPPER_CFG,
        "fetch_head": _HEAD_CFG,
    },
    soft_joint_pos_limit_factor=0.9,
)
 
 
# =============================================================================
#  Action scale (for RL policy output → joint position delta)
#  Formula: 0.25 * effort_limit / stiffness
# =============================================================================
 
FETCH_ARM_ACTION_SCALE = {
    "shoulder_pan_joint":  0.25 * 33.82 / 400.0,   # 0.0211
    "shoulder_lift_joint": 0.25 * 90.76 / 400.0,   # 0.0567
    "upperarm_roll_joint": 0.25 * 56.94 / 300.0,   # 0.0474
    "elbow_flex_joint":    0.25 * 46.18 / 300.0,    # 0.0385
    "forearm_roll_joint":  0.25 * 29.35 / 200.0,    # 0.0367
    "wrist_flex_joint":    0.25 * 25.70 / 200.0,    # 0.0321
    "wrist_roll_joint":    0.25 * 7.36  / 200.0,    # 0.0092
}