# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Fetch robots.

The following configurations are available:

* :obj:`FETCH_PANDA_CFG`: Fetch robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR
    
##
# Configuration
##

FETCH_URDF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path="/home/chengh-wang/Documents/git/isaaclab_fetch/assets/robots/fetch_description/fetch_isaaclab.urdf",
        activate_contact_sensors=True,
        collision_from_visuals=True, 
        collider_type="convex_hull",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
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
            "bellows_joint": 0.0,
        },
    ),
    actuators={
        "fetch_wheels": ImplicitActuatorCfg(
            joint_names_expr=["l_wheel_joint", "r_wheel_joint"],
            effort_limit=8.5,
            velocity_limit=16,
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
        "fetch_bellows": ImplicitActuatorCfg(
            joint_names_expr=["bellows_joint"],
            effort_limit=5.0,
            velocity_limit=0.1,
            stiffness=1000.0,
            damping=100.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FETCH_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/chengh-wang/Documents/git/isaaclab_fetch/assets/robots/fetch_description/fetch_isaaclab/fetch_isaaclab.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4, fix_root_link=False,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # base
            "r_wheel_joint": 0.0,
            "l_wheel_joint": 0.0,
            "head_pan_joint": 0.0,
            "head_tilt_joint": 0.0,
            # fetch arm
            "torso_lift_joint": 0.0,
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": 0.0,
            "upperarm_roll_joint": 0.0,
            "elbow_flex_joint": 0.0,
            "forearm_roll_joint": 0.0,
            "wrist_flex_joint": 0.0,
            "wrist_roll_joint": 0.0,
            # "shoulder_pan_joint": 0.0,
            # "shoulder_lift_joint": 0.0,
            # "upperarm_roll_joint": 0.0,
            # "elbow_flex_joint": 0.0,
            # "forearm_roll_joint": 0.0,
            # "wrist_flex_joint": 0.0,
            # "wrist_roll_joint": 0.0,
            # # tool
            "r_gripper_finger_joint": 0.01,
            "l_gripper_finger_joint": 0.01,
        },
    ),
    actuators={
        "fetch_wheels": ImplicitActuatorCfg(
            joint_names_expr=["l_wheel_joint", "r_wheel_joint"],
            effort_limit=18.5,  # Nm - estimated from motor specs
            velocity_limit=16,  # 16 rad/s
            stiffness=0.0,  # Zero stiffness for velocity control
            damping=20.0,  # Damping for velocity tracking
        ),
                # ============================================================
        # Torso Lift - Prismatic joint
        # Real Fetch: travels 0.4m, effort ~450N
        # ============================================================
        "fetch_torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_lift_joint"],
            effort_limit=450.0,  # N (prismatic joint)
            velocity_limit=0.5,  # m/s
            stiffness=400.0,
            damping=80.0,
        ),
        
        # ============================================================
        # Arm Joints - 7-DOF with realistic torque limits from URDF
        # These are the actual motor torque limits from Fetch's URDF
        # ============================================================
        "fetch_shoulder_pan": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint"],
            effort_limit=33.82,  # Nm
            velocity_limit=1.256,  # rad/s
            stiffness=400.0,
            damping=80.0,
        ),
        "fetch_shoulder_lift": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_lift_joint"],
            effort_limit=131.76,  # Nm - highest torque joint
            velocity_limit=1.454,  # rad/s
            stiffness=400.0,
            damping=80.0,
        ),
        "fetch_upperarm_roll": ImplicitActuatorCfg(
            joint_names_expr=["upperarm_roll_joint"],
            effort_limit=76.94,  # Nm
            velocity_limit=1.571,  # rad/s
            stiffness=400.0,
            damping=80.0,
        ),
        "fetch_elbow_flex": ImplicitActuatorCfg(
            joint_names_expr=["elbow_flex_joint"],
            effort_limit=66.18,  # Nm
            velocity_limit=1.521,  # rad/s
            stiffness=400.0,
            damping=80.0,
        ),
        "fetch_forearm_roll": ImplicitActuatorCfg(
            joint_names_expr=["forearm_roll_joint"],
            effort_limit=29.35,  # Nm
            velocity_limit=1.571,  # rad/s
            stiffness=400.0,
            damping=80.0,
        ),
        "fetch_wrist_flex": ImplicitActuatorCfg(
            joint_names_expr=["wrist_flex_joint"],
            effort_limit=25.70,  # Nm
            velocity_limit=2.268,  # rad/s
            stiffness=400.0,
            damping=80.0,
        ),
        "fetch_wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["wrist_roll_joint"],
            effort_limit=7.36,  # Nm - smallest torque joint
            velocity_limit=2.268,  # rad/s
            stiffness=400.0,
            damping=80.0,
        ),
        
        # ============================================================
        # Gripper - Parallel jaw gripper
        # Real Fetch: max grasp force 245N, stroke 0.1m
        # ============================================================
        "fetch_gripper": ImplicitActuatorCfg(
            joint_names_expr=["r_gripper_finger_joint", "l_gripper_finger_joint"],
            effort_limit=60.0,  # N (prismatic fingers)
            velocity_limit=0.05,  # m/s
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

    #     "fetch_torso": ImplicitActuatorCfg(
    #         joint_names_expr=["torso_lift_joint"],
    #         effort_limit=10000.0,
    #         velocity_limit=0.1,
    #         stiffness=100000.0,
    #         damping=100000.0,
    #     ),
    #     "fetch_arm": ImplicitActuatorCfg(
    #         joint_names_expr=[
    #             "shoulder_pan_joint",
    #             "shoulder_lift_joint",
    #             "upperarm_roll_joint",
    #             "elbow_flex_joint",
    #             "forearm_roll_joint",
    #             "wrist_flex_joint",
    #             "wrist_roll_joint",
    #         ],
    #         effort_limit=10000.0,
    #         velocity_limit=1.0,
    #         stiffness=100000.0,
    #         damping=100000.0,
    #     ),
    #     "fetch_hand": ImplicitActuatorCfg(
    #         joint_names_expr=["r_gripper_finger_joint", "l_gripper_finger_joint"],
    #         effort_limit=1000.0,
    #         velocity_limit=0.05,
    #         stiffness=5000,
    #         damping=1000,
    #     ),
    #     "fetch_head": ImplicitActuatorCfg(
    #         joint_names_expr=["head_pan_joint", "head_tilt_joint"],
    #         effort_limit=1.0,
    #         velocity_limit=1.57,
    #         stiffness=100,
    #         damping=10,
    #     ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Fetch robot."""