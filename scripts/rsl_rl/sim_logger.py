"""
SimLogger — CSV logger for Isaac Lab simulation play.

Outputs the same CSV format as the C++ DeployLogger so both can be
visualized with the same fetch_log_viewer.html.

Usage in play.py:
    from sim_logger import SimLogger
    logger = SimLogger(env, log_dir="logs/play_logs")
    ...
    while simulation_app.is_running():
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        logger.log(env, obs, actions)
    logger.close()
"""

import os
import time
import csv
import torch
import numpy as np
from datetime import datetime


JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "upperarm_roll",
    "elbow_flex", "forearm_roll", "wrist_flex", "wrist_roll",
]

ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
    "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint",
]


class SimLogger:
    """Logs simulation data in the same CSV format as the C++ DeployLogger."""

    def __init__(self, env, log_dir="logs/play_logs", env_index=0, cube_side=0.3):
        """
        Args:
            env: RslRlVecEnvWrapper or gymnasium env
            log_dir: directory to save CSV files
            env_index: which env to log (default 0)
            cube_side: keypoint cube side (must match training)
        """
        self.env_index = env_index
        self.cube_side = cube_side
        self.step_count = 0

        # Unwrap to get base env
        self.base_env = env.unwrapped if hasattr(env, 'unwrapped') else env

        # Resolve arm joint indices
        robot = self.base_env.scene["robot"]
        self.arm_joint_ids = robot.find_joints(ARM_JOINT_NAMES)[0]
        self.ee_body_idx = robot.find_bodies("wrist_roll_link")[0][0]

        # Get action config for scale
        action_cfg = self.base_env.cfg.actions
        if hasattr(action_cfg.arm_action, 'scale'):
            scale = action_cfg.arm_action.scale
            if isinstance(scale, dict):
                self.arm_scale = [scale[n] for n in ARM_JOINT_NAMES]
            elif isinstance(scale, (int, float)):
                self.arm_scale = [float(scale)] * 7
            else:
                self.arm_scale = list(scale) if hasattr(scale, '__iter__') else [float(scale)] * 7
        else:
            self.arm_scale = [0.5] * 7

        # Get stiffness/damping for effort computation
        stiffness = self.base_env.cfg.actions.arm_action
        # Try to get from actuator config
        self.arm_kp = [400, 400, 300, 300, 200, 200, 200]  # defaults
        self.arm_kd = [40, 40, 25, 25, 10, 10, 10]

        # Create log directory and file
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"sim_{timestamp}.csv")

        self.file = open(self.filepath, "w", newline="")
        self.writer = csv.writer(self.file)

        # Write header
        self._write_header()

        self.t_start = time.time()
        self.t_last = self.t_start
        self.last_actions = None

        print(f"[SimLogger] Logging env[{env_index}] to {self.filepath}")
        print(f"[SimLogger] Arm joint IDs: {self.arm_joint_ids}")
        print(f"[SimLogger] Arm scale: {self.arm_scale}")

    def _write_header(self):
        header = ["wall_time", "step", "loop_dt"]

        # Observations — we'll compute obs dim from first log call
        # For now, use the known structure: 7 + 9 + 3 + 3 + 9 = 31
        for i in range(31):
            header.append(f"obs_policy_{i}")

        # Raw actions (9D)
        for i in range(9):
            header.append(f"raw_act_{i}")

        # Processed actions (9D)
        for name in JOINT_NAMES:
            header.append(f"proc_arm_{name}")
        header += ["proc_base_v", "proc_base_w"]

        # Arm cmd_pos, actual_pos, actual_vel, effort_cmd, effort_actual
        for name in JOINT_NAMES:
            header.append(f"cmd_pos_{name}")
        for name in JOINT_NAMES:
            header.append(f"act_pos_{name}")
        for name in JOINT_NAMES:
            header.append(f"act_vel_{name}")
        for name in JOINT_NAMES:
            header.append(f"eff_cmd_{name}")
        for name in JOINT_NAMES:
            header.append(f"eff_act_{name}")

        # Base
        header += ["base_cmd_v", "base_cmd_w"]
        header += ["base_act_vx", "base_act_vy", "base_act_vz"]
        header += ["base_act_wx", "base_act_wy", "base_act_wz"]

        # EE world
        header += ["ee_w_x", "ee_w_y", "ee_w_z", "ee_w_qw", "ee_w_qx", "ee_w_qy", "ee_w_qz"]
        # EE base
        header += ["ee_b_x", "ee_b_y", "ee_b_z", "ee_b_qw", "ee_b_qx", "ee_b_qy", "ee_b_qz"]
        # Odom
        header += ["odom_x", "odom_y", "odom_z", "odom_qw", "odom_qx", "odom_qy", "odom_qz"]
        # Target
        header += ["target_x", "target_y", "target_z", "target_qw", "target_qx", "target_qy", "target_qz"]

        # Keypoint delta
        for k in range(3):
            for d in "xyz":
                header.append(f"kp_delta_{k}_{d}")

        # Metrics
        header += ["ee_dist", "ee_valid"]

        self.writer.writerow(header)

    def log(self, env, obs, actions):
        """Log one step. Call after env.step()."""
        t_now = time.time()
        wall_time = t_now - self.t_start
        loop_dt = t_now - self.t_last
        self.t_last = t_now

        i = self.env_index
        base_env = self.base_env
        robot = base_env.scene["robot"]

        # ── Extract tensors for env[i] ──
        if hasattr(obs, "keys") and "policy" in obs.keys():
            obs_vec = obs["policy"][i].cpu().numpy()
        elif isinstance(obs, dict):
            obs_vec = obs.get("policy", obs.get("obs", list(obs.values())[0]))[i].cpu().numpy()
        else:
            obs_vec = obs[i].cpu().numpy()

        actions_np = actions[i].cpu().numpy()

        # Joint state
        joint_pos = robot.data.joint_pos[i].cpu().numpy()
        joint_vel = robot.data.joint_vel[i].cpu().numpy()

        # Arm actual pos/vel
        arm_pos = joint_pos[self.arm_joint_ids]
        arm_vel = joint_vel[self.arm_joint_ids]

        # Default joint pos for offset
        default_pos = robot.data.default_joint_pos[i].cpu().numpy()
        arm_default = default_pos[self.arm_joint_ids]

        # Compute cmd_pos: default + scale * raw_action
        arm_raw = actions_np[:7]
        arm_cmd_pos = arm_default + np.array(self.arm_scale) * arm_raw

        # Compute effort: kp * (cmd - actual) - kd * vel
        arm_eff_cmd = np.array(self.arm_kp) * (arm_cmd_pos - arm_pos) - np.array(self.arm_kd) * arm_vel

        # Actual effort (from sim applied forces if available)
        if hasattr(robot.data, 'applied_torque') and robot.data.applied_torque is not None:
            all_torques = robot.data.applied_torque[i].cpu().numpy()
            arm_eff_actual = all_torques[self.arm_joint_ids]
        else:
            arm_eff_actual = arm_eff_cmd  # fallback: same as commanded

        # Base velocity
        base_raw = actions_np[7:9] if len(actions_np) >= 9 else [0, 0]
        base_cmd_v = float(base_raw[0]) if len(base_raw) > 0 else 0.0
        base_cmd_w = float(base_raw[1]) if len(base_raw) > 1 else 0.0

        root_lin_vel = robot.data.root_lin_vel_b[i].cpu().numpy()
        root_ang_vel = robot.data.root_ang_vel_b[i].cpu().numpy()

        # Processed actions
        proc_arm = arm_cmd_pos.tolist()
        proc_actions = proc_arm + [base_cmd_v, base_cmd_w]

        # EE pose
        ee_state = robot.data.body_state_w[i, self.ee_body_idx, :]
        ee_pos_w = ee_state[:3].cpu().numpy()
        ee_quat_w = ee_state[3:7].cpu().numpy()  # w,x,y,z

        # Base pose
        root_pos = robot.data.root_pos_w[i].cpu().numpy()
        root_quat = robot.data.root_quat_w[i].cpu().numpy()

        # EE in base frame
        from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul
        root_quat_t = robot.data.root_quat_w[i:i+1]
        root_pos_t = robot.data.root_pos_w[i:i+1]
        ee_pos_t = ee_state[:3].unsqueeze(0)
        ee_quat_t = ee_state[3:7].unsqueeze(0)

        rel_pos_b = quat_apply_inverse(root_quat_t, ee_pos_t - root_pos_t)[0].cpu().numpy()
        rel_quat_b = quat_mul(quat_conjugate(root_quat_t), ee_quat_t)[0].cpu().numpy()

        # Target
        cmd = base_env.command_manager.get_command("ee_pose")
        target_pos = cmd[i, :3].cpu().numpy()
        target_quat = cmd[i, 3:7].cpu().numpy()

        # Keypoint delta
        from fetch_project.tasks.manipulation.reach.mdp.utils import pose_to_keypoints
        ee_kps = pose_to_keypoints(
            ee_state[:3].unsqueeze(0), ee_state[3:7].unsqueeze(0), self.cube_side
        )
        goal_kps = pose_to_keypoints(
            cmd[i:i+1, :3], cmd[i:i+1, 3:7], self.cube_side
        )
        delta_w = goal_kps - ee_kps  # (1, 3, 3)
        # Rotate to base frame
        root_quat_conj = quat_conjugate(root_quat_t)
        kp_delta_flat = []
        for k in range(3):
            d = quat_apply_inverse(root_quat_t, delta_w[0, k:k+1, :])[0].cpu().numpy()
            kp_delta_flat.extend(d.tolist())

        # EE distance
        ee_dist = float(np.linalg.norm(target_pos - ee_pos_w))

        # ── Build row ──
        row = [f"{wall_time:.6f}", self.step_count, f"{loop_dt:.6f}"]

        # Obs (pad/truncate to 31)
        obs_list = obs_vec.tolist()
        while len(obs_list) < 31:
            obs_list.append(0.0)
        row += [f"{v:.6f}" for v in obs_list[:31]]

        # Raw actions (pad to 9)
        raw_list = actions_np.tolist()
        while len(raw_list) < 9:
            raw_list.append(0.0)
        row += [f"{v:.6f}" for v in raw_list[:9]]

        # Processed actions (9)
        row += [f"{v:.6f}" for v in proc_actions]

        # Arm: cmd, actual, vel, eff_cmd, eff_actual
        row += [f"{v:.6f}" for v in arm_cmd_pos]
        row += [f"{v:.6f}" for v in arm_pos]
        row += [f"{v:.6f}" for v in arm_vel]
        row += [f"{v:.6f}" for v in arm_eff_cmd]
        row += [f"{v:.6f}" for v in arm_eff_actual]

        # Base
        row += [f"{base_cmd_v:.6f}", f"{base_cmd_w:.6f}"]
        row += [f"{v:.6f}" for v in root_lin_vel]
        row += [f"{v:.6f}" for v in root_ang_vel]

        # EE world (pos + quat wxyz)
        row += [f"{v:.6f}" for v in ee_pos_w]
        row += [f"{ee_quat_w[0]:.6f}", f"{ee_quat_w[1]:.6f}", f"{ee_quat_w[2]:.6f}", f"{ee_quat_w[3]:.6f}"]

        # EE base
        row += [f"{v:.6f}" for v in rel_pos_b]
        row += [f"{rel_quat_b[0]:.6f}", f"{rel_quat_b[1]:.6f}", f"{rel_quat_b[2]:.6f}", f"{rel_quat_b[3]:.6f}"]

        # Odom
        row += [f"{v:.6f}" for v in root_pos]
        row += [f"{root_quat[0]:.6f}", f"{root_quat[1]:.6f}", f"{root_quat[2]:.6f}", f"{root_quat[3]:.6f}"]

        # Target
        row += [f"{v:.6f}" for v in target_pos]
        row += [f"{target_quat[0]:.6f}", f"{target_quat[1]:.6f}", f"{target_quat[2]:.6f}", f"{target_quat[3]:.6f}"]

        # Keypoint delta (9)
        row += [f"{v:.6f}" for v in kp_delta_flat]

        # Metrics
        row += [f"{ee_dist:.6f}", 1]

        self.writer.writerow(row)
        self.step_count += 1

        # Flush periodically
        if self.step_count % 100 == 0:
            self.file.flush()

    def close(self):
        """Flush and close the CSV file."""
        if self.file and not self.file.closed:
            self.file.flush()
            self.file.close()
            elapsed = time.time() - self.t_start
            print(f"[SimLogger] Saved {self.step_count} steps ({elapsed:.1f}s) to {self.filepath}")
