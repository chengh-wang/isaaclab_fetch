import torch
from isaaclab.utils.math import quat_error_magnitude


class KeypointMetricsWrapper:

    def __init__(self, env, cube_side=0.3):
        self._env = env
        self._cube_side = cube_side
        self._initialized = False
        self._debug_step = 0
        self._debug_interval = 50  # Log debug info every N steps

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _lazy_init(self, num_envs, device):
        self._best_pos_err = torch.full((num_envs,), 1e6, device=device)
        self._best_ori_err = torch.full((num_envs,), 1e6, device=device)
        self._best_kp_dist = torch.full((num_envs,), 1e6, device=device)
        self._ep_len = torch.zeros(num_envs, device=device)
        self._initialized = True

    def step(self, actions):
        obs, rew, dones, infos = self._env.step(actions)

        base_env = self._env.unwrapped
        self._debug_step += 1

        if not self._initialized:
            self._lazy_init(base_env.num_envs, base_env.device)

        self._ep_len += 1

        # EE pose
        asset = base_env.scene["robot"]
        body_idx = asset.find_bodies("wrist_roll_link")[0][0]
        ee_state = asset.data.body_state_w[:, body_idx, :]
        ee_pos, ee_quat = ee_state[:, :3], ee_state[:, 3:7]

        # Goal pose
        cmd = base_env.command_manager.get_command("ee_pose")
        g_pos, g_quat = cmd[:, :3], cmd[:, 3:7]

        # Current errors
        pos_err = torch.norm(ee_pos - g_pos, dim=-1)
        ori_err = quat_error_magnitude(ee_quat, g_quat)

        from .mdp.utils import keypoint_distance
        _, kp_dist = keypoint_distance(ee_pos, ee_quat, g_pos, g_quat, self._cube_side)

        # Update best (minimum) per env
        self._best_pos_err = torch.min(self._best_pos_err, pos_err)
        self._best_ori_err = torch.min(self._best_ori_err, ori_err)
        self._best_kp_dist = torch.min(self._best_kp_dist, kp_dist)

        # On episode end, report best errors then reset
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)

        if "log" not in infos:
            infos["log"] = {}

        if len(done_ids) > 0:
            infos["log"]["Metrics/best_pos_cm"] = self._best_pos_err[done_ids].mean().item() * 100
            infos["log"]["Metrics/best_ori_deg"] = torch.rad2deg(self._best_ori_err[done_ids]).mean().item()
            infos["log"]["Metrics/best_kp_cm"] = self._best_kp_dist[done_ids].mean().item() * 100
            infos["log"]["Metrics/mean_ep_len"] = self._ep_len[done_ids].mean().item()

            # Reset for next episode
            self._best_pos_err[done_ids] = 1e6
            self._best_ori_err[done_ids] = 1e6
            self._best_kp_dist[done_ids] = 1e6
            self._ep_len[done_ids] = 0

        # Always log curriculum stage and instantaneous (for training curve)
        stage = getattr(base_env, "_kp_curriculum_stage", 0)
        infos["log"]["Metrics/curriculum_stage"] = float(stage)
        infos["log"]["Metrics/instant_kp_cm"] = kp_dist.mean().item() * 100

        # =====================================================================
        # DEBUG: Detailed diagnostics (logged every step to wandb)
        # =====================================================================

        # --- 1. Robot base state ---
        root_pos = asset.data.root_pos_w  # (N, 3)
        root_quat = asset.data.root_quat_w  # (N, 4)
        projected_gravity = asset.data.projected_gravity_b  # (N, 3)

        infos["log"]["Debug/base_height_mean"] = root_pos[:, 2].mean().item()
        infos["log"]["Debug/base_height_min"] = root_pos[:, 2].min().item()
        # Tilt = magnitude of xy components of projected gravity (0=flat, 1=sideways)
        tilt = torch.norm(projected_gravity[:, :2], dim=-1)
        infos["log"]["Debug/tilt_mean"] = tilt.mean().item()
        infos["log"]["Debug/tilt_max"] = tilt.max().item()
        infos["log"]["Debug/tilt_gt_0.3"] = (tilt > 0.3).float().mean().item()
        infos["log"]["Debug/tilt_gt_0.5"] = (tilt > 0.5).float().mean().item()

        # --- 2. Base velocity ---
        root_lin_vel = asset.data.root_lin_vel_b  # (N, 3)
        root_ang_vel = asset.data.root_ang_vel_b  # (N, 3)
        infos["log"]["Debug/base_lin_speed"] = torch.norm(root_lin_vel, dim=-1).mean().item()
        infos["log"]["Debug/base_ang_speed"] = torch.norm(root_ang_vel, dim=-1).mean().item()
        infos["log"]["Debug/base_lin_vel_z"] = root_lin_vel[:, 2].mean().item()

        # --- 3. Action magnitudes ---
        act = base_env.action_manager.action
        n_arm = 8  # torso + 7 arm joints
        if act.shape[1] >= n_arm + 2:
            arm_act = act[:, :n_arm]
            base_act = act[:, n_arm:n_arm + 2]
            infos["log"]["Debug/arm_action_mean_abs"] = arm_act.abs().mean().item()
            infos["log"]["Debug/arm_action_max_abs"] = arm_act.abs().max().item()
            infos["log"]["Debug/base_action_mean_abs"] = base_act.abs().mean().item()
            infos["log"]["Debug/base_action_max_abs"] = base_act.abs().max().item()
        infos["log"]["Debug/action_mean_abs"] = act.abs().mean().item()
        infos["log"]["Debug/action_std"] = act.std().item()

        # --- 4. Reward magnitude ---
        infos["log"]["Debug/reward_mean"] = rew.mean().item()
        infos["log"]["Debug/reward_std"] = rew.std().item()
        infos["log"]["Debug/reward_min"] = rew.min().item()
        infos["log"]["Debug/reward_max"] = rew.max().item()

        # --- 5. Per-body contact forces ---
        if self._debug_step % self._debug_interval == 0:
            try:
                contact_sensor = base_env.scene.sensors["contact_forces"]
                net_forces = contact_sensor.data.net_forces_w_history  # (N, H, B, 3)
                force_mag = torch.max(torch.norm(net_forces, dim=-1), dim=1)[0]  # (N, B)

                # Get all body names from the sensor
                body_names = contact_sensor.body_names
                # Log top contacting bodies (by mean force)
                mean_forces = force_mag.mean(dim=0)  # (B,)
                contact_count = (force_mag > 1.0).float().mean(dim=0)  # fraction of envs with contact > 1N

                # Log top 10 contacting bodies
                top_k = min(10, len(body_names))
                top_indices = torch.argsort(mean_forces, descending=True)[:top_k]
                for rank, idx in enumerate(top_indices):
                    idx = idx.item()
                    name = body_names[idx] if idx < len(body_names) else f"body_{idx}"
                    infos["log"][f"Contact/{rank}_{name}_force"] = mean_forces[idx].item()
                    infos["log"][f"Contact/{rank}_{name}_frac"] = contact_count[idx].item()

                # Total bodies in contact
                total_contacts = (force_mag > 1.0).float().sum(dim=1)  # per env
                infos["log"]["Debug/total_contact_bodies_mean"] = total_contacts.mean().item()
                infos["log"]["Debug/total_contact_bodies_max"] = total_contacts.max().item()
            except Exception as e:
                if self._debug_step % (self._debug_interval * 10) == 0:
                    print(f"[DEBUG] Contact logging error: {e}")

        # --- 6. Joint positions vs limits ---
        joint_pos = asset.data.joint_pos
        joint_pos_limits = asset.data.soft_joint_pos_limits  # (N, J, 2)
        lower = joint_pos_limits[:, :, 0]
        upper = joint_pos_limits[:, :, 1]
        at_lower = (joint_pos <= lower + 0.01).float().sum(dim=1)
        at_upper = (joint_pos >= upper - 0.01).float().sum(dim=1)
        infos["log"]["Debug/joints_at_limit_mean"] = (at_lower + at_upper).mean().item()

        # --- 7. NaN/Inf check ---
        has_nan_obs = torch.isnan(obs).any().item() or torch.isinf(obs).any().item()
        has_nan_rew = torch.isnan(rew).any().item() or torch.isinf(rew).any().item()
        has_nan_act = torch.isnan(act).any().item() or torch.isinf(act).any().item()
        if has_nan_obs or has_nan_rew or has_nan_act:
            infos["log"]["Debug/has_nan"] = 1.0
            if self._debug_step % self._debug_interval == 0:
                print(f"[DEBUG NaN] step={self._debug_step} obs_nan={has_nan_obs} rew_nan={has_nan_rew} act_nan={has_nan_act}")
        else:
            infos["log"]["Debug/has_nan"] = 0.0

        return obs, rew, dones, infos