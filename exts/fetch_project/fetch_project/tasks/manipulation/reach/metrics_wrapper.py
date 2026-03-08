import torch
from isaaclab.utils.math import quat_error_magnitude


class KeypointMetricsWrapper:

    def __init__(self, env, cube_side=0.3):
        self._env = env
        self._cube_side = cube_side
        self._initialized = False

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _lazy_init(self, num_envs, device):
        self._best_pos_err = torch.full((num_envs,), 1e6, device=device)
        self._best_ori_err = torch.full((num_envs,), 1e6, device=device)
        self._best_kp_dist = torch.full((num_envs,), 1e6, device=device)
        self._initialized = True

    def step(self, actions):
        obs, rew, dones, infos = self._env.step(actions)

        base_env = self._env.unwrapped

        if not self._initialized:
            self._lazy_init(base_env.num_envs, base_env.device)

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

            # Reset for next episode
            self._best_pos_err[done_ids] = 1e6
            self._best_ori_err[done_ids] = 1e6
            self._best_kp_dist[done_ids] = 1e6

        # Always log curriculum stage and instantaneous (for training curve)
        stage = getattr(base_env, "_kp_curriculum_stage", 0)
        infos["log"]["Metrics/curriculum_stage"] = float(stage)
        infos["log"]["Metrics/instant_kp_cm"] = kp_dist.mean().item() * 100

        return obs, rew, dones, infos