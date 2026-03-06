# save as: fetch_project/tasks/manipulation/reach/metrics_wrapper.py

import torch
from isaaclab.utils.math import quat_error_magnitude


class KeypointMetricsWrapper:
    """Thin wrapper that injects tracking metrics into env.extras["log"].
    
    Wrap AFTER RslRlVecEnvWrapper so it intercepts the infos dict.
    """

    def __init__(self, env, cube_side=0.3):
        self._env = env
        self._cube_side = cube_side

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, actions):
        obs, rew, dones, infos = self._env.step(actions)

        # Access the unwrapped Isaac Lab env
        base_env = self._env.unwrapped

        # EE pose
        asset = base_env.scene["robot"]
        body_idx = asset.find_bodies("wrist_roll_link")[0][0]
        ee_state = asset.data.body_state_w[:, body_idx, :]
        ee_pos, ee_quat = ee_state[:, :3], ee_state[:, 3:7]

        # Goal pose
        cmd = base_env.command_manager.get_command("ee_pose")
        g_pos, g_quat = cmd[:, :3], cmd[:, 3:7]

        # Metrics
        pos_err = torch.norm(ee_pos - g_pos, dim=-1)
        ori_err = quat_error_magnitude(ee_quat, g_quat)

        from .mdp.utils import keypoint_distance
        _, kp_dist = keypoint_distance(ee_pos, ee_quat, g_pos, g_quat, self._cube_side)

        stage = getattr(base_env, "_kp_curriculum_stage", 0)

        # Inject into infos — rsl_rl logs these directly
        if "log" not in infos:
            infos["log"] = {}
        infos["log"]["Metrics/pos_error_cm"] = pos_err.mean().item() * 100
        infos["log"]["Metrics/pos_median_cm"] = pos_err.median().item() * 100
        infos["log"]["Metrics/ori_error_deg"] = torch.rad2deg(ori_err).mean().item()
        infos["log"]["Metrics/kp_dist_cm"] = kp_dist.mean().item() * 100
        infos["log"]["Metrics/curriculum_stage"] = float(stage)

        return obs, rew, dones, infos