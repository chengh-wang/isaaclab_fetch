"""Play a checkpoint and collect steady-state tracking metrics.

Metrics recorded only after the EE has *settled* near the target:
  1. Arrival   : pos_error < 0.2 m  (starts the clock)
  2. Settled   : pos < 0.05 m AND ori < 0.05 rad AND speed < 0.1 m/s
                 for N consecutive steps after arrival
  3. Recording : accumulate errors while settled

Reported per-env and aggregate (mean / median / min / max):
  - Position error  (m, cm)
  - Orientation error — geodesic  (rad, deg)
  - Orientation error — roll / pitch / yaw  (deg)
  - Keypoint mean distance  (m, cm)
  - Reach rate  (% of envs that ever settled)
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL and collect metrics.")
cli_args.add_rsl_rl_args(parser)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# Video arguments
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=300, help="Length of the recorded video (in steps).")
# Camera arguments
parser.add_argument(
    "--camera_offset", type=float, nargs=3, default=[2.5, 2.5, 2.0],
    help="Camera position offset from robot [x, y, z]",
)
parser.add_argument(
    "--camera_target_offset", type=float, nargs=3, default=[0.0, 0.0, 0.5],
    help="Camera look-at offset from robot [x, y, z]",
)
# Metric arguments
parser.add_argument("--play_steps", type=int, default=1000, help="Total play steps to run.")
parser.add_argument("--arrival_pos_thresh", type=float, default=0.2, help="Arrival: pos error threshold (m).")
parser.add_argument("--settled_pos_thresh", type=float, default=0.05, help="Settled: pos error threshold (m).")
parser.add_argument("--settled_ori_thresh", type=float, default=0.05, help="Settled: ori error threshold (rad).")
parser.add_argument("--settled_speed_thresh", type=float, default=0.1, help="Settled: EE speed threshold (m/s).")
parser.add_argument("--settled_hold_steps", type=int, default=20, help="Consecutive steps required to confirm settled.")
parser.add_argument("--print_per_env", action="store_true", default=False, help="Print per-env metrics.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import math

from rsl_rl.runners import OnPolicyRunner

import fetch_project.tasks  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_error_magnitude, quat_mul


# ============================================================================
# Math helpers
# ============================================================================

def quat_to_rpy(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w,x,y,z) to roll-pitch-yaw (intrinsic ZYX).

    Args:
        q: shape (..., 4), order [w, x, y, z]
    Returns:
        rpy: shape (..., 3), order [roll, pitch, yaw] in radians
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis) — clamp for numerical safety
    sinp = 2.0 * (w * y - z * x)
    sinp = sinp.clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)

    # Yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def quat_error_rpy(q_current: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    """Compute per-axis RPY error between two quaternions.

    Args:
        q_current, q_target: shape (N, 4), order [w, x, y, z]
    Returns:
        rpy_error: shape (N, 3), signed [roll, pitch, yaw] error in radians
    """
    # q_err = q_target^{-1} * q_current
    # For unit quaternion, inverse = conjugate = (w, -x, -y, -z)
    q_target_inv = q_target.clone()
    q_target_inv[:, 1:] *= -1.0
    q_err = quat_mul(q_target_inv, q_current)

    # Handle double cover: ensure w >= 0
    neg_w = q_err[:, 0] < 0
    q_err[neg_w] *= -1.0

    return quat_to_rpy(q_err)


# ============================================================================
# Metrics tracker
# ============================================================================

class SettledMetricsTracker:
    """Track per-env arrival → settled → recording logic."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        arrival_pos_thresh: float = 0.2,
        settled_pos_thresh: float = 0.05,
        settled_ori_thresh: float = 0.05,
        settled_speed_thresh: float = 0.1,
        settled_hold_steps: int = 20,
    ):
        self.num_envs = num_envs
        self.device = device

        # Thresholds
        self.arrival_pos_thresh = arrival_pos_thresh
        self.settled_pos_thresh = settled_pos_thresh
        self.settled_ori_thresh = settled_ori_thresh
        self.settled_speed_thresh = settled_speed_thresh
        self.settled_hold_steps = settled_hold_steps

        # State
        self.arrived = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.consecutive_settled = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.is_settled = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ever_settled = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Accumulators  (only filled while settled)
        self.pos_err_sum = torch.zeros(num_envs, device=device)
        self.ori_err_sum = torch.zeros(num_envs, device=device)       # geodesic
        self.rpy_err_sum = torch.zeros(num_envs, 3, device=device)    # |roll|, |pitch|, |yaw|
        self.kp_err_sum = torch.zeros(num_envs, device=device)
        self.settled_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    def step(
        self,
        pos_err: torch.Tensor,      # (N,)
        ori_err: torch.Tensor,      # (N,)  geodesic rad
        rpy_err: torch.Tensor,      # (N, 3) signed rad
        kp_err: torch.Tensor,       # (N,)
        ee_speed: torch.Tensor,     # (N,)
    ):
        N = self.num_envs

        # ---- Phase 1: Arrival ----
        just_arrived = (~self.arrived) & (pos_err < self.arrival_pos_thresh)
        self.arrived |= just_arrived

        # If drifted far away after arrival, reset arrival
        drifted = self.arrived & (pos_err > self.arrival_pos_thresh)
        self.arrived[drifted] = False
        self.consecutive_settled[drifted] = 0
        self.is_settled[drifted] = False

        # ---- Phase 2: Settled check (only for arrived envs) ----
        meets_settled = (
            self.arrived
            & (pos_err < self.settled_pos_thresh)
            & (ori_err < self.settled_ori_thresh)
            & (ee_speed < self.settled_speed_thresh)
        )

        # Consecutive counter
        self.consecutive_settled[meets_settled] += 1
        self.consecutive_settled[~meets_settled & self.arrived] = 0

        # Transition to settled
        newly_settled = (~self.is_settled) & (self.consecutive_settled >= self.settled_hold_steps)
        self.is_settled |= newly_settled
        self.ever_settled |= newly_settled

        # If no longer meeting settled after being settled, stay settled
        # (we keep recording — settled is a "latch" once confirmed)

        # ---- Phase 3: Record (while is_settled AND still meeting criteria) ----
        recording = self.is_settled & meets_settled
        if recording.any():
            self.pos_err_sum[recording] += pos_err[recording]
            self.ori_err_sum[recording] += ori_err[recording]
            self.rpy_err_sum[recording] += rpy_err[recording].abs()
            self.kp_err_sum[recording] += kp_err[recording]
            self.settled_count[recording] += 1

    def report(self, print_per_env: bool = False):
        """Print aggregate and optionally per-env metrics."""
        has_data = self.settled_count > 0
        n_reached = has_data.sum().item()
        reach_rate = n_reached / self.num_envs * 100

        print("\n" + "=" * 72)
        print(f" SETTLED METRICS REPORT   ({n_reached}/{self.num_envs} envs reached, "
              f"reach rate = {reach_rate:.1f}%)")
        print("=" * 72)

        if n_reached == 0:
            print("  No envs reached settled state. Try increasing --play_steps or relaxing thresholds.")
            print("=" * 72)
            return

        # Per-env means (only for envs with data)
        cnt = self.settled_count[has_data].float()
        pos_mean = self.pos_err_sum[has_data] / cnt
        ori_mean = self.ori_err_sum[has_data] / cnt
        rpy_mean = self.rpy_err_sum[has_data] / cnt.unsqueeze(-1)
        kp_mean = self.kp_err_sum[has_data] / cnt

        # ---- Per-env table ----
        if print_per_env:
            env_indices = torch.where(has_data)[0]
            print(f"\n  {'Env':>5}  {'Pos(cm)':>8}  {'Ori(deg)':>9}  {'Roll(°)':>8}  "
                  f"{'Pitch(°)':>9}  {'Yaw(°)':>8}  {'KP(cm)':>8}  {'Steps':>6}")
            print("  " + "-" * 68)
            for i in range(len(env_indices)):
                eidx = env_indices[i].item()
                ii = i  # index into filtered arrays
                print(f"  {eidx:5d}  {pos_mean[ii].item()*100:8.2f}  {math.degrees(ori_mean[ii].item()):9.2f}  "
                      f"{math.degrees(rpy_mean[ii, 0].item()):8.2f}  {math.degrees(rpy_mean[ii, 1].item()):9.2f}  "
                      f"{math.degrees(rpy_mean[ii, 2].item()):8.2f}  {kp_mean[ii].item()*100:8.2f}  "
                      f"{self.settled_count[eidx].item():6d}")

        # ---- Aggregate stats ----
        def _stats(t: torch.Tensor, scale: float = 1.0, is_deg: bool = False):
            """Return formatted string: mean ± std  [min, median, max]"""
            t = t * scale
            if is_deg:
                t = t * (180.0 / math.pi)
            return (f"mean={t.mean().item():7.3f}  std={t.std().item():7.3f}  "
                    f"median={t.median().item():7.3f}  min={t.min().item():7.3f}  max={t.max().item():7.3f}")

        print(f"\n  Aggregate over {n_reached} settled envs:")
        print(f"  Position error (cm)   : {_stats(pos_mean, scale=100)}")
        print(f"  Ori error geodesic (°): {_stats(ori_mean, is_deg=True)}")
        print(f"  Roll  error (°)       : {_stats(rpy_mean[:, 0], is_deg=True)}")
        print(f"  Pitch error (°)       : {_stats(rpy_mean[:, 1], is_deg=True)}")
        print(f"  Yaw   error (°)       : {_stats(rpy_mean[:, 2], is_deg=True)}")
        print(f"  Keypoint error (cm)   : {_stats(kp_mean, scale=100)}")
        print(f"  Reach rate            : {reach_rate:.1f}%  ({n_reached}/{self.num_envs})")
        print("=" * 72 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    """Play with RSL-RL agent and collect tracking metrics."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, device=args_cli.device)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Camera setup for video
    if args_cli.video:
        env_cfg.viewer.eye = tuple(args_cli.camera_offset)
        env_cfg.viewer.lookat = tuple(args_cli.camera_target_offset)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # locate checkpoint
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    log_dir = os.path.dirname(resume_path)

    # video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loaded model checkpoint from: {resume_path}")

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(ppo_runner.alg.policy, export_model_dir, filename="policy.onnx")

    # ---- Setup metrics tracker ----
    base_env = env.unwrapped
    device = base_env.device
    num_envs = base_env.num_envs
    asset = base_env.scene["robot"]
    body_idx = asset.find_bodies("wrist_roll_link")[0][0]

    # Import keypoint util
    from fetch_project.tasks.manipulation.reach.mdp.utils import keypoint_distance

    tracker = SettledMetricsTracker(
        num_envs=num_envs,
        device=device,
        arrival_pos_thresh=args_cli.arrival_pos_thresh,
        settled_pos_thresh=args_cli.settled_pos_thresh,
        settled_ori_thresh=args_cli.settled_ori_thresh,
        settled_speed_thresh=args_cli.settled_speed_thresh,
        settled_hold_steps=args_cli.settled_hold_steps,
    )

    # ---- Run ----
    obs, _ = env.get_observations()
    print(f"\n[INFO] Running {args_cli.play_steps} play steps with {num_envs} envs ...\n")

    for timestep in range(1, args_cli.play_steps + 1):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        # ---- Compute errors ----
        ee_state = asset.data.body_state_w[:, body_idx, :]
        ee_pos = ee_state[:, :3]
        ee_quat = ee_state[:, 3:7]
        ee_vel = ee_state[:, 7:10]
        ee_speed = torch.norm(ee_vel, dim=-1)

        cmd = base_env.command_manager.get_command("ee_pose")
        goal_pos = cmd[:, :3]
        goal_quat = cmd[:, 3:7]

        # Position error
        pos_err = torch.norm(ee_pos - goal_pos, dim=-1)

        # Orientation error — geodesic
        ori_err = quat_error_magnitude(ee_quat, goal_quat)

        # Orientation error — RPY decomposition
        rpy_err = quat_error_rpy(ee_quat, goal_quat)  # (N, 3), signed rad

        # Keypoint error
        _, kp_err = keypoint_distance(ee_pos, ee_quat, goal_pos, goal_quat, 0.3)

        # ---- Feed tracker ----
        tracker.step(pos_err, ori_err, rpy_err, kp_err, ee_speed)

        # ---- Periodic print ----
        if timestep % 200 == 0:
            settled_pct = tracker.ever_settled.float().mean().item() * 100
            arrived_pct = tracker.arrived.float().mean().item() * 100
            print(f"[step {timestep:5d}/{args_cli.play_steps}]  "
                  f"pos={pos_err.mean().item()*100:.1f}cm  "
                  f"ori={math.degrees(ori_err.mean().item()):.1f}°  "
                  f"kp={kp_err.mean().item()*100:.1f}cm  "
                  f"speed={ee_speed.mean().item():.3f}m/s  "
                  f"arrived={arrived_pct:.0f}%  settled={settled_pct:.0f}%")

        # Video early stop
        if args_cli.video and timestep >= args_cli.video_length:
            break

    # ---- Final report ----
    tracker.report(print_per_env=args_cli.print_per_env)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()