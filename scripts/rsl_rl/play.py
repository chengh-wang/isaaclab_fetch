"""Script to play a checkpoint of an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# Video arguments
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=300, help="Length of the recorded video (in steps).")
# Camera arguments
parser.add_argument(
    "--camera_offset",
    type=float,
    nargs=3,
    default=[2.5, 2.5, 2.0],
    help="Camera position offset from robot [x, y, z] (default: 2.5 2.5 2.0)",
)
parser.add_argument(
    "--camera_target_offset",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.5],
    help="Camera look-at offset from robot [x, y, z] (default: 0.0 0.0 0.5)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import fetch_project.tasks  # noqa: F401
from sim_logger import SimLogger

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_pickle


def _load_env_cfg_for_run(task_name: str, device: str, num_envs: int | None, agent_cfg) -> tuple[object, str]:
    """Load the environment config saved with the checkpoint run when available."""
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    env_cfg_path = os.path.join(os.path.dirname(resume_path), "params", "env.pkl")
    if os.path.exists(env_cfg_path):
        print(f"[INFO] Loading environment config from run: {env_cfg_path}")
        env_cfg = load_pickle(env_cfg_path)
        env_cfg.sim.device = device
        if num_envs is not None:
            env_cfg.scene.num_envs = num_envs
    else:
        print(f"[WARN] Saved environment config not found at: {env_cfg_path}")
        print("[WARN] Falling back to the current task configuration.")
        env_cfg = parse_env_cfg(task_name, num_envs=num_envs, device=device)

    return env_cfg, resume_path


def main():
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg, resume_path = _load_env_cfg_for_run(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, agent_cfg=agent_cfg
    )

    # ── Camera: track env 0 from a specific angle ──
    if args_cli.video:
        env_cfg.viewer.eye = tuple(args_cli.camera_offset)
        env_cfg.viewer.lookat = tuple(args_cli.camera_target_offset)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loaded model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(ppo_runner.alg.policy, export_model_dir, filename="policy.onnx")
    # reset environment
    obs = env.get_observations()
    # obs is a TensorDict keyed by observation group; extract 'policy' tensor for debug prints
    if hasattr(obs, "keys") and "policy" in obs.keys():
        obs_tensor = obs["policy"]
    else:
        obs_tensor = obs
    print("JOINT NAMES:", env.unwrapped.scene['robot'].joint_names)
    print("OBS SHAPE:", obs_tensor.shape)
  
    # ── ADD THIS BLOCK ──
    robot = env.unwrapped.scene['robot']
    print("\n=== Joint Order Debug ===")
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i}] {name}")
    
    # Check actual obs values
    print("\n=== Obs Breakdown ===")
    o = obs_tensor[0].cpu().numpy()
    joint_obs_dim = len(env_cfg.observations.policy.joint_pos.params["asset_cfg"].joint_names)
    kp_obs_dim = 9
    action_dim = sum(
        len(getattr(action_term, "joint_names", []) or [])
        if hasattr(action_term, "joint_names")
        else 2
        for action_term in env_cfg.actions.__dict__.values()
        if hasattr(action_term, "class_type")
    )
    vel_obs_dim = (len(o) - joint_obs_dim - kp_obs_dim - action_dim) // 2
    cursor = 0
    print(f"  joint_pos_rel ({joint_obs_dim}): {o[cursor:cursor + joint_obs_dim]}")
    cursor += joint_obs_dim
    print(f"  kp_command    ({kp_obs_dim}): {o[cursor:cursor + kp_obs_dim]}")
    cursor += kp_obs_dim
    print(f"  base_lin_vel  ({vel_obs_dim}): {o[cursor:cursor + vel_obs_dim]}")
    cursor += vel_obs_dim
    print(f"  base_ang_vel  ({vel_obs_dim}): {o[cursor:cursor + vel_obs_dim]}")
    cursor += vel_obs_dim
    print(f"  last_action  ({action_dim}): {o[cursor:cursor + action_dim]}")
    
    # Find resolved joint indices for joint_pos term
    joint_names_train = env_cfg.actions.arm_action.joint_names
    resolved_ids = robot.find_joints(joint_names_train)[0]
    print(f"\n=== Resolved joint IDs for obs ===")
    for name, idx in zip(joint_names_train, resolved_ids):
        print(f"  {name} -> isaac index {idx}")
    # ── END BLOCK ──
    logger = SimLogger(env, log_dir=os.path.join(log_dir, "play_logs"))
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            logger.log(env, obs, actions)
            # ── DEBUG: print first 5 steps ──
            # if timestep < 5:
            #     print(f"\n=== Step {timestep} ===")
            #     print(f"OBS shape: {obs.shape}")
            #     print(f"OBS [env0]: {obs[0].cpu().numpy()}")
            #     # Split obs into terms: joint_pos(8) + kp_command(9) + last_action(10)
            #     o = obs[0].cpu().numpy()
            #     print(f"  joint_pos_rel (8): {o[:8]}")
            #     print(f"  kp_command    (9): {o[8:17]}")
            #     print(f"  last_action  (10): {o[17:27]}")
            #     print(f"ACTION [env0]: {actions[0].cpu().numpy()}")
        timestep += 1

        # --- Debug: curriculum-relevant stats ---
        if timestep % 50 == 0:
            base_env = env.unwrapped
            asset = base_env.scene["robot"]
            body_idx = asset.find_bodies("wrist_roll_link")[0][0]
            ee_state = asset.data.body_state_w[:, body_idx, :]
            ee_pos, ee_quat = ee_state[:, :3], ee_state[:, 3:7]
            ee_vel = ee_state[:, 7:10]
            ee_speed = torch.norm(ee_vel, dim=-1)

            cmd = base_env.command_manager.get_command("ee_pose")
            g_pos, g_quat = cmd[:, :3], cmd[:, 3:7]

            from fetch_project.tasks.manipulation.reach.mdp.utils import keypoint_distance
            _, kp_dist = keypoint_distance(ee_pos, ee_quat, g_pos, g_quat, 0.3)

            close = (kp_dist < 0.15).float().mean().item()
            slow = (ee_speed < 0.05).float().mean().item()
            settled = ((kp_dist < 0.15) & (ee_speed < 0.05)).float().mean().item()

            print(f"\n[PLAY step={timestep}]")
            print(f"  kp_dist: mean={kp_dist.mean().item():.4f}m  median={kp_dist.median().item():.4f}m  "
                  f"min={kp_dist.min().item():.4f}m  max={kp_dist.max().item():.4f}m")
            print(f"  ee_speed: mean={ee_speed.mean().item():.4f}m/s  max={ee_speed.max().item():.4f}m/s")
            print(f"  close(<0.15m): {100*close:.0f}%  slow(<0.05m/s): {100*slow:.0f}%  settled: {100*settled:.0f}%")
            print(f"  Stage0 threshold=0.08m → {'✓ PASS' if kp_dist.median().item() < 0.08 else '✗ FAIL'} (median)")

        if args_cli.video:
            if timestep == args_cli.video_length:
                break

    logger.close()
    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
