"""Quick script to print joint order, action mapping, and reward-critical index checks for Fetch reach env.

Usage:
    python debug_fetch_reach.py --task <your_fetch_reach_task_id> --num_envs 1 --headless

Adapted from G1 debug script for Fetch mobile manipulator.
"""

from isaaclab.app import AppLauncher
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Fetch-Reach-Simple-v0",
                    help="Gym task ID for your Fetch reach env")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np

# === CHANGE THIS IMPORT to wherever your Fetch task is registered ===
import fetch_project.tasks  # noqa: F401  — registers gym envs

from isaaclab_tasks.utils.hydra import hydra_task_config


# ─── Joint name lists (must match your env config) ───
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
FETCH_BASE_JOINTS = ["l_wheel_joint", "r_wheel_joint"]


def _safe_list(x):
    """Convert _joint_ids to a plain list regardless of type."""
    if isinstance(x, slice):
        return None  # means ALL
    if isinstance(x, torch.Tensor):
        return x.cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    uw = env.unwrapped
    robot = uw.scene["robot"]

    # ═══════════════════════════════════════════════════════════════════
    # 1. ALL JOINTS in articulation (URDF order)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("1. URDF JOINT ORDER  (robot.joint_names)")
    print("=" * 90)
    for i, name in enumerate(robot.joint_names):
        tag = ""
        if name in FETCH_ARM_JOINTS:
            tag = "  [ARM]"
        elif name in FETCH_BASE_JOINTS:
            tag = "  [BASE/WHEEL]"
        print(f"  URDF[{i:2d}]  {name:40s}{tag}")
    print(f"  Total: {len(robot.joint_names)} joints")

    # ═══════════════════════════════════════════════════════════════════
    # 2. find_joints() verification — the key thing to check
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("2. find_joints() RESULTS  (what reward functions actually use)")
    print("=" * 90)

    arm_ids, arm_names = robot.find_joints(FETCH_ARM_JOINTS)
    base_ids, base_names = robot.find_joints(FETCH_BASE_JOINTS)

    print(f"\n  Arm joints requested:  {FETCH_ARM_JOINTS}")
    print(f"  Arm joints found ids:  {arm_ids}")
    print(f"  Arm joints found names:{arm_names}")
    for i, (jid, jname) in enumerate(zip(arm_ids, arm_names)):
        expected = FETCH_ARM_JOINTS[i] if i < len(FETCH_ARM_JOINTS) else "?"
        match = "✓" if jname == expected else f"✗ expected '{expected}'"
        print(f"    [{i}] joint_id={jid:2d}  name={jname:40s}  {match}")

    print(f"\n  Base joints requested:  {FETCH_BASE_JOINTS}")
    print(f"  Base joints found ids:  {base_ids}")
    print(f"  Base joints found names:{base_names}")
    for i, (jid, jname) in enumerate(zip(base_ids, base_names)):
        expected = FETCH_BASE_JOINTS[i] if i < len(FETCH_BASE_JOINTS) else "?"
        match = "✓" if jname == expected else f"✗ expected '{expected}'"
        print(f"    [{i}] joint_id={jid:2d}  name={jname:40s}  {match}")

    # Unpenalized joints
    all_ids_set = set(range(len(robot.joint_names)))
    penalized = set(arm_ids) | set(base_ids)
    unpenalized = sorted(all_ids_set - penalized)
    print(f"\n  *** UNPENALIZED joints ({len(unpenalized)}):")
    for uid in unpenalized:
        print(f"    URDF[{uid:2d}]  {robot.joint_names[uid]}")

    # ═══════════════════════════════════════════════════════════════════
    # 3. ACTION MANAGER — term-by-term breakdown
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("3. ACTION MANAGER")
    print("=" * 90)

    action_offset = 0
    for term_idx, (term_name, term) in enumerate(uw.action_manager._terms.items()):
        print(f"\n  ── Term {term_idx}: '{term_name}' ({term.__class__.__name__}) ──")

        # action_dim
        dim = term.action_dim
        print(f"  action_dim = {dim}")
        print(f"  action tensor indices: [{action_offset} : {action_offset + dim}]")

        # _joint_ids
        if hasattr(term, '_joint_ids'):
            ids_list = _safe_list(term._joint_ids)
            if ids_list is None:
                print(f"  _joint_ids = slice(None) → ALL joints")
            else:
                print(f"  _joint_ids = {ids_list}")
        else:
            print(f"  (no _joint_ids attribute — might be DifferentialDrive)")

        # _joint_names
        if hasattr(term, '_joint_names'):
            print(f"  _joint_names ({len(term._joint_names)}):")
            ids_list = _safe_list(getattr(term, '_joint_ids', [])) or list(range(len(term._joint_names)))
            for act_i, jname in enumerate(term._joint_names):
                urdf_i = ids_list[act_i] if act_i < len(ids_list) else "?"
                print(f"    action[{action_offset + act_i:2d}] → URDF[{urdf_i:>2}]  {jname}")
        else:
            print(f"  (no _joint_names — DifferentialDrive uses wheel joints internally)")

        # scale / offset
        for attr in ['_scale', '_offset']:
            if hasattr(term, attr):
                val = getattr(term, attr)
                if isinstance(val, torch.Tensor) and val.numel() > 1:
                    v = val[0].cpu().numpy() if val.ndim > 1 else val.cpu().numpy()
                    print(f"  {attr}:")
                    for j, s in enumerate(v):
                        print(f"    [{j}] {s:.6f}")
                else:
                    print(f"  {attr} = {val}")

        # DifferentialDrive specific attrs
        for attr in ['_wheel_radius', '_wheel_separation',
                     '_max_linear_velocity', '_max_angular_velocity',
                     '_left_wheel_joint_id', '_right_wheel_joint_id',
                     'left_wheel_joint_name', 'right_wheel_joint_name']:
            if hasattr(term, attr):
                print(f"  {attr} = {getattr(term, attr)}")

        action_offset += dim

    print(f"\n  Total action dim: {action_offset}")
    print(f"  action_manager.action.shape = {uw.action_manager.action.shape}")

    # ═══════════════════════════════════════════════════════════════════
    # 4. CRITICAL ALIGNMENT CHECK — action slicing vs joint indices
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("4. CRITICAL ALIGNMENT CHECK for action_rate_weighted / joint_vel_weighted")
    print("=" * 90)

    num_arm = len(arm_ids)
    num_base = len(base_ids)
    total_action = uw.action_manager.action.shape[-1]

    print(f"\n  action_rate_weighted slicing:")
    print(f"    arm_delta  = action[:, 0:{num_arm}]     (len={num_arm})")
    print(f"    base_delta = action[:, {num_arm}:{num_arm + num_base}]  (len={num_base})")
    print(f"    total action dim = {total_action}")

    if num_arm + num_base != total_action:
        print(f"    *** WARNING: arm({num_arm}) + base({num_base}) = {num_arm + num_base} != total({total_action}) ***")
        print(f"    *** {total_action - num_arm - num_base} action dims are NOT penalized! ***")
    else:
        print(f"    ✓ Covers all {total_action} action dims")

    # Check: does first action term correspond to arm?
    terms_list = list(uw.action_manager._terms.items())
    if len(terms_list) >= 1:
        first_name, first_term = terms_list[0]
        print(f"\n    First action term: '{first_name}' (dim={first_term.action_dim})")
        if first_term.action_dim == num_arm:
            print(f"    ✓ First term dim matches arm joint count")
        else:
            print(f"    *** MISMATCH: first term dim={first_term.action_dim} but num_arm={num_arm} ***")

    if len(terms_list) >= 2:
        second_name, second_term = terms_list[1]
        print(f"    Second action term: '{second_name}' (dim={second_term.action_dim})")
        if second_term.action_dim == num_base:
            print(f"    ✓ Second term dim matches base joint count")
        else:
            expected = 2  # DifferentialDrive: [lin_vel, ang_vel]
            if second_term.action_dim == expected:
                print(f"    ✓ DifferentialDrive: action dim={expected} ([lin_vel, ang_vel])")
                print(f"    ⚠ Note: base_delta in action_rate_weighted penalizes")
                print(f"      delta([lin_vel, ang_vel]), NOT delta([left_wheel, right_wheel])")
            else:
                print(f"    *** MISMATCH: second term dim={second_term.action_dim} ***")

    print(f"\n  joint_vel_weighted indexing:")
    print(f"    arm_vel  = joint_vel[:, {list(arm_ids)}]")
    print(f"    base_vel = joint_vel[:, {list(base_ids)}]")
    print(f"    joint_vel shape = {robot.data.joint_vel.shape}")

    # ═══════════════════════════════════════════════════════════════════
    # 5. BODY ORDER (useful for EE body_ids check)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("5. BODY ORDER  (for verifying EE body_ids)")
    print("=" * 90)
    for i, name in enumerate(robot.body_names):
        tag = "  ← EE" if name == "wrist_roll_link" else ""
        print(f"  [{i:2d}] {name}{tag}")

    # Verify SceneEntityCfg body_ids
    from isaaclab.managers import SceneEntityCfg
    ee_cfg = SceneEntityCfg("robot", body_names=["wrist_roll_link"])
    ee_cfg.resolve(uw.scene)
    print(f"\n  SceneEntityCfg('robot', body_names=['wrist_roll_link']).body_ids = {ee_cfg.body_ids}")

    # ═══════════════════════════════════════════════════════════════════
    # 6. DEFAULT JOINT POSITIONS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("6. DEFAULT JOINT POSITIONS & LIMITS")
    print("=" * 90)
    default_pos = robot.data.default_joint_pos[0].cpu().numpy()
    soft_lo = robot.data.soft_joint_pos_limits[0, :, 0].cpu().numpy()
    soft_hi = robot.data.soft_joint_pos_limits[0, :, 1].cpu().numpy()
    for i, name in enumerate(robot.joint_names):
        print(f"  URDF[{i:2d}]  {name:40s}  default={default_pos[i]:8.4f}  limits=[{soft_lo[i]:8.4f}, {soft_hi[i]:8.4f}]")

    # ═══════════════════════════════════════════════════════════════════
    # 7. LIVE VALUES (step once to get non-zero data)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("7. LIVE VALUES AFTER RESET (env 0)")
    print("=" * 90)
    obs, _ = env.reset()
    # Take a few zero-action steps to get meaningful velocities
    zero_action = torch.zeros(1, total_action, device=uw.device)
    for _ in range(10):
        obs, _, _, _, _ = env.step(zero_action)

    jpos = robot.data.joint_pos[0].cpu().numpy()
    jvel = robot.data.joint_vel[0].cpu().numpy()
    action = uw.action_manager.action[0].cpu().numpy()

    print(f"\n  Joint positions & velocities:")
    for i, name in enumerate(robot.joint_names):
        tag = ""
        if name in FETCH_ARM_JOINTS:
            tag = "[ARM]"
        elif name in FETCH_BASE_JOINTS:
            tag = "[BASE]"
        print(f"  URDF[{i:2d}] {name:40s} pos={jpos[i]:8.4f}  vel={jvel[i]:8.4f}  {tag}")

    print(f"\n  Action values:")
    action_offset = 0
    for term_name, term in uw.action_manager._terms.items():
        dim = term.action_dim
        vals = action[action_offset:action_offset + dim]
        print(f"    '{term_name}': {vals}")
        action_offset += dim

    # ═══════════════════════════════════════════════════════════════════
    # 8. REWARD FUNCTION SMOKE TEST
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("8. REWARD FUNCTION SMOKE TEST (one step)")
    print("=" * 90)
    try:
        # Take one more step with random action
        rand_action = torch.randn(1, total_action, device=uw.device) * 0.1
        obs, rew, terminated, truncated, info = env.step(rand_action)
        
        # Print individual reward terms if available
        if hasattr(uw, 'reward_manager'):
            for term_name, term in uw.reward_manager._terms.items():
                # Compute reward
                try:
                    val = term.func(uw, **term.params)
                    weighted = val * term.weight
                    print(f"  {term_name:30s}  raw={val[0].item():10.4f}  weight={term.weight:8.4f}  "
                          f"weighted={weighted[0].item():10.4f}")
                except Exception as e:
                    print(f"  {term_name:30s}  ERROR: {e}")
        
        print(f"\n  Total reward: {rew[0].item():.4f}")
    except Exception as e:
        print(f"  Reward smoke test failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # 9. COMMAND MANAGER
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("9. COMMAND MANAGER")
    print("=" * 90)
    for cmd_name, cmd_term in uw.command_manager._terms.items():
        cmd = uw.command_manager.get_command(cmd_name)
        print(f"  '{cmd_name}': shape={cmd.shape}, type={cmd_term.__class__.__name__}")
        print(f"    values[0]: {cmd[0].cpu().tolist()}")
        if cmd.shape[-1] >= 7:
            print(f"    pos=[{cmd[0, :3].cpu().tolist()}]")
            print(f"    quat=[{cmd[0, 3:7].cpu().tolist()}]")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
