# Fetch Robot Isaac Lab Project

RL training environments for the Fetch mobile manipulator in Isaac Lab. Tasks include reaching, lifting, and drawer opening.

## Prerequisites

- Isaac Lab installed and working (see [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html))
- The Isaac Lab Python environment activated (e.g. `conda activate isaaclab` or whichever env you use)

## Installation

From the repo root, install the extension package in editable mode:

```bash
cd exts/fetch_project
pip install -e .
```

This installs the `fetch_project` package which registers all Fetch gym environments.

## Training

All training scripts are under `scripts/rsl_rl/`. Run from the **repo root**.

### Quick Start (Keypoint Reach)

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Reach-Fetch-Keypoint-v0 \
  --headless \
  --num_envs 1024 \
  --logger wandb \
  --log_project_name fetch_reach_keypoint \
  --enable_cameras \
  --video \
  --video_length 1000 \
  --video_interval 5000
```

### Minimal (no logging, no video)

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Reach-Fetch-Keypoint-v0 \
  --headless \
  --num_envs 1024
```

### Key Arguments

| Argument | Description |
|---|---|
| `--task` | Gym environment ID (see table below) |
| `--num_envs` | Number of parallel environments |
| `--headless` | Run without GUI |
| `--logger wandb` | Use Weights & Biases for logging (also supports `tensorboard`, `neptune`) |
| `--log_project_name` | W&B / Neptune project name |
| `--enable_cameras` | Required when using `--video` |
| `--video` | Record training videos |
| `--video_length` | Video length in env steps (default: 200) |
| `--video_interval` | Steps between recordings (default: 2000) |
| `--seed` | RNG seed |
| `--resume True` | Resume from last checkpoint |
| `--load_run` | Specific run folder name to resume from |
| `--checkpoint` | Specific checkpoint file to resume from |
| `--experiment_name` | Override experiment folder name (default: `reach_fetch`) |

### Resume Training

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Reach-Fetch-Keypoint-v0 \
  --headless \
  --num_envs 1024 \
  --resume True \
  --load_run <run_folder_name>
```

Logs are written to `logs/rsl_rl/<experiment_name>/`.

## Playback / Evaluation

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Reach-Fetch-Keypoint-Play-v0 \
  --num_envs 1
```

To record a video during playback:

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Reach-Fetch-Keypoint-Play-v0 \
  --num_envs 1 \
  --video \
  --video_length 300
```

By default, `play.py` loads the latest checkpoint from the latest run under `logs/rsl_rl/reach_fetch/`. Use `--load_run` and `--checkpoint` to specify a different run or checkpoint.

## Available Environments

### Reach

| Task ID | Description |
|---|---|
| `Isaac-Reach-Fetch-v0` | Joint-position reach (original) |
| `Isaac-Reach-Fetch-Play-v0` | Playback variant of above |
| `Isaac-Reach-Fetch-Simple-v0` | Simplified reach |
| `Isaac-Reach-Fetch-Simple-Play-v0` | Playback variant of above |
| `Isaac-Reach-Fetch-Keypoint-v0` | Keypoint-based reach |
| `Isaac-Reach-Fetch-Keypoint-Play-v0` | Playback variant of above |

### Lift

| Task ID | Description |
|---|---|
| `Isaac-Lift-Cube-Fetch-v0` | Cube lifting |
| `Isaac-Lift-Cube-Fetch-Play-v0` | Playback variant |

### Cabinet (Drawer Opening)

| Task ID | Description |
|---|---|
| `Isaac-Open-Drawer-Fetch-v0` | Drawer opening |
| `Isaac-Open-Drawer-Fetch-Play-v0` | Playback variant |
| `Isaac-Open-Drawer-Fetch-v1` | Drawer opening (point cloud obs) |
| `Isaac-Open-Drawer-Fetch-Play-v1` | Playback variant |

## Project Structure

```
isaaclab_fetch/
  assets/robots/          -- Fetch URDF/USD model files
  exts/fetch_project/     -- Installable extension package (pip install -e .)
    fetch_project/
      robots/             -- Fetch robot ArticulationCfg (fetch.py)
      tasks/
        manipulation/
          reach/          -- Reach task env configs, MDP, rewards
          lift/           -- Lift task env configs, MDP, rewards
          cabinet/        -- Cabinet task env configs
          cabinet_pc/     -- Cabinet task with point cloud observations
  scripts/
    rsl_rl/
      train.py            -- RSL-RL training script
      play.py             -- RSL-RL playback / evaluation script
      cli_args.py         -- CLI argument helpers
    skrl/
      train.py            -- SKRL training script (alternative)
      play.py             -- SKRL playback script (alternative)
  logs/                   -- Training logs and checkpoints
```

## PPO Hyperparameters

Default PPO config is in `exts/fetch_project/fetch_project/tasks/manipulation/reach/config/fetch/agents/rsl_rl_ppo_cfg.py`:

- Max iterations: 30000
- Steps per env: 24
- Save interval: 500
- Actor/Critic: [64, 64] MLP with ELU activation
- Learning rate: 1e-3 (adaptive schedule)
- Gamma: 0.99, Lambda: 0.95
- Entropy coef: 0.01, Clip param: 0.2

## License

MIT
