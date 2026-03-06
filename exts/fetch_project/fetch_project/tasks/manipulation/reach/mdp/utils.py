"""Keypoint utilities for EE pose tracking.

Converts (pos, quat) into 3 keypoints on a virtual cube centered at EE.
Pure math — no Isaac Lab dependency, works in both sim and hardware.
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, quat_inv


def make_keypoint_offsets(cube_side: float, device: torch.device) -> torch.Tensor:
    """3 offsets along EE-local principal axes.

    Returns:
        (3, 3) tensor. Row i = offset along axis i.
    """
    h = cube_side / 2.0
    return torch.tensor(
        [[h, 0.0, 0.0],
         [0.0, h, 0.0],
         [0.0, 0.0, h]],
        device=device,
        dtype=torch.float32,
    )


def pose_to_keypoints(
    pos: torch.Tensor,
    quat: torch.Tensor,
    cube_side: float = 0.3,
) -> torch.Tensor:
    """Convert (pos, quat) batch to 3 keypoints in the same frame.

    Args:
        pos:  (N, 3)
        quat: (N, 4) Isaac convention (w, x, y, z)
        cube_side: virtual cube side length

    Returns:
        (N, 3, 3) — kps[i, k, :] is the k-th keypoint of env i.
    """
    offsets = make_keypoint_offsets(cube_side, pos.device)  # (3, 3)
    N = pos.shape[0]
    kps = torch.zeros(N, 3, 3, device=pos.device, dtype=pos.dtype)
    for k in range(3):
        offset_k = offsets[k].unsqueeze(0).expand(N, -1)   # (N, 3)
        kps[:, k, :] = pos + quat_apply(quat, offset_k)
    return kps


def keypoint_distance(
    pos_a: torch.Tensor,
    quat_a: torch.Tensor,
    pos_b: torch.Tensor,
    quat_b: torch.Tensor,
    cube_side: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-keypoint L2 distances and their mean.

    Returns:
        kp_dists: (N, 3) per-keypoint distances
        mean_dist: (N,)  mean across keypoints
    """
    kps_a = pose_to_keypoints(pos_a, quat_a, cube_side)
    kps_b = pose_to_keypoints(pos_b, quat_b, cube_side)
    kp_dists = torch.norm(kps_a - kps_b, dim=-1)  # (N, 3)
    return kp_dists, kp_dists.mean(dim=-1)


def keypoint_delta_in_frame(
    curr_pos: torch.Tensor,
    curr_quat: torch.Tensor,
    goal_pos: torch.Tensor,
    goal_quat: torch.Tensor,
    frame_quat: torch.Tensor,
    cube_side: float = 0.3,
) -> torch.Tensor:
    """Compute (goal_kps - curr_kps) expressed in a given reference frame.

    Args:
        curr_pos, curr_quat: current EE pose (N, 3), (N, 4)
        goal_pos, goal_quat: target EE pose  (N, 3), (N, 4)
        frame_quat: quaternion of the reference frame (N, 4), e.g. base quat
        cube_side: virtual cube side length

    Returns:
        (N, 9) — flattened keypoint deltas in the reference frame.
    """
    curr_kps = pose_to_keypoints(curr_pos, curr_quat, cube_side)  # (N,3,3)
    goal_kps = pose_to_keypoints(goal_pos, goal_quat, cube_side)  # (N,3,3)
    delta_w = goal_kps - curr_kps  # (N, 3, 3) in world frame

    frame_quat_inv = quat_inv(frame_quat)  # (N, 4)
    delta_f = torch.zeros_like(delta_w)
    for k in range(3):
        delta_f[:, k, :] = quat_apply(frame_quat_inv, delta_w[:, k, :])

    return delta_f.reshape(-1, 9)
