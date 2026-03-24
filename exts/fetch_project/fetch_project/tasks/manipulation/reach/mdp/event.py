import torch
import omni.usd
from pxr import UsdPhysics, UsdShade, Sdf, PhysxSchema
from isaaclab.managers import SceneEntityCfg

def set_caster_friction(
    env,
    env_ids: torch.Tensor | None,
    static_friction: float = 0.0,
    dynamic_friction: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set zero friction on base_link collision and caster wheel meshes."""
    stage = omni.usd.get_context().get_stage()

    # Create zero-friction material
    mtl_path = "/World/Materials/CasterMtl"
    if not stage.GetPrimAtPath(mtl_path).IsValid():
        mtl_prim = stage.DefinePrim(mtl_path, "Material")
        mat = UsdPhysics.MaterialAPI.Apply(mtl_prim)
        mat.CreateStaticFrictionAttr(static_friction)
        mat.CreateDynamicFrictionAttr(dynamic_friction)
        mat.CreateRestitutionAttr(0.0)

    mtl = UsdShade.Material(stage.GetPrimAtPath(mtl_path))

    num_envs = env.num_envs if hasattr(env, "num_envs") else env.unwrapped.num_envs
    ids = range(num_envs) if env_ids is None else env_ids.tolist()

    # All meshes that need zero friction: casters + base_link collision
    # zero_friction_meshes = [
    #     "base_link/collisions/base_link_collision",
    #     "base_link/visuals/r_wheel_link/mesh",
    #     "base_link/visuals/r_wheel_link_0/mesh",
    #     "base_link/visuals/r_wheel_link_1/mesh",
    #     "base_link/visuals/r_wheel_link_2/mesh",
    # ]
    zero_friction_meshes = [
        # ── base body collision (主碰撞体！) ──
        "base_link/visuals/base_link/mesh",
        # ── caster wheels (4个万向轮) ──
        "base_link/visuals/r_wheel_link/mesh",
        "base_link/visuals/r_wheel_link_0/mesh",
        "base_link/visuals/r_wheel_link_1/mesh",
        "base_link/visuals/r_wheel_link_2/mesh",
        # ── base 底部其他碰撞体 ──
        "base_link/visuals/torso_fixed_link/mesh",
        "base_link/collisions/estop_link/mesh",
        "base_link/collisions/laser_link/mesh",
    ]
    for env_id in ids:
        for mesh in zero_friction_meshes:
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot/{mesh}")
            if prim.IsValid():
                binding = UsdShade.MaterialBindingAPI.Apply(prim)
                binding.Bind(mtl, UsdShade.Tokens.weakerThanDescendants, "physics")
                if env_id == 0:
                    print(f"[CASTER] ✅ Set zero friction: {mesh}")
            else:
                if env_id == 0:
                    print(f"[CASTER] ❌ NOT FOUND: {mesh}")
        robot_path = f"/World/envs/env_{env_id}/Robot"
        robot_prim = stage.GetPrimAtPath(robot_path)
        
def reset_selected_joints_by_offset(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position_range: tuple[float, float] = (-0.1, 0.1),
    velocity_range: tuple[float, float] = (0.0, 0.0),
):
    asset = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)

    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        raise ValueError("asset_cfg.joint_ids is None. Please provide joint_names in SceneEntityCfg.")

    # default values for selected envs, all joints
    default_joint_pos = asset.data.default_joint_pos[env_ids].clone()
    default_joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # random offsets only for selected joints
    pos_offset = torch.empty((len(env_ids), len(joint_ids)), device=asset.device).uniform_(
        position_range[0], position_range[1]
    )
    vel_offset = torch.empty((len(env_ids), len(joint_ids)), device=asset.device).uniform_(
        velocity_range[0], velocity_range[1]
    )

    default_joint_pos[:, joint_ids] += pos_offset
    default_joint_vel[:, joint_ids] += vel_offset

    # optional: clamp to joint limits
    if hasattr(asset.data, "soft_joint_pos_limits") and asset.data.soft_joint_pos_limits is not None:
        lower = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids, 0]
        upper = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids, 1]
        default_joint_pos[:, joint_ids] = torch.clamp(default_joint_pos[:, joint_ids], lower, upper)

    asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)


