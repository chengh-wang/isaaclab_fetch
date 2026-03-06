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
    zero_friction_meshes = [
        "base_link/collisions/base_link_collision",
        "base_link/visuals/r_wheel_link/mesh",
        "base_link/visuals/r_wheel_link_0/mesh",
        "base_link/visuals/r_wheel_link_1/mesh",
        "base_link/visuals/r_wheel_link_2/mesh",
    ]

    for env_id in ids:
        for mesh in zero_friction_meshes:
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot/{mesh}")
            if prim.IsValid():
                binding = UsdShade.MaterialBindingAPI.Apply(prim)
                binding.Bind(mtl, UsdShade.Tokens.weakerThanDescendants, "physics")
        robot_path = f"/World/envs/env_{env_id}/Robot"
        robot_prim = stage.GetPrimAtPath(robot_path)
