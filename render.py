import numpy as np
import torch
import nvdiffrast.torch as dr

from util import normalize


def render_rna(
    glctx, pos, pos_idx, normals, uvs, camera_pos, light_pos, mvp, emission, ambient, resolution, model
) -> torch.Tensor:
    pos_clip = torch.matmul(pos[None, ...], mvp.transpose(1, 2))
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    mask = rast_out[..., -1:] == 0  # Mask out background.

    pos_scaled = pos[..., :3]
    pos_scaled = (pos_scaled - pos_scaled.min()) / (pos_scaled.max() - pos_scaled.min())
    vtx_attr = torch.cat([pos_scaled, normals, uvs], axis=-1)[None, ...]
    attr_out, attr_out_db = dr.interpolate(vtx_attr, rast_out, pos_idx)

    pos_ip = attr_out[..., :3]
    normals_ip = normalize(attr_out[..., 3:6])
    uvs_ip = attr_out[..., 6:8]
    view_dir = normalize(camera_pos[:, None, None, :] - pos_ip)
    light_dir = normalize(light_pos[:, None, None, :] - pos_ip)

    input = torch.cat([pos_ip, normals_ip, view_dir, light_dir, uvs_ip], axis=-1)
    indices = torch.nonzero(~mask.squeeze(-1), as_tuple=True)
    min_row_idx = torch.min(indices[1]).item()
    max_row_idx = torch.max(indices[1]).item()
    min_col_idx = torch.min(indices[2]).item()
    max_col_idx = torch.max(indices[2]).item()
    input = input[:, min_row_idx : max_row_idx + 1, min_col_idx : max_col_idx + 1, :]
    output = model(input)
    bsdf_shape = list(mask.shape[:-1]) + [output.shape[-1]]
    bsdf = torch.zeros(bsdf_shape, device=input.device)
    bsdf[:, min_row_idx : max_row_idx + 1, min_col_idx : max_col_idx + 1, :] = output

    light_dist2 = torch.sum((light_pos[:, None, None, :] - pos_ip) ** 2, -1, keepdim=True)
    # ldotn = torch.sum(light_dir * normals_ip, -1, keepdim=True)  # L dot N.
    # ldotn = torch.max(torch.zeros_like(ldotn), ldotn)
    intensity = emission / light_dist2  # Light intensity.

    color = ambient + intensity * bsdf
    color = torch.where(mask, torch.zeros_like(color), color)  # Black background.
    color = torch.clamp(color, 0, 1)
    return color


def render_refl(
    glctx,
    pos,
    pos_idx,
    normals,
    uvs,
    camera_pos,
    light_pos,
    mvp,
    emission,
    ambient,
    tex,
    phong_ks,
    phong_exp,
    resolution,
) -> torch.Tensor:
    # Transform and rasterize.
    pos_clip = torch.matmul(pos[None, ...], mvp.transpose(1, 2))
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, [resolution, resolution])
    mask = rast_out[..., -1:] == 0  # Mask out background.

    vtx_attr = torch.cat([pos[..., :3], normals], axis=-1)[None, ...]
    attr_out, attr_out_db = dr.interpolate(vtx_attr, rast_out, pos_idx)  # Interpolated attributes.
    uvs_ip, uvs_ip_db = dr.interpolate(uvs[None, ...], rast_out, pos_idx)  # Interpolated UVs.

    # Phong light.
    pos_ip = attr_out[..., :3]
    normals_ip = normalize(attr_out[..., 3:6])

    view_dir = normalize(camera_pos[:, None, None, :] - pos_ip)
    light_dir = normalize(light_pos[:, None, None, :] - pos_ip)
    light_dist2 = torch.sum((light_pos[:, None, None, :] - pos_ip) ** 2, -1, keepdim=True)
    half_dir = normalize(view_dir + light_dir)
    hdotn = torch.sum(half_dir * normals_ip, -1, keepdim=True)  # H dot N.
    hdotn = torch.max(torch.zeros_like(hdotn), hdotn)
    ldotn = torch.sum(light_dir * normals_ip, -1, keepdim=True)  # L dot N.
    ldotn = torch.max(torch.zeros_like(ldotn), ldotn)
    intensity = emission / light_dist2  # Light intensity.

    phong_kd = dr.texture(tex[None, ...], uvs_ip, filter_mode="linear")  # Diffuse texture.
    color = ambient + intensity * (phong_kd * ldotn + phong_ks * (hdotn**phong_exp))
    color = torch.where(mask, torch.zeros_like(color), color)  # Black background.
    color = torch.clamp(color, 0, 1)
    return color


# def render_rna_(
#     glctx, pos, pos_idx, normals, uvs, camera_pos, light_pos, mvp, emission, ambient, resolution, model
# ) -> torch.Tensor:
#     pos_clip = torch.cat([pos, torch.ones_like(pos[..., :1])], axis=1)
#     pos_clip = torch.matmul(pos_clip, mvp.t())[None, ...]
#     rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])

#     vtx_view_dir = normalize(camera_pos[None, ...] - pos)
#     vtx_light_dir = normalize(light_pos[None, ...] - pos)
#     pos_scaled = (pos - pos.min()) / (pos.max() - pos.min())
#     input = torch.cat([pos_scaled, normals, vtx_view_dir, vtx_light_dir, uvs], axis=-1)
#     vtx_bsdf = model(input)

#     vtx_attr = torch.cat([pos, normals, vtx_bsdf], axis=1)[None, ...]
#     attr_out, attr_out_db = dr.interpolate(vtx_attr, rast_out, pos_idx)

#     pos_ip = attr_out[..., :3]
#     normals_ip = normalize(attr_out[..., 3:6])
#     bsdf = attr_out[..., 6:9]

#     view_dir = normalize(camera_pos - pos_ip)
#     light_dir = normalize(light_pos - pos_ip)
#     light_dist2 = torch.sum((light_pos - pos_ip) ** 2, -1, keepdim=True)

#     ldotn = torch.sum(light_dir * normals_ip, -1, keepdim=True)  # L dot N.
#     ldotn = torch.max(torch.zeros_like(ldotn), ldotn)
#     intensity = emission / light_dist2  # Light intensity.

#     color = ambient + intensity * bsdf
#     mask = rast_out[..., -1:] == 0  # Mask out background.
#     color = torch.where(mask, torch.zeros_like(color), color)  # White background.
#     color = torch.clamp(color, 0, 1)
#     return color
