import pathlib
import numpy as np
import torch
import nvdiffrast.torch as dr
import json

import util
from util import normalize
from configs import MainConfig
from mesh import Mesh


class Render:
    def __init__(self, config: MainConfig):
        self.config = config
        self.glctx = dr.RasterizeCudaContext()
        self.mesh = Mesh(config)
        self.pos = self.mesh.pos
        self.pos_idx = self.mesh.pos_idx
        pos_scaled = self.pos[..., :3]
        pos_scaled = (pos_scaled - pos_scaled.min()) / (pos_scaled.max() - pos_scaled.min())
        self.vtx_attr = torch.cat([pos_scaled, self.mesh.normals, self.mesh.uvs], axis=-1)[None, ...]
        # smooth display

        self.smooth_angle = 0.0
        workdir = f"{pathlib.Path(__file__).absolute().parents[0]}"
        ans_path = f"{workdir}/{config.file_paths.data_dir}/{config.file_paths.data}/train/ans.json"
        with open(ans_path, "r") as f:
            self.ans = json.load(f)
        self.smooth_light_pos = np.asarray(self.ans["light_pos"], np.float32)
        self.smooth_emission = np.asarray(self.ans["emission"], np.float32)
        self.smooth_ambient = np.asarray(self.ans["ambient"], np.float32)
        self.smooth_proj = np.asarray(self.ans["proj"], np.float32)
        self.smooth_emission = torch.as_tensor(self.smooth_emission, dtype=torch.float32, device="cuda")[None, ...]
        self.smooth_ambient = torch.as_tensor(self.smooth_ambient, dtype=torch.float32, device="cuda")[None, ...]

    def render(self, camera_pos, light_pos, mvp, emission, ambient, resolution, model) -> torch.Tensor:
        pos_clip = torch.matmul(self.pos[None, ...], mvp.transpose(1, 2))
        rast_out, rast_out_db = dr.rasterize(self.glctx, pos_clip, self.pos_idx, resolution)
        mask = rast_out[..., -1:] == 0  # Mask out background.

        attr_out, attr_out_db = dr.interpolate(self.vtx_attr, rast_out, self.pos_idx)

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
        intensity = emission[:, None, None, :] / light_dist2  # Light intensity.

        color = ambient[:, None, None, :] + intensity * bsdf
        color = torch.where(mask, torch.zeros_like(color), color)  # Black background.
        color = torch.clamp(color, 0, 1)
        return color

    def display_render(self, resolution, model) -> torch.Tensor:
        self.smooth_angle += 0.1
        is_light_moving = self.config.file_paths.data == "move_light"
        if is_light_moving:
            rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(0.8 * np.pi))
            rot_light = util.rotate_y(-self.smooth_angle)
        else:
            rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(self.smooth_angle))
            rot_light = util.rotate_y(0)
        mv = np.matmul(util.translate(0, 0, -3.5), rot)
        mvp = np.matmul(self.smooth_proj, mv).astype(np.float32)
        mvp = torch.as_tensor(mvp, dtype=torch.float32, device="cuda")[None, ...]
        camera_pos = torch.as_tensor(np.linalg.inv(mv)[:3, 3], dtype=torch.float32, device="cuda")[None, ...]
        light_pos = torch.as_tensor(np.matmul(rot_light, self.smooth_light_pos)[..., :3], dtype=torch.float32, device="cuda")[
            None, ...
        ]
        return self.render(camera_pos, light_pos, mvp, self.smooth_emission, self.smooth_ambient, resolution, model)


# def render_rna(
#     glctx, pos, pos_idx, normals, uvs, camera_pos, light_pos, mvp, emission, ambient, resolution, model
# ) -> torch.Tensor:
#     pos_clip = torch.matmul(pos[None, ...], mvp.transpose(1, 2))
#     rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution)
#     mask = rast_out[..., -1:] == 0  # Mask out background.

#     pos_scaled = pos[..., :3]
#     pos_scaled = (pos_scaled - pos_scaled.min()) / (pos_scaled.max() - pos_scaled.min())
#     vtx_attr = torch.cat([pos_scaled, normals, uvs], axis=-1)[None, ...]
#     attr_out, attr_out_db = dr.interpolate(vtx_attr, rast_out, pos_idx)

#     pos_ip = attr_out[..., :3]
#     normals_ip = normalize(attr_out[..., 3:6])
#     uvs_ip = attr_out[..., 6:8]
#     view_dir = normalize(camera_pos[:, None, None, :] - pos_ip)
#     light_dir = normalize(light_pos[:, None, None, :] - pos_ip)

#     input = torch.cat([pos_ip, normals_ip, view_dir, light_dir, uvs_ip], axis=-1)
#     indices = torch.nonzero(~mask.squeeze(-1), as_tuple=True)
#     min_row_idx = torch.min(indices[1]).item()
#     max_row_idx = torch.max(indices[1]).item()
#     min_col_idx = torch.min(indices[2]).item()
#     max_col_idx = torch.max(indices[2]).item()
#     input = input[:, min_row_idx : max_row_idx + 1, min_col_idx : max_col_idx + 1, :]
#     output = model(input)
#     bsdf_shape = list(mask.shape[:-1]) + [output.shape[-1]]
#     bsdf = torch.zeros(bsdf_shape, device=input.device)
#     bsdf[:, min_row_idx : max_row_idx + 1, min_col_idx : max_col_idx + 1, :] = output

#     light_dist2 = torch.sum((light_pos[:, None, None, :] - pos_ip) ** 2, -1, keepdim=True)
#     # ldotn = torch.sum(light_dir * normals_ip, -1, keepdim=True)  # L dot N.
#     # ldotn = torch.max(torch.zeros_like(ldotn), ldotn)
#     intensity = emission[:, None, None, :] / light_dist2  # Light intensity.

#     color = ambient[:, None, None, :] + intensity * bsdf
#     color = torch.where(mask, torch.zeros_like(color), color)  # Black background.
#     color = torch.clamp(color, 0, 1)
#     return color


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
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution)
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
    intensity = emission[:, None, None, :] / light_dist2  # Light intensity.

    phong_kd = dr.texture(tex[None, ...], uvs_ip, filter_mode="linear")  # Diffuse texture.
    color = ambient[:, None, None, :] + intensity * (phong_kd * ldotn + phong_ks * (hdotn**phong_exp))
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
