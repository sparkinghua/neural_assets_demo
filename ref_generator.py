import torch
import nvdiffrast.torch as dr
from PIL import Image
import numpy as np
import trimesh
import pathlib
import imageio
from tqdm import tqdm

import util
from render import render_point_refl


def ref_generator(radius=3.5, samples=63, x=0.4, n=1.0, f=200.0, resolution=[512, 512], fps=10):
    light_pos = np.asarray([2.0, 2.0, -2.0, 1.0], np.float32)
    emission = np.asarray([1.0, 1.0, 1.0], np.float32) * 5.0
    ambient = np.asarray([1.0, 1.0, 1.0], np.float32) * 0.01
    phong_rgb = np.asarray([1.0, 0.8, 0.6], np.float32)
    phong_exp = 100.0
    proj = util.projection(x=x, n=n, f=f)

    glctx = dr.RasterizeCudaContext()

    scenedir = f"{pathlib.Path(__file__).absolute().parents[0]}/scene"
    scene = trimesh.load(f"{scenedir}/spot/spot_triangulated_good.obj")
    pos_data = scene.vertices
    pos_idx_data = scene.faces
    normals_data = scene.vertex_normals
    uvs_data = scene.visual.uv

    pos = np.asarray(pos_data, np.float32).reshape(-1, 3)
    pos_idx = np.asarray(pos_idx_data, np.int32).reshape(-1, 3)
    normals = np.asarray(normals_data, np.float32).reshape(-1, 3)
    uvs = np.asarray(uvs_data, np.float32).reshape(-1, 2)
    tex_data = Image.open(f"{scenedir}/spot/spot_texture.png").transpose(Image.FLIP_LEFT_RIGHT)
    tex = np.asarray(tex_data, np.float32) / 255.0

    pos_idx = torch.as_tensor(pos_idx, dtype=torch.int32, device="cuda")
    pos = torch.as_tensor(pos, dtype=torch.float32, device="cuda")
    normals = torch.as_tensor(normals, dtype=torch.float32, device="cuda")
    uvs = torch.as_tensor(uvs, dtype=torch.float32, device="cuda")
    emission = torch.as_tensor(emission, dtype=torch.float32, device="cuda")[None, ...]
    ambient = torch.as_tensor(ambient, dtype=torch.float32, device="cuda")[None, ...]
    tex = torch.as_tensor(tex, dtype=torch.float32, device="cuda")
    pos = torch.cat([pos, torch.ones_like(pos[..., :1])], axis=-1)
    phong_rgb_t = torch.as_tensor(phong_rgb, dtype=torch.float32, device="cuda")

    ang = 0.0
    frames = []
    for _ in tqdm(range(samples)):
        ang = ang + 0.1
        a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(0.8 * np.pi))
        a_rot_light = util.rotate_y(-ang)
        a_mv = np.matmul(util.translate(0, 0, -3.5), a_rot)
        a_mvp = np.matmul(proj, a_mv).astype(np.float32)
        a_mvp = torch.as_tensor(a_mvp, dtype=torch.float32, device="cuda")[None, ...]
        a_camera_pos = torch.as_tensor(np.linalg.inv(a_mv)[:3, 3], dtype=torch.float32, device="cuda")[None, ...]
        a_light_pos = torch.as_tensor(np.matmul(a_rot_light, light_pos)[..., :3], dtype=torch.float32, device="cuda")[None, ...]
        color = render_point_refl(
            glctx,
            pos,
            pos_idx,
            normals,
            uvs,
            a_camera_pos,
            a_light_pos,
            a_mvp,
            emission,
            ambient,
            tex,
            phong_rgb_t,
            phong_exp,
            resolution,
        )
        result_image = color.detach()[0].cpu().numpy()[::-1]
        frames.append((result_image * 255).astype(np.uint8))
    with imageio.get_writer("./output/ref_move.mp4", fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
        frames = []


if __name__ == "__main__":
    ref_generator()
    print("Done.")
