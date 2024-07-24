import os
import torch
import nvdiffrast.torch as dr
from PIL import Image
import numpy as np
import trimesh
import pathlib
import imageio
import json
from tqdm import tqdm
import argparse

import util
from render import render_point_refl, render_rand_refl


def fibonacci_sphere(samples=1024, radius=1.0):
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        points.append((x * radius, y * radius, z * radius))
    return np.array(points, dtype=np.float32)


def data_generator_static(args):
    samples: int = args.samples
    radius: float = args.radius
    x: float = args.x
    n: float = args.n
    f: float = args.f
    resolution: list = args.resolution
    dataset: str = args.dataset
    data_name: str = args.data_name

    light_pos = np.asarray([2.0, 2.0, -2.0, 1.0], np.float32)
    emission = np.asarray([1.0, 1.0, 1.0], np.float32) * 5.0
    ambient = np.asarray([1.0, 1.0, 1.0], np.float32) * 0.01
    phong_rgb = np.asarray([1.0, 0.8, 0.6], np.float32)
    phong_exp = 100.0
    proj = util.projection(x=x, n=n, f=f)

    ans_dict = dict()
    ans_dict["samples"] = samples
    ans_dict["resolution"] = resolution
    ans_dict["light_pos"] = light_pos.tolist()
    ans_dict["emission"] = emission.tolist()
    ans_dict["ambient"] = ambient.tolist()
    ans_dict["phong_rgb"] = phong_rgb.tolist()
    ans_dict["phong_exp"] = phong_exp
    ans_dict["proj"] = proj.tolist()

    r_mvp = []
    r_camera_pos = []
    r_light_pos = []
    r_emission = []
    r_ambient = []

    for i in range(samples):
        r_rot = util.random_rotation_translation(0.25)
        r_mv = np.matmul(util.translate(0, 0, -radius), r_rot)
        r_mvp.append(np.matmul(proj, r_mv).astype(np.float32))
        r_camera_pos.append(np.linalg.inv(r_mv)[:3, 3])
        r_light_pos.append(light_pos[..., :3])
        r_emission.append(emission)
        r_ambient.append(ambient)
    r_light_pos = np.stack(r_light_pos, axis=0)
    r_camera_pos = np.stack(r_camera_pos, axis=0)
    r_mvp = np.stack(r_mvp, axis=0)
    r_emission = np.stack(r_emission, axis=0)
    r_ambient = np.stack(r_ambient, axis=0)

    datadir = f"{pathlib.Path(__file__).absolute().parents[0]}/data/{data_name}/{dataset}"
    if not os.path.exists(datadir):
        os.makedirs(datadir, exist_ok=True)
    with open(f"{datadir}/ans.json", "w") as f:
        json.dump(ans_dict, f, indent=2)
    with open(f"{datadir}/meta.npy", "wb") as f:
        np.savez_compressed(
            f, camera_pos=r_camera_pos, light_pos=r_light_pos, mvp=r_mvp, emission=r_emission, ambient=r_ambient
        )

    r_light_pos = torch.as_tensor(r_light_pos, dtype=torch.float32, device="cuda")
    r_camera_pos = torch.as_tensor(r_camera_pos, dtype=torch.float32, device="cuda")
    r_mvp = torch.as_tensor(r_mvp, dtype=torch.float32, device="cuda")
    r_emission = torch.as_tensor(r_emission, dtype=torch.float32, device="cuda")
    r_ambient = torch.as_tensor(r_ambient, dtype=torch.float32, device="cuda")
    phong_rgb_t = torch.as_tensor(phong_rgb, dtype=torch.float32, device="cuda")

    bacth_size = 16
    imgs = []
    glctx = dr.RasterizeCudaContext()

    scenedir = f"{pathlib.Path(__file__).absolute().parents[0]}/scene"
    scene = trimesh.load(f"{scenedir}/spot/spot.obj")

    pos = np.asarray(scene.vertices, np.float32).reshape(-1, 3)
    pos_idx = np.asarray(scene.faces, np.int32).reshape(-1, 3)
    normals = np.asarray(scene.vertex_normals, np.float32).reshape(-1, 3)
    uvs = np.asarray(scene.visual.uv, np.float32).reshape(-1, 2)
    tex_data = Image.open(f"{scenedir}/spot/spot_texture.png").transpose(Image.FLIP_LEFT_RIGHT)
    tex = np.asarray(tex_data, np.float32) / 255.0

    pos_idx = torch.as_tensor(pos_idx, dtype=torch.int32, device="cuda")
    pos = torch.as_tensor(pos, dtype=torch.float32, device="cuda")
    normals = torch.as_tensor(normals, dtype=torch.float32, device="cuda")
    uvs = torch.as_tensor(uvs, dtype=torch.float32, device="cuda")
    tex = torch.as_tensor(tex, dtype=torch.float32, device="cuda")
    pos = torch.cat([pos, torch.ones_like(pos[..., :1])], axis=-1)
    for i in tqdm(range(0, samples, bacth_size)):
        r_camera_pos_batch = r_camera_pos[i : i + bacth_size]
        r_light_pos_batch = r_light_pos[i : i + bacth_size]
        r_mvp_batch = r_mvp[i : i + bacth_size]
        r_emission_batch = r_emission[i : i + bacth_size]
        r_ambient_batch = r_ambient[i : i + bacth_size]
        with torch.no_grad():
            img_batch = render_point_refl(
                glctx,
                pos,
                pos_idx,
                normals,
                uvs,
                r_camera_pos_batch,
                r_light_pos_batch,
                r_mvp_batch,
                r_emission_batch,
                r_ambient_batch,
                tex,
                phong_rgb_t,
                phong_exp,
                resolution,
            )
            img_batch = img_batch.detach().cpu().numpy()
            for img in img_batch:
                imgs.append(img)
    for i, img in enumerate(imgs):
        imageio.imwrite(f"{datadir}/img_{i:04d}.png", (img * 255).astype(np.uint8))
    print("Data generation done.")


def data_generator_move(args):
    samples: int = args.samples
    radius: float = args.radius
    x: float = args.x
    n: float = args.n
    f: float = args.f
    resolution: list = args.resolution
    dataset: str = args.dataset
    data_name: str = args.data_name

    light_pos = np.asarray([2.0, 2.0, -2.0, 1.0], np.float32)
    emission = np.asarray([1.0, 1.0, 1.0], np.float32) * 5.0
    ambient = np.asarray([1.0, 1.0, 1.0], np.float32) * 0.01
    phong_rgb = np.asarray([1.0, 0.8, 0.6], np.float32)
    phong_exp = 100.0
    proj = util.projection(x=x, n=n, f=f)

    ans_dict = dict()
    ans_dict["samples"] = samples
    ans_dict["resolution"] = resolution
    ans_dict["light_pos"] = light_pos.tolist()
    ans_dict["emission"] = emission.tolist()
    ans_dict["ambient"] = ambient.tolist()
    ans_dict["phong_rgb"] = phong_rgb.tolist()
    ans_dict["phong_exp"] = phong_exp
    ans_dict["proj"] = proj.tolist()

    r_mvp = []
    r_camera_pos = []
    r_light_pos = []
    r_emission = []
    r_ambient = []

    for i in range(samples):
        r_rot = util.random_rotation_translation(0.25)
        r_rot_light = util.random_rotation_translation(0.25)
        r_mv = np.matmul(util.translate(0, 0, -radius), r_rot)
        r_mvp.append(np.matmul(proj, r_mv).astype(np.float32))
        r_camera_pos.append(np.linalg.inv(r_mv)[:3, 3])
        r_light_pos.append(np.matmul(r_rot_light, light_pos)[..., :3])
        r_emission.append(emission * np.random.uniform(0.5, 1.5, 3))
        r_ambient.append(ambient * np.random.uniform(0.5, 1.5, 3))
    r_light_pos = np.stack(r_light_pos, axis=0)
    r_camera_pos = np.stack(r_camera_pos, axis=0)
    r_mvp = np.stack(r_mvp, axis=0)
    r_emission = np.stack(r_emission, axis=0)
    r_ambient = np.stack(r_ambient, axis=0)

    datadir = f"{pathlib.Path(__file__).absolute().parents[0]}/data/{data_name}/{dataset}"
    if not os.path.exists(datadir):
        os.makedirs(datadir, exist_ok=True)
    with open(f"{datadir}/ans.json", "w") as f:
        json.dump(ans_dict, f, indent=2)
    with open(f"{datadir}/meta.npy", "wb") as f:
        np.savez_compressed(
            f, camera_pos=r_camera_pos, light_pos=r_light_pos, mvp=r_mvp, emission=r_emission, ambient=r_ambient
        )

    r_light_pos = torch.as_tensor(r_light_pos, dtype=torch.float32, device="cuda")
    r_camera_pos = torch.as_tensor(r_camera_pos, dtype=torch.float32, device="cuda")
    r_mvp = torch.as_tensor(r_mvp, dtype=torch.float32, device="cuda")
    r_emission = torch.as_tensor(r_emission, dtype=torch.float32, device="cuda")
    r_ambient = torch.as_tensor(r_ambient, dtype=torch.float32, device="cuda")
    phong_rgb_t = torch.as_tensor(phong_rgb, dtype=torch.float32, device="cuda")

    bacth_size = 16
    imgs = []
    glctx = dr.RasterizeCudaContext()

    scenedir = f"{pathlib.Path(__file__).absolute().parents[0]}/scene"
    scene = trimesh.load(f"{scenedir}/spot/spot.obj")

    pos = np.asarray(scene.vertices, np.float32).reshape(-1, 3)
    pos_idx = np.asarray(scene.faces, np.int32).reshape(-1, 3)
    normals = np.asarray(scene.vertex_normals, np.float32).reshape(-1, 3)
    uvs = np.asarray(scene.visual.uv, np.float32).reshape(-1, 2)
    tex_data = Image.open(f"{scenedir}/spot/spot_texture.png").transpose(Image.FLIP_LEFT_RIGHT)
    tex = np.asarray(tex_data, np.float32) / 255.0

    pos_idx = torch.as_tensor(pos_idx, dtype=torch.int32, device="cuda")
    pos = torch.as_tensor(pos, dtype=torch.float32, device="cuda")
    normals = torch.as_tensor(normals, dtype=torch.float32, device="cuda")
    uvs = torch.as_tensor(uvs, dtype=torch.float32, device="cuda")
    tex = torch.as_tensor(tex, dtype=torch.float32, device="cuda")
    pos = torch.cat([pos, torch.ones_like(pos[..., :1])], axis=-1)
    for i in tqdm(range(0, samples, bacth_size)):
        r_camera_pos_batch = r_camera_pos[i : i + bacth_size]
        r_light_pos_batch = r_light_pos[i : i + bacth_size]
        r_mvp_batch = r_mvp[i : i + bacth_size]
        r_emission_batch = r_emission[i : i + bacth_size]
        r_ambient_batch = r_ambient[i : i + bacth_size]

        with torch.no_grad():
            img_batch = render_point_refl(
                glctx,
                pos,
                pos_idx,
                normals,
                uvs,
                r_camera_pos_batch,
                r_light_pos_batch,
                r_mvp_batch,
                r_emission_batch,
                r_ambient_batch,
                tex,
                phong_rgb_t,
                phong_exp,
                resolution,
            )
            img_batch = img_batch.detach().cpu().numpy()
            for img in img_batch:
                imgs.append(img)
    for i, img in enumerate(imgs):
        imageio.imwrite(f"{datadir}/img_{i:04d}.png", (img * 255).astype(np.uint8))
    print("Data generation done.")


def data_generator_rand(args):
    samples: int = args.samples
    radius: float = args.radius
    x: float = args.x
    n: float = args.n
    f: float = args.f
    resolution: list = args.resolution
    dataset: str = args.dataset
    data_name: str = args.data_name

    light_pos = np.asarray([2.0, 2.0, -2.0, 1.0], np.float32)
    emission = np.asarray([1.0, 1.0, 1.0], np.float32) * 5.0
    ambient = np.asarray([1.0, 1.0, 1.0], np.float32) * 0.01
    phong_rgb = np.asarray([1.0, 0.8, 0.6], np.float32)
    phong_exp = 100.0
    proj = util.projection(x=x, n=n, f=f)

    ans_dict = dict()
    ans_dict["samples"] = samples
    ans_dict["resolution"] = resolution
    ans_dict["light_pos"] = light_pos.tolist()
    ans_dict["emission"] = emission.tolist()
    ans_dict["ambient"] = ambient.tolist()
    ans_dict["phong_rgb"] = phong_rgb.tolist()
    ans_dict["phong_exp"] = phong_exp
    ans_dict["proj"] = proj.tolist()

    r_mvp = []
    r_camera_pos = []
    r_ambient = []

    for i in range(samples):
        r_rot = util.random_rotation_translation(0.25)
        r_mv = np.matmul(util.translate(0, 0, -radius), r_rot)
        r_mvp.append(np.matmul(proj, r_mv).astype(np.float32))
        r_camera_pos.append(np.linalg.inv(r_mv)[:3, 3])
        r_ambient.append(ambient * np.random.uniform(0.5, 1.5, 3))
    r_camera_pos = np.stack(r_camera_pos, axis=0)
    r_mvp = np.stack(r_mvp, axis=0)
    r_ambient = np.stack(r_ambient, axis=0)

    datadir = f"{pathlib.Path(__file__).absolute().parents[0]}/data/{data_name}/{dataset}"
    if not os.path.exists(datadir):
        os.makedirs(datadir, exist_ok=True)
    with open(f"{datadir}/ans.json", "w+") as f:
        json.dump(ans_dict, f, indent=2)
    with open(f"{datadir}/meta.npy", "wb") as f:
        np.savez_compressed(f, camera_pos=r_camera_pos, mvp=r_mvp, ambient=r_ambient)

    r_camera_pos = torch.as_tensor(r_camera_pos, dtype=torch.float32, device="cuda")
    r_mvp = torch.as_tensor(r_mvp, dtype=torch.float32, device="cuda")
    r_ambient = torch.as_tensor(r_ambient, dtype=torch.float32, device="cuda")
    phong_rgb_t = torch.as_tensor(phong_rgb, dtype=torch.float32, device="cuda")

    bacth_size = 4
    imgs = []
    glctx = dr.RasterizeCudaContext()

    scenedir = f"{pathlib.Path(__file__).absolute().parents[0]}/scene"
    scene = trimesh.load(f"{scenedir}/spot/spot.obj")

    pos = np.asarray(scene.vertices, np.float32).reshape(-1, 3)
    pos_idx = np.asarray(scene.faces, np.int32).reshape(-1, 3)
    normals = np.asarray(scene.vertex_normals, np.float32).reshape(-1, 3)
    uvs = np.asarray(scene.visual.uv, np.float32).reshape(-1, 2)
    tex_data = Image.open(f"{scenedir}/spot/spot_texture.png").transpose(Image.FLIP_LEFT_RIGHT)
    tex = np.asarray(tex_data, np.float32) / 255.0

    pos_idx = torch.as_tensor(pos_idx, dtype=torch.int32, device="cuda")
    pos = torch.as_tensor(pos, dtype=torch.float32, device="cuda")
    normals = torch.as_tensor(normals, dtype=torch.float32, device="cuda")
    uvs = torch.as_tensor(uvs, dtype=torch.float32, device="cuda")
    tex = torch.as_tensor(tex, dtype=torch.float32, device="cuda")
    pos = torch.cat([pos, torch.ones_like(pos[..., :1])], axis=-1)
    cnt = 0
    for i in tqdm(range(0, samples, bacth_size)):
        r_camera_pos_batch = r_camera_pos[i : i + bacth_size]
        r_ambient_batch = r_ambient[i : i + bacth_size]
        r_mvp_batch = r_mvp[i : i + bacth_size]

        ksi = np.random.uniform(0.0, 1.0, (bacth_size, resolution[0], resolution[1], 2))
        phi = 2.0 * np.pi * ksi[..., 0]
        z = 2.0 * ksi[..., 1] - 1.0
        r = np.sqrt(1.0 - z**2)
        x, y = np.cos(phi) * r, np.sin(phi) * r
        r_light_dir = np.stack([x, y, z], axis=-1)

        r_emission = np.random.uniform(0.1, 2.0, (bacth_size, resolution[0], resolution[1], 3))

        r_light_dir_batch = np.asarray(r_light_dir, np.float32)
        r_light_dir_batch = torch.as_tensor(r_light_dir_batch, dtype=torch.float32, device="cuda")
        r_emission_batch = np.asarray(r_emission, np.float32)
        r_emission_batch = torch.as_tensor(r_emission_batch, dtype=torch.float32, device="cuda")

        with torch.no_grad():
            img_batch = render_rand_refl(
                glctx,
                pos,
                pos_idx,
                normals,
                uvs,
                r_camera_pos_batch,
                r_light_dir_batch,
                r_mvp_batch,
                r_emission_batch,
                r_ambient_batch,
                tex,
                phong_rgb_t,
                phong_exp,
                resolution,
            )
            img_batch = img_batch.detach().cpu().numpy()
            for b in range(bacth_size):
                imageio.imwrite(f"{datadir}/img_{cnt:04d}.png", (img_batch[b] * 255).astype(np.uint8))
                np.save(f"{datadir}/light_dir_{cnt:04d}.npy", r_light_dir[b])
                np.save(f"{datadir}/emission_{cnt:04d}.npy", r_emission[b])
                cnt += 1
    print("Data generation done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--radius", type=float, default=3.5)
    parser.add_argument("--x", type=float, default=0.4)
    parser.add_argument("--n", type=float, default=1.0)
    parser.add_argument("--f", type=float, default=200.0)
    parser.add_argument("--resolution", type=int, nargs=2, default=[512, 512])

    parser.add_argument("--data-name", type=str, choices=["static", "move", "rand"], default="static")
    parser.add_argument("--dataset", type=str, choices=["train", "test", "valid"], default="train")

    args = parser.parse_args()

    data_name = args.data_name
    if data_name == "static":
        data_generator_static(args)
    elif data_name == "move":
        data_generator_move(args)
    elif data_name == "rand":
        data_generator_rand(args)
    else:
        raise NotImplementedError
