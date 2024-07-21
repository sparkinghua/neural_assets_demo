import pathlib
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
from PIL import Image
from matplotlib import pyplot as plt
import imageio

import util
from model import RNA
from render import render_rna, render_refl


if __name__ == "__main__":
    datadir = f"{pathlib.Path(__file__).absolute().parents[0]}/data"
    log_interval = 10
    display_interval = 4
    display_res = 1024
    frames = []
    fps = 10

    scene = trimesh.load(f"{datadir}/spot/spot_triangulated_good.obj")
    pos_data = scene.vertices
    pos_idx_data = scene.faces
    normals_data = scene.vertex_normals
    uvs_data = scene.visual.uv

    tex_data = Image.open(f"{datadir}/spot/spot_texture.png").transpose(Image.FLIP_LEFT_RIGHT)
    tex = np.array(tex_data, np.float32) / 255.0

    pos = np.array(pos_data, np.float32).reshape(-1, 3)
    pos_idx = np.array(pos_idx_data, np.int32).reshape(-1, 3)
    normals = np.array(normals_data, np.float32).reshape(-1, 3)
    uvs = np.array(uvs_data, np.float32).reshape(-1, 2)
    light_pos = np.asarray([2.0, 2.0, -2.0, 1.0], np.float32)
    emission = np.asarray([1.0, 1.0, 1.0], np.float32) * 5.0
    ambient = np.asarray([1.0, 1.0, 1.0], np.float32) * 0.01
    print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], pos.shape[0]))
    # print(pos.shape, pos_idx.shape, normals.shape, uvs.shape, tex.shape)
    # exit()

    # Move all the stuff to GPU.
    pos_idx = torch.as_tensor(pos_idx, dtype=torch.int32, device="cuda")
    pos = torch.as_tensor(pos, dtype=torch.float32, device="cuda")
    normals = torch.as_tensor(normals, dtype=torch.float32, device="cuda")
    uvs = torch.as_tensor(uvs, dtype=torch.float32, device="cuda")
    emission = torch.as_tensor(emission, dtype=torch.float32, device="cuda")
    ambient = torch.as_tensor(ambient, dtype=torch.float32, device="cuda")
    tex = torch.as_tensor(tex, dtype=torch.float32, device="cuda")

    pos = torch.cat([pos, torch.ones_like(pos[..., :1])], axis=-1)

    resolution = 512

    # Target Phong parameters.
    phong_rgb = np.asarray([1.0, 0.8, 0.6], np.float32)
    phong_exp = 100.0
    phong_rgb_t = torch.as_tensor(phong_rgb, dtype=torch.float32, device="cuda")

    # Learned variables: neural network.
    in_features = 12
    hidden_features = 256
    out_features = 3
    num_layers = 4
    model = RNA(in_features, out_features, hidden_features, num_layers).cuda()

    max_iter = 40000
    warmup_iter = 1000
    batch_size = 2
    lr = 1e-3

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter - warmup_iter, eta_min=1e-7)

    # Render.
    ang = 0.0
    imgloss_avg = []
    imgloss_log = []
    imgpnsr_log = []
    glctx = dr.RasterizeCudaContext()

    proj = util.projection(x=0.4, n=1.0, f=200.0)

    for it in range(max_iter):
        model.train()
        r_camera_pos = []
        r_light_pos = []
        r_mvp = []
        for _ in range(batch_size):
            # Random rotation/translation matrix for optimization.
            r_rot = util.random_rotation_translation(0.25)
            r_rot_light = util.rotate_y(0.25)
            # Modelview and modelview + projection matrices.
            r_mv = np.matmul(util.translate(0, 0, -3.5), r_rot)
            r_mvp.append(np.matmul(proj, r_mv).astype(np.float32))
            # Solve camera positions.
            r_camera_pos.append(np.linalg.inv(r_mv)[:3, 3])
            # Solve light positions.
            r_light_pos.append(np.matmul(r_rot_light, light_pos)[..., :3])

        r_camera_pos = torch.as_tensor(np.stack(r_camera_pos, axis=0), dtype=torch.float32, device="cuda")
        r_light_pos = torch.as_tensor(np.stack(r_light_pos, axis=0), dtype=torch.float32, device="cuda")
        r_mvp = torch.as_tensor(np.stack(r_mvp, axis=0), dtype=torch.float32, device="cuda")

        color = render_refl(
            glctx,
            pos,
            pos_idx,
            normals,
            uvs,
            r_camera_pos,
            r_light_pos,
            r_mvp,
            emission,
            ambient,
            tex,
            phong_rgb_t,
            phong_exp,
            resolution,
        )

        color_opt = render_rna(
            glctx, pos, pos_idx, normals, uvs, r_camera_pos, r_light_pos, r_mvp, emission, ambient, resolution, model
        )

        loss = torch.mean((color - color_opt) ** 2)  # L2 pixel loss.
        optimizer.zero_grad()
        loss.backward()
        if it < warmup_iter:
            optimizer.param_groups[0]["lr"] = lr * (it + 1) / warmup_iter
            optimizer.step()
        else:
            optimizer.step()
            scheduler.step()

        imgloss_log.append(loss.detach().cpu().numpy())
        imgloss_avg.append(imgloss_log[-1])
        imgpnsr_log.append(-10.0 * np.log10(imgloss_log[-1]))

        if log_interval and (it % log_interval == 0):
            imgloss_val, imgloss_avg = np.mean(np.asarray(imgloss_avg, np.float32)), []
            psnr_val = -10.0 * np.log10(imgloss_val)
            s = "iter=%d, img_rmse=%f, img_psnr=%f" % (it, imgloss_val, psnr_val)
            print(s)

        display_image = display_interval and (it % (10 // batch_size) == 0)
        if display_image:
            model.eval()
            ang = ang + 0.1
            a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(ang))
            # a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(0.8 * np.pi))
            a_rot_light = util.rotate_y(0)
            # a_rot_light = util.rotate_y(-ang)
            a_mv = np.matmul(util.translate(0, 0, -3.5), a_rot)
            a_mvp = np.matmul(proj, a_mv).astype(np.float32)
            a_mvp = torch.as_tensor(a_mvp, dtype=torch.float32, device="cuda")[None, ...]
            a_camera_pos = torch.as_tensor(np.linalg.inv(a_mv)[:3, 3], dtype=torch.float32, device="cuda")[None, ...]
            a_light_pos = torch.as_tensor(np.matmul(a_rot_light, light_pos)[..., :3], dtype=torch.float32, device="cuda")[
                None, ...
            ]
            # color_opt = render_refl(
            #     glctx,
            #     pos,
            #     pos_idx,
            #     normals,
            #     uvs,
            #     a_camera_pos,
            #     a_light_pos,
            #     a_mvp,
            #     emission,
            #     ambient,
            #     tex,
            #     phong_rgb_t,
            #     phong_exp,
            #     resolution,
            # )
            color_opt = render_rna(
                glctx, pos, pos_idx, normals, uvs, a_camera_pos, a_light_pos, a_mvp, emission, ambient, resolution, model
            )
            result_image = color_opt.detach()[0].cpu().numpy()[::-1]
            util.display_image(result_image, size=display_res, title="%d / %d" % (it, max_iter))
            frames.append((result_image * 255).astype(np.uint8))
    with imageio.get_writer("./output/result.mp4", fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
        frames = []
    print("Done.")
    # ----------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Image Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid()
    ax.scatter(range(len(imgloss_log)), imgloss_log)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Image PSNR")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PSNR")
    ax.grid()
    ax.scatter(range(len(imgpnsr_log)), imgpnsr_log)

    fig.savefig("./output/result.png")
    plt.show()
