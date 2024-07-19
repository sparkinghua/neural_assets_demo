import pathlib
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
from PIL import Image
from matplotlib import pyplot as plt

import util
from model import RNA
from render import render_rna, render_refl


if __name__ == "__main__":
    datadir = f"{pathlib.Path(__file__).absolute().parents[0]}/data"
    log_interval = 10
    display_interval = 10
    display_res = 1024

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

    resolution = 256

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

    max_iter = 1000000
    lr = 1e-3

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000)

    # Render.
    ang = 0.0
    imgloss_avg = []
    imagloss_log = []
    glctx = dr.RasterizeCudaContext()

    for it in range(max_iter + 1):
        # Random rotation/translation matrix for optimization.
        r_rot = util.random_rotation_translation(0.25)
        r_rot_light = util.rotate_y(0.25)

        # Smooth rotation for display.
        ang = ang + 0.01
        a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(0.8 * np.pi))
        a_rot_light = util.rotate_y(-ang)

        # Modelview and modelview + projection matrices.
        proj = util.projection(x=0.4, n=1.0, f=200.0)
        r_mv = np.matmul(util.translate(0, 0, -3.5), r_rot)
        r_mvp = np.matmul(proj, r_mv).astype(np.float32)
        a_mv = np.matmul(util.translate(0, 0, -3.5), a_rot)
        a_mvp = np.matmul(proj, a_mv).astype(np.float32)
        a_mvc = a_mvp
        r_mvp = torch.as_tensor(r_mvp, dtype=torch.float32, device="cuda")
        a_mvp = torch.as_tensor(a_mvp, dtype=torch.float32, device="cuda")

        # Solve camera positions.
        a_camera_pos = torch.as_tensor(np.linalg.inv(a_mv)[:3, 3], dtype=torch.float32, device="cuda")
        r_camera_pos = torch.as_tensor(np.linalg.inv(r_mv)[:3, 3], dtype=torch.float32, device="cuda")

        # Solve light positions.
        a_light_pos = torch.as_tensor(np.matmul(a_rot_light, light_pos)[..., :3], dtype=torch.float32, device="cuda")
        r_light_pos = torch.as_tensor(np.matmul(r_rot_light, light_pos)[..., :3], dtype=torch.float32, device="cuda")

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
        optimizer.step()
        scheduler.step()

        imagloss_log.append(loss.detach().cpu().numpy())
        imgloss_avg.append(imagloss_log[-1])

        if log_interval and (it % log_interval == 0):
            imgloss_val, imgloss_avg = np.mean(np.asarray(imgloss_avg, np.float32)), []
            s = "iter=%d, img_rmse=%f" % (it, imgloss_val)
            print(s)

        display_image = display_interval and (it % display_interval == 0)
        if display_image:
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
    print("Done.")
    # ----------------------------------------------------------------------------
    plt.plot(imagloss_log)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Loss vs Iterations")
    plt.grid()
    plt.savefig("loss.png")
    plt.show()
