import torch
import numpy as np
import pathlib
import trimesh

from configs import MainConfig


class Mesh:
    def __init__(self, config: MainConfig):
        self.config = config
        """Load scene: vertices, faces, normals, and uvs."""
        workdir = f"{pathlib.Path(__file__).absolute().parents[0]}"
        scenedir = f"{workdir}/{config.file_paths.scene_dir}"
        obj_path = f"{scenedir}/{config.file_paths.scene}/{config.file_paths.scene}.obj"

        print("Loading meshes...")
        scene = trimesh.load(obj_path)
        self.pos = np.asarray(scene.vertices, np.float32).reshape(-1, 3)
        self.pos_idx = np.asarray(scene.faces, np.int32).reshape(-1, 3)
        self.normals = np.asarray(scene.vertex_normals, np.float32).reshape(-1, 3)
        if config.model_param.has_uv:
            self.uvs = np.asarray(scene.visual.uv, np.float32).reshape(-1, 2)
        else:
            self.uvs = None
        print("Mesh has %d triangles and %d vertices." % (self.pos_idx.shape[0], self.pos.shape[0]))

        self.pos_idx = torch.as_tensor(self.pos_idx, dtype=torch.int32, device="cuda")
        self.pos = torch.as_tensor(self.pos, dtype=torch.float32, device="cuda")
        self.pos = torch.cat([self.pos, torch.ones_like(self.pos[..., :1])], axis=-1)
        self.normals = torch.as_tensor(self.normals, dtype=torch.float32, device="cuda")
        if self.uvs is not None:
            self.uvs = torch.as_tensor(self.uvs, dtype=torch.float32, device="cuda")
