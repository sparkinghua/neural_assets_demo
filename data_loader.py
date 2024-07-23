import torch
import numpy as np
import pathlib
from tqdm import tqdm
import imageio
from torch.utils.data import Dataset, DataLoader

from configs import MainConfig


class RenderDataset(Dataset):
    """Dataset class for loading scene data."""

    def __init__(self, config):
        self.config = config

        workdir = f"{pathlib.Path(__file__).absolute().parents[0]}"
        datadir = f"{workdir}/{config.file_paths.data_dir}/{config.file_paths.data}"
        with open(f"{datadir}/meta.npy", "rb") as f:
            data = np.load(f, allow_pickle=True)
            self.r_light_pos = np.asarray(data["light_pos"], np.float32)
            self.r_camera_pos = np.asarray(data["camera_pos"], np.float32)
            self.r_mvp = np.asarray(data["mvp"], np.float32)
            self.r_emission = np.asarray(data["emission"], np.float32)
            self.r_ambient = np.asarray(data["ambient"], np.float32)
        self.samples = self.r_mvp.shape[0]
        # shuffle the data
        self.index = np.arange(0, self.samples)
        np.random.shuffle(self.index)
        self.r_light_pos = self.r_light_pos[self.index]
        self.r_camera_pos = self.r_camera_pos[self.index]
        self.r_mvp = self.r_mvp[self.index]
        self.r_emission = self.r_emission[self.index]
        self.r_ambient = self.r_ambient[self.index]

        print("Loading images...")
        self.imgs = []
        for i in tqdm(range(self.samples)):
            img = imageio.v2.imread(f"{datadir}/img_{self.index[i]:04d}.png")
            img = np.asarray(img, np.float32) / 255.0
            self.imgs.append(img)
        self.resolution = np.asarray(self.imgs[0].shape[:2], np.int32)
        print("Loaded %d images with resolution %dx%d." % (self.samples, self.resolution[0], self.resolution[1]))

        # Move all the stuff to GPU.
        self.r_emission = torch.as_tensor(self.r_emission, dtype=torch.float32, device="cuda")
        self.r_ambient = torch.as_tensor(self.r_ambient, dtype=torch.float32, device="cuda")
        self.r_light_pos = torch.as_tensor(self.r_light_pos, dtype=torch.float32, device="cuda")
        self.r_camera_pos = torch.as_tensor(self.r_camera_pos, dtype=torch.float32, device="cuda")
        self.r_mvp = torch.as_tensor(self.r_mvp, dtype=torch.float32, device="cuda")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img_batch = torch.as_tensor(self.imgs[idx], dtype=torch.float32, device="cuda")
        return (
            img_batch,
            self.r_light_pos[idx],
            self.r_camera_pos[idx],
            self.r_mvp[idx],
            self.r_emission[idx],
            self.r_ambient[idx],
        )
    
    def get_data(self):
        return self.r_light_pos, self.r_camera_pos, self.r_mvp, self.r_emission, self.r_ambient
    
    def get_img(self):
        return self.imgs


def load_data(config: MainConfig) -> tuple[DataLoader, dict]:
    """Load data: reference images, light positions, camera positions, mvp matrices, emission, and ambient."""
    dataset = RenderDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.optim.batch_size, shuffle=True, num_workers=0)
    metadata = dict()
    metadata["samples"] = dataset.samples
    metadata["resolution"] = dataset.resolution
    return dataloader, metadata
