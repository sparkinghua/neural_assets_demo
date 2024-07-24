import numpy as np
import torch
from torch import nn

from configs import ModelParamConfig


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


class RNA(nn.Module):
    def __init__(self, config: ModelParamConfig):
        super(RNA, self).__init__()
        self.config = config
        self.in_features = config.in_features
        self.out_features = config.out_features
        self.hidden_features = config.hidden_features
        self.num_layers = config.num_layers
        self.has_uv = config.has_uv
        self.fourier_enc = config.fourier_enc
        if self.fourier_enc:
            self.L_pos = config.L_pos
            self.L_dir = config.L_dir
            self.in_features += 6 * self.L_pos + 6 * self.L_dir * 3
        else:
            self.L_pos = 0
            self.L_dir = 0

        if self.has_uv:
            self.in_features += config.tex_channels
            self.tex_res = config.tex_res
            self.tex_channels = config.tex_channels
            self.texture = nn.Parameter(torch.zeros(1, self.tex_channels, self.tex_res, self.tex_res))
            torch.nn.init.kaiming_uniform_(self.texture)
        else:
            self.tex_res = None
            self.texture = None

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.in_features, self.hidden_features))
        self.layers.append(nn.LeakyReLU())
        for _ in range(self.num_layers - 2):
            self.layers.append(nn.Linear(self.hidden_features, self.hidden_features))
            self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(self.hidden_features, self.out_features))
        self.layers.append(nn.ReLU())

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_uv:
            uvs = 2.0 * x[..., -2:] - 1.0
            color = torch.nn.functional.grid_sample(
                self.texture.expand(uvs.shape[0], -1, -1, -1), uvs, mode="bilinear", padding_mode="border", align_corners=True
            ).permute(0, 2, 3, 1)
            # color = torch.clamp(color, 0.0, 1.0)
            x = torch.cat([x[..., :-2], color], dim=-1)
        if self.fourier_enc:
            pos = x[..., :3]
            pos_enc = pos
            for l in range(self.L_pos):
                pos_enc = torch.cat([pos_enc, torch.sin(2**l * np.pi * pos), torch.cos(2**l * np.pi * pos)], dim=-1)
            dir = x[..., 3:12]
            dir_enc = dir
            for l in range(self.L_dir):
                dir_enc = torch.cat([dir_enc, torch.sin(2**l * np.pi * dir), torch.cos(2**l * np.pi * dir)], dim=-1)
            output = torch.cat([pos_enc, dir_enc, x[..., 12:]], dim=-1)
        else:
            output = x

        for layer in self.layers:
            output = layer(output)
        return output
