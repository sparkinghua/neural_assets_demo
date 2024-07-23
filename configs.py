from dataclasses import dataclass, asdict
from typing import Optional
import tyro


@dataclass(frozen=True)
class FilePathConfig:
    """Configuration for file paths."""

    data_dir: str = "data"
    """Directory containing data."""
    data: str = "move_light"
    """Name of the data."""
    output_dir: str = "output"
    """Directory for output."""
    scene_dir: str = "scene"
    """Directory containing scenes."""
    scene: str = "spot"
    """Name of the scene."""
    resume: Optional[str] = None
    """path to the checkpoint to resume training from"""


@dataclass(frozen=True)
class LogDisplayConfig:
    """Configuration for logging and display."""

    log_interval: int = 64
    """Log metrics and learning rate every N iterations."""
    display_interval: int = 64
    """Display images every N iterations."""
    display_res: int = 1024
    """Resolution of displayed images."""
    fps: int = 10
    """Frames per second for video rendering."""
    checkpoint_interval: int = 10
    """save a checkpoint every N iterations"""


@dataclass(frozen=True)
class ModelParamConfig:
    """Configuration for model hyperparameters."""

    in_features: int = 12
    """number of input features in the model"""
    out_features: int = 3
    """number of output features in the model"""
    hidden_features: int = 256
    """number of hidden features in the model"""
    num_layers: int = 4
    """number of layers in the model"""
    has_uv: bool = True
    """whether the loaded mesh has uv coordinates"""
    tex_res: int = 512
    """resolution of the texture in the model"""
    tex_channels: int = 9
    """number of channels in the texture"""
    fourier_enc: bool = True
    """whether to use fourier encoding for the input"""
    L_pos: int = 4
    """number of fourier frequencies for position encoding"""
    L_dir: int = 4
    """number of fourier frequencies for direction encoding"""


@dataclass(frozen=True)
class OptimConfig:
    """Configuration for optimization."""

    max_iter: int = 20
    """maximum number of iterations"""
    warmup_iter: int = 2
    """number of warmup iterations"""
    batch_size: int = 4
    """batch size for training"""
    lr: float = 1e-3
    """learning rate for training"""


@dataclass(frozen=True)
class MainConfig:
    """Main configuration dataclass."""

    file_paths: FilePathConfig = FilePathConfig()
    log_display: LogDisplayConfig = LogDisplayConfig()
    model_param: ModelParamConfig = ModelParamConfig()
    optim: OptimConfig = OptimConfig()


@dataclass
class MainArgs:
    config: tyro.conf.FlagConversionOff[MainConfig] = MainConfig()


def get_config():
    return tyro.cli(MainArgs, default=MainArgs()).config


if __name__ == "__main__":
    from pprint import pprint

    config = get_config()
    pprint(asdict(config))
