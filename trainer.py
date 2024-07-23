from dataclasses import asdict
import json
import pathlib
import numpy as np
import torch
from matplotlib import pyplot as plt
import imageio
from tqdm import tqdm
import logging
import time
import os

import util
from model import RNA
from configs import MainConfig
from data_loader import load_data
from render import Render


class Trainer:
    def __init__(self, config: MainConfig, model: RNA):
        self.config = config
        self.model = model
        self.render = Render(config)
        self.dataloader, self.metadata = load_data(config)

        self.max_iter = config.optim.max_iter
        self.warmup_iter = config.optim.warmup_iter
        self.batch_size = config.optim.batch_size
        self.log_interval = config.log_display.log_interval
        self.display_interval = config.log_display.display_interval
        self.display_res = config.log_display.display_res
        self.fps = config.log_display.fps
        self.checkpoint_interval = config.log_display.checkpoint_interval
        self.train_samples = self.metadata["train"]["samples"]
        self.valid_samples = self.metadata["valid"]["samples"]
        self.test_samples = self.metadata["test"]["samples"]
        self.train_resolution = self.metadata["train"]["resolution"]

        self.train_psnr = []
        self.valid_psnr = []
        self.frames = []
        self.init_iter = 0

        workdir = f"{pathlib.Path(__file__).absolute().parents[0]}"
        self.outputdir = f"{workdir}/{config.file_paths.output_dir}/" + time.strftime("%m%d-%H%M%S")
        os.makedirs(self.outputdir, exist_ok=True)
        logfile = f"{self.outputdir}/train.log"
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler(logfile)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        self.logger.addHandler(handler)

        with open(f"{self.outputdir}/config.json", "w") as f:
            json.dump(asdict(config), f, indent=4)

        self.lr = config.optim.lr
        self.weight_decay = config.optim.weight_decay
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=(self.max_iter - self.warmup_iter) * self.train_samples / self.batch_size, eta_min=1e-7
        )
        self.loss_fn = torch.nn.MSELoss()

        if config.file_paths.resume is not None:
            checkpoint = f"{workdir}/{config.file_paths.output_dir}/" + config.file_paths.resume
            self.load_checkpoint(checkpoint)

    def train(self):
        imgloss_avg = []
        self.logger.info("Start training...")
        for iter in range(self.init_iter, self.max_iter):
            print(f"iter={iter+1}/{self.max_iter}")
            self.model.train()
            i = -self.batch_size
            for color, light_pos, camera_pos, mvp, emission, ambient in tqdm(self.dataloader["train"]):
                i += self.batch_size
                color_opt = self.render.render(camera_pos, light_pos, mvp, emission, ambient, self.train_resolution, self.model)
                loss = self.loss_fn(color, color_opt)
                self.optimizer.zero_grad()
                loss.backward()
                if iter < self.warmup_iter:
                    self.optimizer.param_groups[0]["lr"] = (
                        self.lr * (iter * self.train_samples + i + 1) / (self.warmup_iter * self.train_samples)
                    )
                    self.optimizer.step()
                else:
                    self.optimizer.step()
                    self.scheduler.step()

                imgloss_avg.append(loss.detach().cpu().numpy())

                if self.log_interval and (i % self.log_interval == 0):
                    imgloss_val, imgloss_avg = np.mean(np.asarray(imgloss_avg, np.float32)), []
                    psnr_val = -10.0 * np.log10(imgloss_val)
                    s = "iter=%d, sample=%d, lr=%g, train_psnr=%f" % (
                        iter,
                        i,
                        self.optimizer.param_groups[0]["lr"],
                        psnr_val,
                    )
                    self.train_psnr.append(psnr_val)
                    self.logger.info(s)
                display_image = self.display_interval and (i % self.display_interval == 0)
                if display_image:
                    self.model.eval()
                    with torch.no_grad():
                        color_opt = self.render.display_render(self.train_resolution, self.model)
                    result_image = color_opt.detach()[0].cpu().numpy()[::-1]
                    util.display_image(
                        result_image,
                        size=self.display_res,
                        title="%d / %d, %d / %d" % (iter, self.max_iter, i, self.train_samples),
                    )
                    self.frames.append((result_image * 255).astype(np.uint8))
            # Validation
            valid_psnr_val = self.valid()
            self.valid_psnr.append(valid_psnr_val)
            s = "iter=%d, valid_psnr=%f" % (iter, valid_psnr_val)
            self.logger.info(s)
            # Save checkpoint
            save_checkpoint = self.checkpoint_interval and ((iter + 1) % self.checkpoint_interval == 0)
            if save_checkpoint:
                self.save_checkpoint(iter)
        self.logger.info("Training done.")
        test_psnr_val = self.test()
        s = "test_psnr=%f" % (test_psnr_val)
        self.logger.info(s)

    def valid(self) -> float:
        valid_loss_val = 0
        valid_len = 0
        # self.model.eval()
        self.logger.info("Validation...")
        for color, light_pos, camera_pos, mvp, emission, ambient in self.dataloader["valid"]:
            with torch.no_grad():
                color_opt = self.render.render(camera_pos, light_pos, mvp, emission, ambient, self.train_resolution, self.model)
                valid_loss_val += self.loss_fn(color, color_opt)
            valid_len += 1
        valid_loss_val /= valid_len
        valid_loss_val = valid_loss_val.detach().cpu().numpy()
        valid_psnr_val = -10.0 * np.log10(valid_loss_val)
        return valid_psnr_val

    def test(self) -> float:
        test_loss_val = 0
        test_len = 0
        self.model.eval()
        self.logger.info("Testing...")
        for color, light_pos, camera_pos, mvp, emission, ambient in self.dataloader["test"]:
            with torch.no_grad():
                color_opt = self.render.render(camera_pos, light_pos, mvp, emission, ambient, self.train_resolution, self.model)
                test_loss_val += self.loss_fn(color, color_opt)
            test_len += 1
        test_loss_val /= test_len
        test_loss_val = test_loss_val.detach().cpu().numpy()
        test_psnr_val = -10.0 * np.log10(test_loss_val)
        return test_psnr_val

    def save_checkpoint(self, iter):
        if not os.path.exists(f"{self.outputdir}/checkpoints"):
            os.makedirs(f"{self.outputdir}/checkpoints", exist_ok=True)
        checkpoint_path = f"{self.outputdir}/checkpoints/{iter+1}.pth"
        torch.save(
            {
                "iter": iter,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.init_iter = checkpoint["iter"]
        print(f"Checkpoint loaded from {checkpoint_path}, starting from iter {self.init_iter + 1}")

    def show_and_save(self):
        with imageio.get_writer(f"{self.outputdir}/train.mp4", fps=self.fps) as writer:
            for frame in self.frames:
                writer.append_data(frame)
            self.frames = []
        # imgloss_log = np.asarray(self.imgloss_log, np.float32)
        # imgpsnr_log = np.asarray(self.imgpsnr_log, np.float32)
        # fig = plt.figure(figsize=(12, 6))
        # ax = fig.add_subplot(1, 2, 1)
        # ax.set_title("Image Loss")
        # ax.set_xlabel("Step")
        # ax.set_ylabel("Loss")
        # ax.grid()
        # ax.scatter(range(0, len(imgloss_log) * self.log_interval, self.log_interval), imgloss_log)

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 2, 1)
        ax.set_title("Train PSNR")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("PSNR")
        ax.grid()
        ax.plot(np.linspace(1, self.max_iter, len(self.train_psnr)), self.train_psnr)

        ax = fig.add_subplot(1, 2, 2)
        ax.set_title("Validation PSNR")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("PSNR")
        ax.grid()
        ax.plot(np.linspace(1, self.max_iter, len(self.valid_psnr)), self.valid_psnr)

        fig.savefig(f"{self.outputdir}/psnr.png")
        with open(f"{self.outputdir}/psnr.npy", "wb") as f:
            np.savez_compressed(f, train_psnr=self.train_psnr, valid_psnr=self.valid_psnr)
        plt.show()
