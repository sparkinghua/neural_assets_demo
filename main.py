import time

from trainer import Trainer
from model import RNA
from configs import get_config


if __name__ == "__main__":
    config = get_config()

    model = RNA(config.model_param).cuda()

    trainer = Trainer(config, model)

    if config.file_paths.mode == "train":
        time_start = time.time()
        trainer.train()
        time_end = time.time()
        delay = time.strftime("%H:%M:%S", time.gmtime(time_end - time_start))
        print(f"Time: {delay}")
        trainer.logger.info(f"Time: {delay}")

        trainer.show_and_save()
    else:
        psnr = trainer.test()
        trainer.logger.info(f"PSNR: {psnr}")
        print(f"PSNR: {psnr}")
        trainer.show_result()
