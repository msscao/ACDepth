# Adapted from https://github.com/md4all/md4all/blob/main/test_simple.py
import argparse
import torch
import pytorch_lightning as pl
from config.config import get_cfg
from data.custom_dataset import CustomDataModule
from trainer import ACDepth


if __name__ == '__main__':
    # Parser configurations
    parser = argparse.ArgumentParser(description="Test ACDepth loading a model checkpoint and predicting the depth of "
                                                 "an arbitrary input image.")
    parser.add_argument("--config",default='./config/test_ACDepth_nuscenes.yaml', type=str, help="Path to configuration file")
    parser.add_argument("--image_paths",default='', type=str, nargs='+', help="Path to input image")
    parser.add_argument("--daytimes", type=str, help="Daytime of the image, if it is unknown we do not perform time "
                                                    "dependent normalization; instead we perform image-wise "
                                                    "normalization (Only needs to be added if we use daytime dependent "
                                                    "normalization)")
    parser.add_argument("--output_path", default='./output_vis', type=str, help="Path where the output depth map should be stored")

    # Parse and prepare necessary information
    args = parser.parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    image_paths = args.image_paths
    daytimes = args.daytimes
    cfg.EVALUATION.SAVE.QUANTITATIVE_RES_PATH = args.output_path
    cfg.EVALUATION.SAVE.QUALITATIVE_RES_PATH = args.output_path

    # Use minimalistic dataset
    dm = CustomDataModule(cfg, image_paths, daytimes)

    # Check if checkpoint path is specified
    if cfg.LOAD.CHECKPOINT_PATH is None:
        raise AssertionError("For prediction you need to specify a path to a checkpoint")

    # Load model checkpoint here manually as it does not work along with trainer.predict
    model = ACDepth.load_from_checkpoint(cfg.LOAD.CHECKPOINT_PATH, cfg=cfg, is_train=False)

    # Configure trainer
    trainer = pl.Trainer(
        accelerator=cfg.SYSTEM.ACCELERATOR,
        devices=cfg.SYSTEM.DEVICES,
        precision=cfg.SYSTEM.PRECISION,
        logger=False,
        deterministic=cfg.SYSTEM.DETERMINISTIC or cfg.SYSTEM.DETERMINISTIC_ALGORITHMS,
        benchmark=cfg.SYSTEM.BENCHMARK
    )

    # Enable warn_only since some modules do not have a deterministic algorithm implemented
    if cfg.SYSTEM.DETERMINISTIC or cfg.SYSTEM.DETERMINISTIC_ALGORITHMS:
        torch.use_deterministic_algorithms(mode=True, warn_only=True)

    # Start prediction
    trainer.predict(model, dm)
