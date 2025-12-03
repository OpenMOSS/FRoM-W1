import os
from omegaconf import OmegaConf

import torch
import pytorch_lightning as pl

from hGPT.config import parse_args, instantiate_from_config
from hGPT.data.build_data import build_data
from hGPT.models.build_model import build_model
from hGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae
from hGPT.logger import create_logger
from hGPT.callback import build_callbacks

# Disable huggingface tokenizer deadlock warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')

def main(phase = "train"):
    # Configs
    cfg = parse_args(phase)
    
    # Seed
    pl.seed_everything(cfg.SEED_VALUE, workers=True)
    
    # Logger
    logger = create_logger(cfg, phase)
    logger.info(OmegaConf.to_yaml(cfg))
    
    pl_loggers = [] # Metric Logger
    for loggerName in cfg.LOGGER.TYPE:
        if loggerName == 'tenosrboard' or cfg.LOGGER.WANDB.params.project:
            pl_logger = instantiate_from_config(
                eval(f'cfg.LOGGER.{loggerName.upper()}'))
            pl_loggers.append(pl_logger)
        else:
            raise NotImplementedError

    # Callbacks
    callbacks = build_callbacks(cfg, phase, logger)
    logger.info("Callbacks initialized")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))
    
    # Model
    model = build_model(cfg, datamodule)
    logger.info("model {} loaded".format(cfg.model.target))
    
    ## Strict load pretrianed model
    if cfg.TRAIN.PRETRAINED:
        load_pretrained(cfg, model, logger)
    
    ## Strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger)

    # Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=cfg.FOLDER_EXP,
        max_epochs=cfg.TRAIN.END_EPOCH,
        precision=cfg.TRAIN.PRECISION,
        accumulate_grad_batches=cfg.TRAIN.ACCUMULATE_GRAD_BATCHES,
        logger=pl_loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        # https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html
        # strategy="ddp_find_unused_parameters_true"
        # if len(cfg.DEVICE) > 1 else 'auto',
        strategy=cfg.TRAIN.STRATEGY,
        benchmark=False,
        # deterministic=False,
        # RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation, 
        # but you set 'torch.use_deterministic_algorithms(True)'. 
        # You can turn off determinism just for this operation
        deterministic=False,
    )
    logger.info("Trainer initialized")

    # Pytorch 2.0 Compile
    # if torch.__version__ >= "2.0.0":
    #     model = torch.compile(model, mode="reduce-overhead")
    # model = torch.compile(model)

    # Lightning Fitting
    if cfg.TRAIN.RESUME:
        trainer.fit(model,
                    datamodule=datamodule,
                    ckpt_path=cfg.TRAIN.PRETRAINED)
    else:
        trainer.fit(model, datamodule=datamodule)

    # Training ends
    logger.info(
        f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Training ends!")


if __name__ == "__main__":
    main()
