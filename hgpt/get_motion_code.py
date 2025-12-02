import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import pytorch_lightning as pl

from hGPT.config import parse_args
from hGPT.data.build_data import build_data
from hGPT.models.build_model import build_model
from hGPT.utils.load_checkpoint import load_pretrained_vae

def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1 # BATCH_SIZE == 1 --> no padding
    cfg.EVAL.BATCH_SIZE = 1
    cfg.TEST.BATCH_SIZE = 1

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    datasets = build_data(cfg, phase='token')
    print("datasets module initialized")
    
    output_dir = datasets.motion_token_path
    os.makedirs(output_dir, exist_ok=True)

    # create model
    model = build_model(cfg, datasets)
    print("model loaded")

    # Strict load vae model
    assert cfg.TRAIN.PRETRAINED_VAE is not None
    load_pretrained_vae(cfg, model)

    if cfg.ACCELERATOR == "gpu":
        model = model.to('cuda')

    idx = 0
    for batch in tqdm(datasets.train_dataloader(),
                      desc=f'train motion tokenize'):
        name = batch['name']
        pose = batch['motion']
        pose = pose.cuda().float()

        if pose.shape[1] == 0:
            print ("name:\n", name)
            print ("pose:\n", pose)
            continue
        
        target, _ = model.vae.encode(pose)
        target = target.to('cpu').numpy()

        target_path = os.path.join(output_dir, name[0] + '.npy')
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path, target)
        if idx % 1000 == 0:
            print (f"idx {idx}, length: {len(target[0])}")
        idx += 1

    idx = 0
    for batch in tqdm(datasets.val_dataloader(),
                      desc=f'val motion tokenize'):
        name = batch['name']
        pose = batch['motion']
        pose = pose.cuda().float()

        if pose.shape[1] == 0:
            print ("name:\n", name)
            print ("pose:\n", pose)
            continue
        
        target, _ = model.vae.encode(pose)
        target = target.to('cpu').numpy()

        target_path = os.path.join(output_dir, name[0] + '.npy')
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path, target)
        if idx % 1000 == 0:
            print (f"idx {idx}, length: {len(target[0])}")
        idx += 1
        
    idx = 0
    for batch in tqdm(datasets.test_dataloader(),
                      desc=f'test motion tokenize'):
        name = batch['name']
        pose = batch['motion']
        pose = pose.cuda().float()

        if pose.shape[1] == 0:
            print ("name:\n", name)
            print ("pose:\n", pose)
            continue
        
        target, _ = model.vae.encode(pose)
        target = target.to('cpu').numpy()

        target_path = os.path.join(output_dir, name[0] + '.npy')
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path, target)
        if idx % 1000 == 0:
            print (f"idx {idx}, length: {len(target[0])}")
        idx += 1
        
    print(
        f'Motion tokenization done, the motion tokens are saved to {output_dir}'
    )


if __name__ == "__main__":
    main()
