from os.path import join as pjoin
from omegaconf import OmegaConf

from hGPT.config import instantiate_from_config

def build_data(cfg, phase="train"):
    data_config = OmegaConf.to_container(cfg.DATASET, resolve=True)
    data_config['params'] = {'cfg': cfg, 'phase': phase}
    
    if isinstance(data_config['target'], str):
        return instantiate_from_config(data_config)
    elif isinstance(data_config['target'], list):
        raise NotImplementedError