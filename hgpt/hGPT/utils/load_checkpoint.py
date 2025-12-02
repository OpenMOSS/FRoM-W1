import torch

def load_pretrained(cfg, model, logger=None, phase="train"):
    if logger is not None:
        logger.info(f"Loading pretrain model from {cfg.TRAIN.PRETRAINED}")
        
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
    elif phase == "demo":
        ckpt_path = cfg.DEMO.CHECKPOINTS
    
    state_dict = torch.load(ckpt_path, map_location="cpu",weights_only=False)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    elif 'module' in state_dict.keys():
        state_dict = state_dict['module']
    else:
        raise NotImplementedError
    
    model.load_state_dict(state_dict, strict=True)
    return model


def load_pretrained_vae(cfg, model, logger=None):
    if logger is not None:
        logger.info(f"Loading pretrain vae from {cfg.TRAIN.PRETRAINED_VAE}")
        
    state_dict = torch.load(cfg.TRAIN.PRETRAINED_VAE,
                            map_location="cpu", weights_only=False)['state_dict']

    from collections import OrderedDict
    vae_dict = OrderedDict()
    for k, v in state_dict.items():
        if "motion_vae" in k:
            name = k.replace("motion_vae.", "")
            vae_dict[name] = v
        elif "vae" in k:
            name = k.replace("vae.", "")
            vae_dict[name] = v
            
    if hasattr(model, 'vae'):
        model.vae.load_state_dict(vae_dict, strict=True)
    else:
        model.motion_vae.load_state_dict(vae_dict, strict=True)
    
    return model
