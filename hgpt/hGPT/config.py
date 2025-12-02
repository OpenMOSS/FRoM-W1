import os
import importlib
import glob
from argparse import ArgumentParser
from os.path import join as pjoin
from omegaconf import OmegaConf

def get_obj_from_str(string, reload=False):
    """
    Get object from string
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    Instantiate object from config
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_module_config(cfg, filepath="./configs"):
    """
    Load yaml config files from subfolders
    """
    yamls = glob.glob(pjoin(filepath, '*', '*.yaml'))
    yamls = [y.replace(filepath, '') for y in yamls]
    for yaml in yamls:
        nodes = yaml.replace('.yaml', '').replace(os.sep, '.')
        nodes = nodes[1:] if nodes[0] == '.' else nodes
        OmegaConf.update(cfg, nodes, OmegaConf.load(filepath + yaml))
    return cfg

def resume_config(cfg: OmegaConf):
    """
    Resume model and wandb
    """
    if cfg.TRAIN.RESUME:
        resume = cfg.TRAIN.RESUME
        if os.path.exists(resume):
            # Checkpoints
            cfg.TRAIN.PRETRAINED = pjoin(resume, "checkpoints", "last.ckpt")
            # Wandb
            wandb_files = os.listdir(pjoin(resume, "wandb", "latest-run"))
            wandb_run = [item for item in wandb_files if "run-" in item][0]
            cfg.LOGGER.WANDB.params.id = wandb_run.replace("run-","").replace(".wandb", "")
        else:
            raise ValueError("Resume path is not right.")

    return cfg

def parse_args(phase="train"):
    """
    Parse arguments and load config files
    """
    parser = ArgumentParser()
    
    # Training Options
    group = parser.add_argument_group("Training options")
    group.add_argument(
        "--cfg_assets",
        type=str,
        required=True,
        help="config file for asset paths",
    )  
    group.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="config file",
    )
    if phase in ["train", "test"]:
        group.add_argument("--task",
                           type=str,
                           required=False,
                           help="evaluation task type")
        group.add_argument("--num_nodes",
                           type=int,
                           required=False,
                           help="number of nodes")
        group.add_argument("--device",
                           type=int,
                           nargs="+",
                           required=False,
                           help="training device")
        group.add_argument("--batch_size",
                           type=int,
                           required=False,
                           help="training batch size")
        group.add_argument("--nodebug",
                           action="store_true",
                           required=False,
                           help="debug or not")
    elif phase == "demo":
        group.add_argument("--task",
            type=str,
            required=False,
            help="evaluation task type")
        group.add_argument(
            "--example",
            type=str,
            required=False,
            help="input text and lengths with txt format",
        )
        group.add_argument(
            "--out_dir",
            type=str,
            required=False,
            help="output dir",
        )
    else:
        raise NotImplementedError
    params = parser.parse_args()
    
    # Load yaml config files
    OmegaConf.register_new_resolver("eval", eval)
    cfg_assets = OmegaConf.load(params.cfg_assets)
    cfg_exp = OmegaConf.load(params.cfg)
    cfg_exp = get_module_config(cfg_exp, cfg_assets.ARCHS_FOLDER)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)

    # Update config with arguments
    if phase in ["train", "test"]:
        cfg.TRAIN.BATCH_SIZE = params.batch_size if params.batch_size else cfg.TRAIN.BATCH_SIZE
        cfg.DEVICE = params.device if params.device else cfg.DEVICE
        cfg.NUM_NODES = params.num_nodes if params.num_nodes else cfg.NUM_NODES
        cfg.model.params.task = params.task if params.task else cfg.model.params.task
        cfg.DEBUG = not params.nodebug if params.nodebug is not None else cfg.DEBUG
    elif phase == "demo":
        cfg.DEMO.EXAMPLE = params.example
        cfg.DEMO.TASK = params.task
        cfg.TEST.FOLDER = params.out_dir if params.out_dir else cfg.TEST.FOLDER
        os.makedirs(cfg.TEST.FOLDER, exist_ok=True)
    else:
        raise NotImplementedError

    # Debug mode
    if cfg.DEBUG:
        cfg.DEVICE = [0]
        cfg.NAME = "debug--" + cfg.NAME
        # cfg.LOGGER.VAL_EVERY_STEPS = 1
        
    # Resume config
    cfg = resume_config(cfg)
    return cfg
