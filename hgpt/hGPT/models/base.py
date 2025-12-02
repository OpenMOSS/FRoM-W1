import os
from pathlib import Path
from collections import OrderedDict
from os.path import join as pjoin
import logging

import numpy as np
import torch
from pytorch_lightning import LightningModule

from hGPT.metrics import BaseMetrics
from hGPT.config import get_obj_from_str

class BaseModel(LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cfg = cfg
        self.configure_metrics()

        # Ablation
        self.test_step_outputs = []
        self.times = []
        self.rep_i = 0

    ## OPTIMIZER
    def configure_optimizers(self):
        # Optimizer
        optim_target = self.hparams.cfg.TRAIN.OPTIM.target
        if len(optim_target.split('.')) == 1:
            optim_target = 'torch.optim.' + optim_target
        optimizer = get_obj_from_str(optim_target)(
            params=self.parameters(), **self.hparams.cfg.TRAIN.OPTIM.params)

        # Scheduler
        scheduler_target = self.hparams.cfg.TRAIN.LR_SCHEDULER.target
        if len(scheduler_target.split('.')) == 1:
            scheduler_target = 'torch.optim.lr_scheduler.' + scheduler_target
        lr_scheduler = get_obj_from_str(scheduler_target)(
            optimizer=optimizer, **self.hparams.cfg.TRAIN.LR_SCHEDULER.params)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    # STEP
    def training_step(self, batch, batch_idx):
        return self.allsplit_step(self.cfg.TRAIN.SPLIT, batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step(self.cfg.EVAL.SPLIT, batch, batch_idx)

    def test_step(self, batch, batch_idx):
        outputs = self.allsplit_step(self.cfg.TEST.SPLIT, batch, batch_idx)
        self.test_step_outputs.append(outputs)
        return outputs

    # EPOCH
    def on_train_epoch_end(self):
        # Log steps and losses
        dico = self.step_log_dict()
        # Log losses
        dico.update(self.loss_log_dict('train'))
        # Write to log only if not sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def on_validation_epoch_end(self):
        # Log steps and losses
        dico = self.step_log_dict()
        # Log losses
        dico.update(self.loss_log_dict('train'))
        dico.update(self.loss_log_dict('val'))
        # Log metrics
        dico.update(self.metrics_log_dict())
        # Write to log only if not sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def on_test_epoch_end(self):
        # Log metrics
        dico = self.metrics_log_dict()
        # print(dico)
        metric_names = dico.keys()
        save_flag = True
        # exit(0)
        # Write to log only if not sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)
        for name in metric_names:
            if 'MultiModality' in name:
                save_flag = False
        if save_flag:
            self.save_npy(self.test_step_outputs)
        self.rep_i = self.rep_i + 1
        # Free up the memory
        self.test_step_outputs.clear()

    # LOG
    def step_log_dict(self):
        return {
            "epoch": float(self.trainer.current_epoch),
            "step": float(self.trainer.current_epoch)
        }

    def loss_log_dict(self, split: str):
        losses = self._losses['losses_' + split]
        loss_dict = losses.compute(split)
        return loss_dict

    def metrics_log_dict(self):
        # For TM2TMetrics MM
        if self.trainer.datamodule.is_mm and "TM2TMetrics" in self.hparams.metrics_dict:
            metrics_dicts = ['MMMetrics']
        else:
            metrics_dicts = self.hparams.metrics_dict

        # Compute all metrics
        metrics_log_dict = {}
        for metric in metrics_dicts:
            metrics_dict = getattr(
                self.metrics,
                metric).compute(sanity_flag=self.trainer.sanity_checking)
            metrics_log_dict.update({
                f"Metrics/{metric}": value.item()
                for metric, value in metrics_dict.items()
            })

        return metrics_log_dict

    def preprocess_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        
        metric_state_dict = self.metrics.state_dict()
        loss_state_dict = self._losses.state_dict()

        for k, v in metric_state_dict.items():
            new_state_dict['metrics.' + k] = v

        for k, v in loss_state_dict.items():
            new_state_dict['_losses.' + k] = v

        for k, v in state_dict.items():
            if '_losses' not in k and 'Metrics' not in k:
                new_state_dict[k] = v

        return new_state_dict

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = self.preprocess_state_dict(state_dict)
        super().load_state_dict(new_state_dict, strict)

    # METRICS
    def configure_metrics(self):
        self.metrics = BaseMetrics(datamodule=self.datamodule, **self.hparams)

    def save_npy(self, outputs):
        cfg = self.hparams.cfg
        output_dir = Path(
            os.path.join(
                cfg.FOLDER,
                str(cfg.model.target.split('.')[-2].lower()),
                str(cfg.NAME),
                "samples_" + cfg.TIME,
            ))
        
        if cfg.TEST.SAVE_PREDICTIONS:
            names = [i[0] for i in outputs] 
            feats_gt = [i[1] for i in outputs]
            feats_rst = [i[2] for i in outputs]
            joints_gt = [i[3] for i in outputs]
            joints_rst = [i[4] for i in outputs]
            lengths = [i[5] for i in outputs]
            texts = [i[6] for i in outputs]
            cots_ref = [i[7] for i in outputs]
            cots_rst = [i[8] for i in outputs]
            
            if cfg.TEST.DATASETS[0].lower() in ["humanml3d", "motionx"]:
                for i in range(len(joints_rst)): # i: batch size idx
                    for bid in range(
                            min(cfg.TEST.BATCH_SIZE, joints_rst[i].shape[0])):
                        # name
                        tmp_name = names[i][bid]

                        # gt joints
                        gt_joints = joints_gt[i][bid][:lengths[i][bid]].cpu(
                        ).numpy()
                        npypath = output_dir / f"{tmp_name}_joints_gt.npy"
                        npypath_parent = npypath.parent
                        npypath_parent.mkdir(parents=True, exist_ok=True)
                        np.save(npypath, gt_joints)

                        # gen joints
                        gen_joints = joints_rst[i][bid][:lengths[i][bid]].cpu(
                        ).numpy()
                        npypath = output_dir / f"{tmp_name}_joints_pred_{self.rep_i}.npy"
                        npypath_parent = npypath.parent
                        npypath_parent.mkdir(parents=True, exist_ok=True)
                        np.save(npypath, gen_joints)
                        
                        # gt feats
                        feats = feats_gt[i][bid][:lengths[i][bid]].cpu(
                        ).numpy()
                        npypath = output_dir / f"{tmp_name}_feats_gt.npy"
                        npypath_parent = npypath.parent
                        npypath_parent.mkdir(parents=True, exist_ok=True)
                        np.save(npypath, feats)
                        
                        # save pred feats
                        gen_feats = feats_rst[i][bid][:lengths[i][bid]].cpu(
                        ).numpy()
                        npypath = output_dir / f"{tmp_name}_feats_pred_{self.rep_i}.npy"
                        npypath_parent = npypath.parent
                        npypath_parent.mkdir(parents=True, exist_ok=True)
                        np.save(npypath, gen_feats)

                        # save caption
                        caption = texts[i][bid]
                        with open(output_dir / f"{tmp_name}_caption.txt", "a") as f:
                            f.write(f"{caption}\n")
                        
                        # cot
                        if cots_ref[i][bid] != '':
                            cot_ref = cots_ref[i][bid]
                            with open(output_dir / f"{tmp_name}_cot_gt.txt", "a") as f:
                                f.write(f"{cot_ref}\n")
                                    
                            cot_rst = cots_rst[i][bid]
                            with open(output_dir / f"{tmp_name}_cot_pred_{self.rep_i}.txt", "a") as f:
                                f.write(f"{cot_rst}\n")     
            else:
                raise NotImplementedError