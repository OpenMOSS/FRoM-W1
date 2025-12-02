import os
import time
import random
import json
from os.path import join as pjoin

import torch

from hGPT.config import instantiate_from_config
from hGPT.models.base import BaseModel
from hGPT.losses.hgpt import GPTLosses

class HumanoidGPT(BaseModel):
    def __init__(self,
                 cfg,
                 datamodule,
                 lm,
                 motion_vae,
                 codebook_size,
                 condition,
                 metrics_dict,
                 **kwargs):
        
        self.save_hyperparameters(ignore='datamodule', logger=False)
        self.datamodule = datamodule
        super().__init__(cfg=cfg)

        # Instantiate motion tokenizer
        if motion_vae != None:
            self.vae = instantiate_from_config(motion_vae)
                
        # Instantiate motion-language model
        if cfg.TRAIN.STAGE == "vae":
            self.lm = None
        else:
            self.lm = instantiate_from_config(lm)

        # Freeze the motion tokenizer for lm training
        if 'lm' in self.hparams.stage:
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False

        # Instantiate the losses
        self._losses = torch.nn.ModuleDict({
            split: GPTLosses(cfg, self.hparams.stage, self.datamodule.njoints)
            for split in ["losses_train", "losses_test", "losses_val"]
        })

        # Data transform
        self.feats2joints = datamodule.feats2joints
        
    def train_lm_forward(self, batch):
        names = batch['name']
        texts = batch["text"]
        tokens_ref = batch["m_tokens"]
        tokens_len = batch["m_tokens_len"]
        cot = batch['cot']
        tasks = batch["tasks"]

        outputs = self.lm(texts, tokens_ref, tokens_len, tasks, cot)
        return {'outputs': outputs}

    @torch.no_grad()
    def val_t2m_forward(self, batch):
        names = batch["name"]
        feats_ref = batch["motion"]
        lengths = batch["m_length"]
        texts = batch["text"]
        cot_ref = batch["cot"]
        tasks = []
        
        if self.trainer.datamodule.is_mm:
            raise NotImplementedError
            texts = texts * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS
        
        # if self.trainer.datamodule.is_mm:
            # instructions = pjoin(self.datamodule.hparams.data_root, 'instructions', 
            #                      'template_instructions.json')
            # raise NotImplementedError

        if self.hparams.cfg.DATASET.TASK_PATH:
            instructions = pjoin(self.hparams.cfg.DATASET.TASK_PATH)
            instructions = json.load(open(instructions, 'r'))
            tasks = [instructions["Text-to-Motion"]["t2m"]] * len(texts)

        min_len = lengths.copy()
        # Forward
        outputs, cleaned_text = self.lm.generate_conditional(texts,
                                               lengths=lengths,
                                               stage='test',
                                               cot=cot_ref,
                                               tasks=tasks)

        # Motion Decode
        feats_rst = torch.zeros_like(feats_ref)
                
        for i in range(len(texts)):
            outputs[i] = torch.clamp(outputs[i],
                                     0,
                                     self.hparams.codebook_size - 1,
                                     out=None)
            if len(outputs[i]) > 1:
                motion = self.vae.decode(outputs[i])
            else:
                motion = torch.zeros_like(feats_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], lengths[i])
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :lengths[i]]

        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": min_len,
            "text": texts,
            "name": names,
            "cot_ref": cot_ref,
            "cot_rst": cleaned_text
        }
        return rs_set
    
    def train_vae_forward(self, batch):
        # batch detach
        feats_ref = batch["motion"]
        joints_ref = self.feats2joints(feats_ref)
        
        # motion encode & decode
        feats_rst, loss_commit, perplexity = self.vae(feats_ref)
        joints_rst = self.feats2joints(feats_rst)
        
        # return set
        rs_set = {
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "loss_commit": loss_commit,
            "perplexity": perplexity,
        }
        return rs_set

    @torch.no_grad()
    def val_vae_forward(self, batch, split="train"):
        # Detach batch
        names = batch["name"]
        feats_ref = batch["motion"]
        lengths = batch["m_length"]
        texts = batch["text"]
        batch_size = len(names)
        
        # Repeat for multimodal evaluation
        if self.trainer.datamodule.is_mm:
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS

        # Motion encode & decode
        feats_rst = torch.zeros_like(feats_ref)

        for i in range(len(feats_ref)):
            if lengths[i] == 0:
                continue
            feats_pred, _, _ = self.vae(feats_ref[i:i + 1, :lengths[i]])
            feats_rst[i:i + 1, :feats_pred.shape[1], :] = feats_pred
        
        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        # Return set
        rs_set = {
            "name": names,
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": lengths,
            "text": texts,
            "cot_ref": [''] * batch_size,
            "cot_rst": [''] * batch_size
        }
        
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        # Compute the losses
        loss = None
        if split in ["train"]:
            if self.hparams.stage == "vae":
                rs_set = self.train_vae_forward(batch)
                loss = self._losses['losses_' + split].update(rs_set)
            elif self.hparams.stage in ["lm_pretrain", "lm_instruct"]:
                rs_set = self.train_lm_forward(batch)
                loss = self._losses['losses_' + split].update(rs_set)
            else:
                raise NotImplementedError
            
        elif split in ["val", "test", "test_small"]:
            # Compute the metrics
            if self.hparams.stage == "vae":
                rs_set = self.val_vae_forward(batch, split)
                
            elif self.hparams.stage in ["lm_pretrain", "lm_instruct"]:
                if self.hparams.task == "t2m":
                    rs_set = self.val_t2m_forward(batch)
                else:
                    raise NotImplementedError

            if self.hparams.task not in ["m2t"]:
                if self.trainer.datamodule.is_mm:
                    metrics_dicts = ['MMMetrics']
                else:
                    metrics_dicts = self.hparams.metrics_dict
                    
                if self.hparams.task not in ['pred', 'inbetween'] and 'PredMetrics' in metrics_dicts:
                    metrics_dicts.remove('PredMetrics')

                for metric in metrics_dicts:
                    lengths = batch['m_length']
                    if metric == "TemosMetric":
                        getattr(self.metrics,
                                metric).update(rs_set["joints_rst"],
                                            rs_set["joints_ref"], lengths)
                    elif metric == "TM2TMetrics":
                        if self.hparams.stage in [
                                "lm_instruct", "lm_pretrain"
                        ]:
                            word_embs = batch['word_embs']
                            pos_ohot = batch['pos_ohot']
                            text_lengths = batch['text_len']
                            if self.trainer.datamodule.is_mm:
                                word_embs = word_embs.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                                pos_ohot = pos_ohot.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                                text_lengths = text_lengths.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                        else:
                            word_embs = None
                            pos_ohot = None
                            text_lengths = None

                        getattr(self.metrics, metric).update(
                            feats_ref=rs_set["m_ref"],
                            feats_rst=rs_set["m_rst"],
                            lengths_ref=lengths,
                            lengths_rst=rs_set['length'],
                            word_embs=word_embs,
                            pos_ohot=pos_ohot,
                            text_lengths=text_lengths,
                        )
                    elif metric == "UncondMetrics":
                        getattr(self.metrics, metric).update(
                            recmotion_embeddings=rs_set["lat_rm"],
                            gtmotion_embeddings=rs_set["lat_m"],
                            lengths=lengths,
                        )
                    elif metric == "MRMetrics":
                        getattr(self.metrics,
                                metric).update(rs_set["joints_rst"],
                                            rs_set["joints_ref"], lengths)
                    elif metric == "PredMetrics":
                        getattr(self.metrics,
                                metric).update(rs_set["joints_rst"],
                                            rs_set["joints_ref"], lengths)
                    elif metric == "MMMetrics":
                        getattr(self.metrics,
                                metric).update(rs_set["m_rst"],
                                            rs_set['length'])
                    else:
                        raise TypeError(f"Not support this metric {metric}")

            else:
                raise NotImplementedError

            # return forward output rather than loss during test
            if 'test' in split:
                if self.hparams.task in ["vae", "t2m"]:
                    return rs_set["name"], rs_set['m_ref'], rs_set['m_rst'], \
                        rs_set["joints_ref"], rs_set["joints_rst"], rs_set["length"], \
                        rs_set["text"], rs_set["cot_ref"], rs_set["cot_rst"]
                else:
                    raise NotImplementedError
        return loss