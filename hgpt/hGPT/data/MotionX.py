
import os 
from os.path import join as pjoin
import numpy as np
import torch

from .default import (
    Text2MotionDatasetBaseDist,
    MotionDatasetVQVAE, 
    Text2MotionDatasetTokenDist,
    Text2MotionDatasetTrainDist, 
    Text2MotionDatasetEval,
    Text2MotionDatasetBase,
    Text2MotionDatasetToken,
    Text2MotionDatasetTrain
    )
from .default.word_vectorizer import WordVectorizer
from .base import collate_tensors, BASEDataModule
from .motionx.scripts.motion_process import (process_file, recover_from_ric)

# name, motion, m_length, m_tokens, m_tokens_len, caption, sent_len, "_".join(tokens), 
# word_embeddings, pos_one_hots, all_captions, cot, task
def motionx_collate(batch):
    # Notnone
    notnone_batches = [b for b in batch if b is not None]
    
    # Sort by text length
    EvalFlag = False if notnone_batches[0][6] is None else True
    if EvalFlag:
        notnone_batches.sort(key=lambda x: x[6], reverse=True)
    
    # Name
    adapted_batch = {
        "name": [b[0] for b in notnone_batches]
    }
        
    # Motion
    if notnone_batches[0][1] is not None:
        adapted_batch.update({
            "motion":
            collate_tensors([torch.tensor(b[1]) for b in notnone_batches]),
            "m_length": [b[2] for b in notnone_batches],
        })
        
    # Motion token 
    if notnone_batches[0][3] is not None:
        adapted_batch.update({
            "m_tokens":
            collate_tensors([torch.tensor(b[3]) for b in notnone_batches]),
            "m_tokens_len": [b[4] for b in notnone_batches],
        })
        
    # Text
    if notnone_batches[0][5] is not None:
        adapted_batch.update({
            "text": [b[5] for b in notnone_batches],
            "all_captions": [b[10] for b in notnone_batches],
            "cot": [b[11] for b in notnone_batches]
        })

    # Evaluation related
    if EvalFlag:
        adapted_batch.update({
            "text_len":
            collate_tensors([torch.tensor(b[6]) for b in notnone_batches]),
            "tokens": [b[7] for b in notnone_batches],
            "word_embs":
            collate_tensors(
                [torch.tensor(b[8]) for b in notnone_batches]).float(),
            "pos_ohot":
            collate_tensors(
                [torch.tensor(b[9]) for b in notnone_batches]).float(),
        })
        
    # Tasks
    if notnone_batches[0][12] is not None:
        adapted_batch.update({"tasks": [b[12] for b in notnone_batches]})
        
    return adapted_batch


class MotionXDataModule(BASEDataModule):
    def __init__(self, cfg, **kwargs):
        super().__init__(collate_fn=motionx_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        
        # Basic info of the dataset
        self.name = cfg.DATASET.MOTIONX.NAME
        self.njoints = cfg.DATASET.MOTIONX.NJOINTS
        self.unit_len = cfg.DATASET.MOTIONX.UNIT_LEN
        self.hparams.split_path = cfg.DATASET.MOTIONX.SPLIT_PATH
        self.data_root = cfg.DATASET.MOTIONX.DATA_ROOT
        self.motion_token_path = cfg.DATASET.MOTIONX.MOTION_TOKEN_PATH
        
        # Motion and Text
        self.hparams.motion_feat_path = cfg.DATASET.MOTIONX.MOTION_FEAT_PATH
        self.hparams.text_path = cfg.DATASET.MOTIONX.SEMANTIC_TEXT_PATH
        self.hparams.cot_path = cfg.DATASET.MOTIONX.COT_PATH
        
        mean_std_root = cfg.DATASET.MOTIONX.MEAN_STD_PATH
        self.hparams.mean = np.load(pjoin(mean_std_root, "mean.npy"))
        self.hparams.std = np.load(pjoin(mean_std_root, "std.npy"))
        
        eval_mean_std_root = cfg.DATASET.MOTIONX.EVAL_MEAN_STD_PATH
        self.hparams.mean_eval = np.load(pjoin(eval_mean_std_root, "mean.npy"))
        self.hparams.std_eval = np.load(pjoin(eval_mean_std_root, "std.npy"))
           
        # Length and FPS of the dataset
        self.hparams.min_motion_length = cfg.DATASET.MOTIONX.MIN_MOTION_LEN
        self.hparams.max_motion_length = cfg.DATASET.MOTIONX.MAX_MOTION_LEN
        self.hparams.unit_length = cfg.DATASET.MOTIONX.UNIT_LEN
        self.hparams.fps = cfg.DATASET.MOTIONX.FRAME_RATE

        # Dataset switch
        self.hparams.stage = cfg.TRAIN.STAGE # stage "vae" , "lm_pretrain", "lm_instruct"
        if cfg.TRAIN.STAGE == "vae":
            # print("Here")
            # exit(0)
            self.hparams.win_size = cfg.DATASET.MOTIONX.VAE_WIN_SIZE
            self.hparams.max_text_len = cfg.DATASET.MOTIONX.MAX_TEXT_LEN
            self.hparams.w_vectorizer = WordVectorizer(
                cfg.DATASET.WORD_VERTILIZER_PATH, 'our_vab')
            self.DatasetTrain = MotionDatasetVQVAE
            self.DatasetEval = Text2MotionDatasetEval
        elif cfg.TRAIN.STAGE == "token":
            self.DatasetTrain = Text2MotionDatasetTokenDist
            self.DatasetEval = Text2MotionDatasetTokenDist
            # self.DatasetTrain = Text2MotionDatasetToken
            # self.DatasetEval = Text2MotionDatasetToken
        elif 'lm' in cfg.TRAIN.STAGE:
            self.hparams.task_path = cfg.DATASET.TASK_PATH
            self.hparams.motion_token_path = cfg.DATASET.MOTIONX.MOTION_TOKEN_PATH
            self.hparams.max_text_len = cfg.DATASET.MOTIONX.MAX_TEXT_LEN
            self.hparams.std_text = cfg.DATASET.MOTIONX.STD_TEXT
            # self.hparams.truncate = cfg.DATASET.MOTIONX.TRUNCATE
            # self.hparams.test_bs = cfg.TEST.BATCH_SIZE
            
            self.hparams.w_vectorizer = WordVectorizer(
                cfg.DATASET.WORD_VERTILIZER_PATH, 'our_vab')
            self.DatasetTrain = Text2MotionDatasetTrainDist
            # self.DatasetTrain = Text2MotionDatasetTrain
            self.DatasetEval = Text2MotionDatasetEval
        else:
            raise NotImplementedError
        
        # DEBUG
        self.hparams.debug = cfg.DEBUG
        
    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features
    
    def feats2joints(self, features):
        features = self.denormalize(features)
        return recover_from_ric(features, self.njoints)

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
