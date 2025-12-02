import random
import codecs as cs
from rich.progress import track
from os.path import join as pjoin
import numpy as np

from .dataset_t2m_base import Text2MotionDatasetBaseDist, Text2MotionDatasetBase

class MotionDatasetVQVAE(Text2MotionDatasetBase):
    def __init__(
        self,
        motion_feat_path,
        text_path,
        cot_path,
        split_path,
        split,
        mean,
        std,
        min_motion_length,
        max_motion_length,
        unit_length,
        fps,
        win_size,
        debug=False,
        **kwargs,
    ):
        super().__init__(motion_feat_path, text_path, cot_path, split_path, split, mean, std, 
                         min_motion_length, max_motion_length, unit_length, fps, debug, **kwargs)

        # Filter out the motions that are too short
        self.window_size = win_size
        name_list = list(self.name_list)
        # print(f"{name_list=}")
        # exit(0)
        for name in self.name_list:
            motion = self.data_dict[name]["motion"]
            if motion.shape[0] < self.window_size:
                name_list.remove(name)
                self.data_dict.pop(name)
        self.name_list = name_list

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length = data["motion"], data["length"]

        # Crop into window size
        idx = random.randint(0, motion.shape[0] - self.window_size)
        motion = motion[idx:idx + self.window_size]
        
        # Z Normalization
        motion = (motion - self.mean) / self.std

        # return name, motion, m_length, m_tokens, m_tokens_len, caption, sent_len, tokens, word_embeddings, pos_one_hots, all_captions, cot, task
        return name, motion, m_length, None, None, None, None, None, None, None, None, None, None


def unit_test():
    test_dataset = MotionDatasetVQVAE(
        motion_feat_path = 'datasets/motionx/data/motion_data/vectors_623',
        text_path = 'datasets/motionx/data/texts/semantic_labels',
        cot_path = 'datasets/motionx/data/texts/cot/v3',
        split_path = 'datasets/motionx/data/split',
        split = 'test',
        mean = 0.0,
        std = 1.0,
        min_motion_length = 40,
        max_motion_length = 400,
        unit_length = 4,
        fps = 30,
        win_size = 64,
        debug=True,
    )
    print (len(test_dataset))
    print (test_dataset[10])
    
if __name__ == "__main__":
    unit_test()