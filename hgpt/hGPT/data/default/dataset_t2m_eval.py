import random
import numpy as np

from .dataset_t2m_base import Text2MotionDatasetBaseDist, Text2MotionDatasetBase

class Text2MotionDatasetEval(Text2MotionDatasetBaseDist):
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
        max_text_len,
        w_vectorizer,
        debug=False,
        mean_eval=None,
        std_eval=None,
        # truncate=False,
        # test_bs=32,
        **kwargs,
    ):
        super().__init__(motion_feat_path, text_path, cot_path, split_path, split, mean, std, min_motion_length,
                         max_motion_length, unit_length, fps, debug, **kwargs)
        self.max_text_len = max_text_len
        self.w_vectorizer = w_vectorizer
        
        # yfgao: WARNING! This is only for MM test
        # if truncate:
        #     raise NotImplementedError
        #     len_dataset = len(self.name_list)
        #     self.name_list = self.name_list[:len_dataset - len_dataset % test_bs]

    def __getitem__(self, idx):
        name = self.name_list[idx]
        # name = '013514'
        
        data = self.data_dict[name]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        
        # print (">>> text list: ", text_list)
        
        # Randomly select a caption
        text_data = random.choice(text_list)
        # print(text_data)
        caption = text_data["caption"]
        tokens = text_data["tokens"]
        cot = text_data["cot"]

        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]
        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        elif len(all_captions) == 2:
            all_captions = all_captions + all_captions[0:1]
        elif len(all_captions) == 1:
            all_captions = all_captions * 3

        # Text
        if len(tokens) < self.max_text_len:
            # pad with "unk"
            # if sent_len == 0:
            #     tokens = ["sos/OTHER"] + ["eos/OTHER"]
            # else:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        
        # Crop the motions in to times of unit_length, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"
        
        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        
        # Z Normalization
        motion = (motion - self.mean) / self.std

        # for eval.
        # if m_length < self.max_motion_length:
        #     motion = np.concatenate([motion,
        #                              np.zeros((self.max_motion_length - m_length, motion.shape[1]), dtype=np.float32)
        #                              ], axis=0)

        # return name, motion, m_length, m_tokens, m_tokens_len, caption, sent_len, "_".join(tokens), word_embeddings, pos_one_hots, all_captions, cot, task
        return name, motion, m_length, None, None, caption, sent_len, "_".join(tokens), word_embeddings, pos_one_hots, all_captions, cot, None


def unit_test():
    from word_vectorizer import WordVectorizer
    w_vectorizer = WordVectorizer('deps/glove/', "our_vab")
    
    test_dataset = Text2MotionDatasetEval(
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
        max_text_len = 100,
        w_vectorizer = w_vectorizer,
        debug=True,
    )
    print (len(test_dataset))
    print (test_dataset[10])
    
if __name__ == "__main__":
    unit_test()