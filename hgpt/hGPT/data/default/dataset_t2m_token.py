import random
from tqdm import tqdm
import codecs as cs
from os.path import join as pjoin
from rich.progress import track
import random
from tqdm import tqdm
import codecs as cs
from os.path import join as pjoin
from rich.progress import track
import numpy as np
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

import numpy as np
from torch.utils.data import Dataset

SHOW_INFO=False

class Text2MotionDatasetToken(Dataset):
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
        debug=False,
        **kwargs,
    ):
        # init
        cot_path = cot_path if cot_path != '' else None
        split_file = pjoin(split_path, f'{split}.txt')
        
        self.mean = mean
        self.std = std        
        self.min_motion_length = min_motion_length
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length
        self.fps = fps
            
        # load id list
        self.id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())
            print (f"Id list num: {len(self.id_list)}")
                
        if debug:
            enumerator = enumerate(self.id_list)
            maxdata = 100
            subset = '_tiny'
        else:
            enumerator = enumerate(
                track(
                    self.id_list,
                    f"Loading Dataset {split_file}",
                ))
            maxdata = 1e10
            subset = ''
            
        # load motion
        new_name_list = []
        length_list = []
        data_dict = {}
        for idx, name in enumerator:
            if len(new_name_list) >= maxdata:
                break
            
            try:
                motion = np.load(pjoin(motion_feat_path, name + '.npy'), allow_pickle=True)
            except Exception as e:
                if SHOW_INFO:
                    print (f"Loading data error: {name}. Skipped.")
                continue
            
            if np.isnan(motion).any():
                if SHOW_INFO:
                    print (f"Found nan value: {name}. skiped.")
                continue
            
            if (len(motion)) <  self.min_motion_length or (len(motion) >= self.max_motion_length):
                if SHOW_INFO:
                    print (f"Length out of range: {name}. curr: {len(motion)} min: {self.min_motion_length} max: {self.max_motion_length}. skiped.")
                continue
            
            data_dict[name] = {'motion': motion,
                            'length': len(motion),
                            'name': name}
            new_name_list.append(name)
            length_list.append(len(motion))
            
        self.data_dict = data_dict
        self.name_list = new_name_list
    
    def __len__(self):
        return len(self.name_list)  
        
    def __getitem__(self, idx):
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # name, motion, m_length, m_tokens, m_tokens_len, caption, sent_len, "_".join(tokens), word_embeddings, pos_one_hots, all_captions, cot, task
        return name, motion, m_length, None, None, None, None, None, None, None, None, None, None

class Text2MotionDatasetTokenDist(Dataset):
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
        debug=False,
        num_workers=16,  # 添加多线程参数
        **kwargs,
    ):
        # init
        cot_path = cot_path if cot_path != '' else None
        split_file = pjoin(split_path, f'{split}.txt')
        
        self.mean = mean
        self.std = std        
        self.min_motion_length = min_motion_length
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length
        self.fps = fps
            
        # load id list
        self.id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())
            print(f"Id list num: {len(self.id_list)}")
                
        if debug:
            maxdata = 100
        else:
            maxdata = 1e10
            
        # 使用多线程加载数据
        data_dict = {}
        new_name_list = []
        length_list = []
        
        # 创建线程安全的数据结构
        data_queue = queue.Queue()
        processed_count = 0
        lock = threading.Lock()
        
        def process_single_item(item):
            nonlocal processed_count
            idx, name = item
            
            # 检查是否已达到最大数据量
            with lock:
                if len(new_name_list) >= maxdata:
                    return
            
            try:
                # 加载motion数据
                motion = np.load(pjoin(motion_feat_path, name + '.npy'), allow_pickle=True)
                
                # 数据验证
                if np.isnan(motion).any():
                    if SHOW_INFO:
                        print(f"Found nan value: {name}. skipped.")
                    return
                
                if (len(motion)) < self.min_motion_length or (len(motion) >= self.max_motion_length):
                    if SHOW_INFO:
                        print(f"Length out of range: {name}. curr: {len(motion)} min: {self.min_motion_length} max: {self.max_motion_length}. skipped.")
                    return
                
                # 创建数据项
                data_item = {
                    'motion': motion,
                    'length': len(motion),
                    'name': name
                }
                
                # 将结果放入队列
                data_queue.put((name, data_item, len(motion)))
                
                with lock:
                    processed_count += 1
                    
            except Exception as e:
                if SHOW_INFO:
                    print(f"Loading data error: {name}. Skipped. Error: {e}")
        
        # 准备要处理的数据项
        items_to_process = list(enumerate(self.id_list))
        if debug:
            items_to_process = items_to_process[:maxdata]
        
        # 创建进度条（在主线程中显示）
        if not debug:
            pbar = tqdm(total=len(items_to_process), desc=f"Loading Dataset {split_file}")
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(process_single_item, item) for item in items_to_process]
            
            # 等待所有任务完成并更新进度
            for future in tqdm(futures, desc="Processing items") if debug else futures:
                future.result()
                if not debug:
                    pbar.update(1)
        
        if not debug:
            pbar.close()
        
        # 从队列中收集所有结果
        while not data_queue.empty():
            try:
                name, data_item, length = data_queue.get_nowait()
                data_dict[name] = data_item
                new_name_list.append(name)
                length_list.append(length)
            except queue.Empty:
                break
        
        # 按长度排序（可选，根据原代码逻辑）
        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        # 或者保持原顺序
        name_list = new_name_list
            
        self.data_dict = data_dict
        self.name_list = name_list
        
        print(f"Successfully loaded {len(self.name_list)} items out of {len(self.id_list)}")
    
    def __len__(self):
        return len(self.name_list)  
        
    def __getitem__(self, idx):
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # name, motion, m_length, m_tokens, m_tokens_len, caption, sent_len, "_".join(tokens), word_embeddings, pos_one_hots, all_captions, cot, task
        return name, motion, m_length, None, None, None, None, None, None, None, None, None, None

def unit_test():
    test_dataset = Text2MotionDatasetToken(
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
        debug=True,
    )
    print (len(test_dataset))
    print (test_dataset[10])
    
if __name__ == "__main__":
    unit_test()