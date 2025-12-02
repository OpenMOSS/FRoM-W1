import codecs as cs
import json
import os
import pickle
import queue
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from os.path import join as pjoin

import numpy as np
import rich
import spacy
from rich.progress import track
from torch.utils.data import Dataset

SHOW_INFO=True

class Text2MotionDatasetTrain(Dataset):
    def __init__(
        self,
        motion_token_path,
        text_path,
        cot_path,
        split_path,
        split,
        mean,
        std,
        unit_length,
        fps,
        task_path,
        std_text,
        debug=False,
        **kwargs,
    ):
        # init
        cot_path = cot_path if cot_path != '' else None
        split_file = pjoin(split_path, f'{split}.txt')
        
        self.mean = mean
        self.std = std        
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

        new_name_list = []
        data_dict = {}

        for i, name in enumerator:
            if len(new_name_list) > maxdata:
                break
            
            # load motion token
            try:
                m_token_list = np.load(pjoin(motion_token_path, f'{name}.npy'), allow_pickle=True)
                
            except Exception as e:
                if SHOW_INFO:
                    print (f"Loading data error: {name}. Skipped.")
                continue

            if np.isnan(m_token_list).any() or m_token_list.shape[0] == 0:
                if SHOW_INFO:
                    print (f"found nan or none: {name}. skiped.")
                continue

            # load cot
            cot_list = []
            if cot_path != None:
                cot_file = pjoin(cot_path, name + '.txt')
                if not os.path.exists(cot_file):
                    if SHOW_INFO:
                        print (f"No such a cot file: {name}. skipped.")
                    continue
                else:
                    with open(cot_file, 'r') as f:
                        cot_list = f.readlines()

            # load text
            text_data = []
            flag = False
            with cs.open(pjoin(text_path, name + '.txt')) as f:
                lines = f.readlines()
                for line_idx, line in enumerate(lines):
                    text_dict = {}
                    line_split = line.split('#')
                    try:
                        caption = line_split[0]
                        t_tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        assert f_tag <= to_tag, "f_tag > t_tag"
                    except Exception as e:
                        if SHOW_INFO:
                            print(f"Error load text: {name}. Error: {str(e)}. Line data: {line_split}. skipped.")
                        continue

                    text_dict['caption'] = caption
                    text_dict['tokens'] = t_tokens
                    
                    # TODO: fix len(cot_list) != len(lines)
                    if cot_path == None:
                        text_dict['cot'] = ''
                    elif len(cot_list) != len(lines):
                        if SHOW_INFO:
                            print (f"cot != text lines: {name}. {len(cot_list)} vs {len(lines)}")
                        continue
                    else:
                        text_dict['cot'] = cot_list[line_idx]
                    
                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        m_token_list_new = [
                            tokens[int(f_tag * fps / unit_length
                                        ):int(to_tag * fps /
                                                unit_length)]
                            for tokens in m_token_list
                            if int(f_tag * fps / unit_length) <
                            int(to_tag * fps / unit_length)
                        ]
                        if len(m_token_list_new) == 0:
                            if SHOW_INFO:
                                print (f"new motion token list is []: {name}. skipped.")
                            continue

                        new_name = '%s_%f_%f' % (name, f_tag,
                                                    to_tag)
                        data_dict[new_name] = {
                            'm_token_list': m_token_list_new,
                            'text': [text_dict]
                        }
                        new_name_list.append(new_name)

            if flag:
                data_dict[name] = {
                    'm_token_list': m_token_list,
                    'text': text_data
                }
                new_name_list.append(name)
        
        self.data_dict = data_dict
        self.name_list = new_name_list
        print(f"Successfully loaded {len(self.name_list)} samples")
        # exit(0)
        
        # texts
        self.std_text = std_text
        self.nlp = spacy.load('en_core_web_sm')

        # tasks
        self.instructions = json.load(open(task_path, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])

        
    def __len__(self):
        return len(self.name_list) * len(self.tasks)

    def __getitem__(self, idx):
        data_idx = idx % len(self.name_list)
        task_idx = random.randint(0, len(self.tasks) - 1)
        
        name = self.name_list[data_idx]
        data = self.data_dict[name]
        m_token_list, text_list = data['m_token_list'], data['text']
        
        m_tokens = random.choice(m_token_list)
        text_data = random.choice(text_list)
        caption = text_data['caption']
        cot = text_data["cot"]
        
        if self.std_text:
            doc = self.nlp(caption)
            word_list = []
            pos_list = []
            for token in doc:
                word = token.text
                if not word.isalpha():
                    continue
                if (token.pos_ == 'NOUN'
                        or token.pos_ == 'VERB') and (word != 'left'):
                    word_list.append(token.lemma_)
                else:
                    word_list.append(word)
                pos_list.append(token.pos_)
            caption = ' '.join(word_list)
        
        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

        coin = np.random.choice([False, False, True])
        if coin:
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]

        m_tokens_len = m_tokens.shape[0]
        task = self.tasks[task_idx]
        
        # name, motion, m_length, m_tokens, m_tokens_len, caption, sent_len, "_".join(tokens), word_embeddings, pos_one_hots, all_captions, cot, task
        return name, None, None, m_tokens, m_tokens_len, caption, None, None, None, None, all_captions, cot, task


class Text2MotionDatasetTrainDist(Dataset):
    # def __init__(
    #     self,
    #     motion_token_path,
    #     text_path,
    #     cot_path,
    #     split_path,
    #     split,
    #     mean,
    #     std,
    #     unit_length,
    #     fps,
    #     task_path,
    #     std_text,
    #     debug=False,
    #     num_workers=16,
    #     **kwargs,
    # ):
    #     # init
    #     cot_path = cot_path if cot_path != '' else None
    #     split_file = pjoin(split_path, f'{split}.txt')
        
    #     self.mean = mean
    #     self.std = std        
    #     self.unit_length = unit_length
    #     self.fps = fps
    #     self.num_workers = num_workers
            
    #     # load id list
    #     self.id_list = []
    #     with cs.open(split_file, "r") as f:
    #         for line in f.readlines():
    #             self.id_list.append(line.strip())
    #     print(f"Id list num: {len(self.id_list)}")
                
    #     if debug:
    #         maxdata = 100
    #         subset = '_tiny'
    #         id_sublist = self.id_list[:maxdata]
    #     else:
    #         maxdata = 1e10
    #         subset = ''
    #         id_sublist = self.id_list

    #     # 使用多线程加载数据
    #     self.data_dict = {}
    #     self.name_list = []
        
    #     print(f"Loading dataset with {self.num_workers} workers...")
    #     self._load_data_parallel(id_sublist, motion_token_path, text_path, cot_path, maxdata)

    #     # texts
    #     self.std_text = std_text
    #     self.nlp = spacy.load('en_core_web_sm')

    #     # tasks
    #     self.instructions = json.load(open(task_path, 'r'))
    #     self.tasks = []
    #     for task in self.instructions.keys():
    #         for subtask in self.instructions[task].keys():
    #             self.tasks.append(self.instructions[task][subtask])

    def _load_single_data(self, name, motion_token_path, text_path, cot_path):
        """加载单个数据样本"""
        try:
            # load motion token
            m_token_list = np.load(pjoin(motion_token_path, f'{name}.npy'), allow_pickle=True)
        except Exception as e:
            if SHOW_INFO:
                print(f"Loading data error: {name}. Skipped. Error: {e}")
            return None, None

        if np.isnan(m_token_list).any() or m_token_list.shape[0] == 0:
            if SHOW_INFO:
                print(f"found nan or none: {name}. skiped.")
            return None, None

        # load cot
        cot_list = []
        if cot_path is not None:
            cot_file = pjoin(cot_path, name + '.txt')
            if not os.path.exists(cot_file):
                if SHOW_INFO:
                    print(f"No such a cot file: {name}. skipped.")
                return None, None
            else:
                with open(cot_file, 'r') as f:
                    cot_list = f.readlines()

        # load text
        text_data = []
        flag = False
        text_file = pjoin(text_path, name + '.txt')
        if not os.path.exists(text_file):
            if SHOW_INFO:
                print(f"No such a text file: {name}. skipped.")
            return None, None
            
        try:
            with cs.open(text_file) as f:
                lines = f.readlines()
                for line_idx, line in enumerate(lines):
                    text_dict = {}
                    line_split = line.split('#')
                    try:
                        caption = line_split[0]
                        t_tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        assert f_tag <= to_tag, "f_tag > t_tag"
                    except:
                        if SHOW_INFO:
                            print(f"Error load text: {name}. skipped.")
                        continue

                    text_dict['caption'] = caption
                    text_dict['tokens'] = t_tokens
                    
                    # TODO: fix len(cot_list) != len(lines)
                    if cot_path is None:
                        text_dict['cot'] = ''
                    elif len(cot_list) != len(lines):
                        if SHOW_INFO:
                            print(f"cot != text lines: {name}. {len(cot_list)} vs {len(lines)}")
                        continue
                    else:
                        text_dict['cot'] = cot_list[line_idx]
                    
                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        m_token_list_new = [
                            tokens[int(f_tag * self.fps / self.unit_length
                                        ):int(to_tag * self.fps /
                                                self.unit_length)]
                            for tokens in m_token_list
                            if int(f_tag * self.fps / self.unit_length) <
                            int(to_tag * self.fps / self.unit_length)
                        ]
                        if len(m_token_list_new) == 0:
                            if SHOW_INFO:
                                print(f"new motion token list is []: {name}. skipped.")
                            continue

                        new_name = '%s_%f_%f' % (name, f_tag, to_tag)
                        data_item = {
                            'm_token_list': m_token_list_new,
                            'text': [text_dict]
                        }
                        return new_name, data_item

            if flag:
                data_item = {
                    'm_token_list': m_token_list,
                    'text': text_data
                }
                return name, data_item
                
        except Exception as e:
            if SHOW_INFO:
                print(f"Error processing text for {name}: {e}")
            return None, None
            
        return None, None

    def _load_data_parallel(self, id_list, motion_token_path, text_path, cot_path, maxdata):
        """使用多线程并行加载数据"""
        data_dict = {}
        name_list = []
        lock = threading.Lock()
        
        def process_success(result):
            name, data_item = result
            if name is not None and data_item is not None:
                with lock:
                    if len(name_list) < maxdata:
                        data_dict[name] = data_item
                        name_list.append(name)
                    else:
                        return False  # 停止处理
            return True

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            future_to_name = {
                executor.submit(self._load_single_data, name, motion_token_path, text_path, cot_path): name 
                for name in id_list
            }
            
            # 处理完成的任务
            completed = 0
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                    if not process_success(result):
                        break  # 达到最大数据量，停止处理
                    completed += 1
                    if completed % 100 == 0:
                        print(f"Processed {completed}/{len(id_list)} samples, loaded {len(name_list)} valid samples")
                except Exception as e:
                    if SHOW_INFO:
                        print(f"Error processing {name}: {e}")
        
        self.data_dict = data_dict
        self.name_list = name_list
        print(f"Successfully loaded {len(self.name_list)} samples")

    def __len__(self):
        return len(self.name_list) * len(self.tasks)

    def __getitem__(self, idx):
        data_idx = idx % len(self.name_list)
        task_idx = random.randint(0, len(self.tasks) - 1)
        
        name = self.name_list[data_idx]
        data = self.data_dict[name]
        m_token_list, text_list = data['m_token_list'], data['text']
        
        m_tokens = random.choice(m_token_list)
        text_data = random.choice(text_list)
        caption = text_data['caption']
        cot = text_data["cot"]
        
        if self.std_text:
            doc = self.nlp(caption)
            word_list = []
            pos_list = []
            for token in doc:
                word = token.text
                if not word.isalpha():
                    continue
                if (token.pos_ == 'NOUN'
                        or token.pos_ == 'VERB') and (word != 'left'):
                    word_list.append(token.lemma_)
                else:
                    word_list.append(word)
                pos_list.append(token.pos_)
            caption = ' '.join(word_list)
        
        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

        coin = np.random.choice([False, False, True])
        if coin:
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]

        m_tokens_len = m_tokens.shape[0]
        task = self.tasks[task_idx]
        
        # name, motion, m_length, m_tokens, m_tokens_len, caption, sent_len, "_".join(tokens), word_embeddings, pos_one_hots, all_captions, cot, task
        return name, None, None, m_tokens, m_tokens_len, caption, None, None, None, None, all_captions, cot, task

    def __init__(
        self,
        motion_token_path,
        text_path,
        cot_path,
        split_path,
        split,
        mean,
        std,
        unit_length,
        fps,
        task_path,
        std_text,
        debug=False,
        num_workers=4,  # 添加多线程参数
        **kwargs,
    ):
        # init
        cot_path = cot_path if cot_path != '' else None
        split_file = pjoin(split_path, f'{split}.txt')
        
        self.mean = mean
        self.std = std        
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
            subset = '_tiny'
        else:
            maxdata = 1e10
            subset = ''

        # 使用多线程加载数据
        data_dict = {}
        new_name_list = []
        
        # 创建线程安全的数据结构
        data_queue = queue.Queue()
        processed_count = 0
        lock = threading.Lock()
        
        def process_single_item(item):
            nonlocal processed_count
            i, name = item
            
            with lock:
                if len(new_name_list) >= maxdata:
                    return
            
            try:
                # load motion token
                try:
                    m_token_list = np.load(pjoin(motion_token_path, f'{name}.npy'), allow_pickle=True)
                except Exception as e:
                    if SHOW_INFO:
                        print(f"Loading data error: {name}. Skipped.")
                    return

                if np.isnan(m_token_list).any() or m_token_list.shape[0] == 0:
                    if SHOW_INFO:
                        print(f"found nan or none: {name}. skiped.")
                    return

                # load cot
                cot_list = []
                if cot_path is not None:
                    cot_file = pjoin(cot_path, name + '.txt')
                    if not os.path.exists(cot_file):
                        if SHOW_INFO:
                            print(f"No such a cot file: {name}. skipped.")
                        return
                    else:
                        with open(cot_file, 'r') as f:
                            cot_list = f.readlines()

                # load text
                text_data = []
                flag = False
                try:
                    with cs.open(pjoin(text_path, name + '.txt')) as f:
                        lines = f.readlines()
                        for line_idx, line in enumerate(lines):
                            text_dict = {}
                            line_split = line.split('#')
                            try:
                                caption = line_split[0]
                                t_tokens = line_split[1].split(' ')
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag
                                assert f_tag <= to_tag, "f_tag > t_tag"
                            except:
                                if SHOW_INFO:
                                    print(f"Error load text: {name}. skipped.")
                                continue

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            
                            # TODO: fix len(cot_list) != len(lines)
                            if cot_path is None:
                                text_dict['cot'] = ''
                            elif len(cot_list) != len(lines):
                                if SHOW_INFO:
                                    print(f"cot != text lines: {name}. {len(cot_list)} vs {len(lines)}")
                                continue
                            else:
                                text_dict['cot'] = cot_list[line_idx]
                            
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                m_token_list_new = [
                                    tokens[int(f_tag * fps / unit_length
                                                ):int(to_tag * fps /
                                                        unit_length)]
                                    for tokens in m_token_list
                                    if int(f_tag * fps / unit_length) <
                                    int(to_tag * fps / unit_length)
                                ]
                                if len(m_token_list_new) == 0:
                                    if SHOW_INFO:
                                        print(f"new motion token list is []: {name}. skipped.")
                                    continue

                                new_name = '%s_%f_%f' % (name, f_tag, to_tag)
                                
                                # 将分段数据放入队列
                                segmented_data = {
                                    'm_token_list': m_token_list_new,
                                    'text': [text_dict]
                                }
                                data_queue.put(('segmented', new_name, segmented_data))
                
                except Exception as e:
                    print(f"Error processing text for {name}: {e}")
                    return

                # 处理完整数据
                if flag and text_data:
                    full_data = {
                        'm_token_list': m_token_list,
                        'text': text_data
                    }
                    data_queue.put(('full', name, full_data))
                    
                with lock:
                    processed_count += 1
                    
            except Exception as e:
                print(f"Unexpected error processing {name}: {e}")
        
        # 准备要处理的数据项
        items_to_process = list(enumerate(self.id_list))
        if debug:
            items_to_process = items_to_process[:maxdata]
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            if debug:
                # 在debug模式下使用tqdm显示进度
                from tqdm import tqdm
                futures = [executor.submit(process_single_item, item) for item in items_to_process]
                for future in tqdm(futures, desc="Loading Dataset"):
                    future.result()
            else:
                # 在非debug模式下使用rich.progress
                futures = []
                with rich.progress.Progress(
                    rich.progress.SpinnerColumn(),
                    rich.progress.TextColumn("[progress.description]{task.description}"),
                    rich.progress.BarColumn(),
                    rich.progress.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    rich.progress.TimeRemainingColumn(),
                ) as progress:
                    task = progress.add_task(f"Loading Dataset {split_file}", total=len(items_to_process))
                    
                    # 提交所有任务
                    for item in items_to_process:
                        future = executor.submit(process_single_item, item)
                        futures.append(future)
                    
                    # 等待所有任务完成并更新进度
                    for future in futures:
                        future.result()
                        progress.update(task, advance=1)
        
        # 从队列中收集所有结果
        while not data_queue.empty():
            try:
                data_type, name, data_item = data_queue.get_nowait()
                if data_type == 'full':
                    data_dict[name] = data_item
                    new_name_list.append(name)
                else:  # segmented
                    data_dict[name] = data_item
                    new_name_list.append(name)
            except queue.Empty:
                break
        
        self.data_dict = data_dict
        self.name_list = new_name_list

        # texts
        self.std_text = std_text
        self.nlp = spacy.load('en_core_web_sm')

        # tasks
        self.instructions = json.load(open(task_path, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])
        
        print(f"Successfully loaded {len(self.name_list)} items")

    def __len__(self):
        return len(self.name_list) * len(self.tasks)

    def __getitem__(self, idx):
        data_idx = idx % len(self.name_list)
        task_idx = random.randint(0, len(self.tasks) - 1)
        
        name = self.name_list[data_idx]
        data = self.data_dict[name]
        m_token_list, text_list = data['m_token_list'], data['text']
        
        m_tokens = random.choice(m_token_list)
        text_data = random.choice(text_list)
        caption = text_data['caption']
        cot = text_data["cot"]
        
        if self.std_text:
            doc = self.nlp(caption)
            word_list = []
            pos_list = []
            for token in doc:
                word = token.text
                if not word.isalpha():
                    continue
                if (token.pos_ == 'NOUN'
                        or token.pos_ == 'VERB') and (word != 'left'):
                    word_list.append(token.lemma_)
                else:
                    word_list.append(word)
                pos_list.append(token.pos_)
            caption = ' '.join(word_list)
        
        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

        coin = np.random.choice([False, False, True])
        if coin:
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]

        m_tokens_len = m_tokens.shape[0]
        task = self.tasks[task_idx]
        
        # name, motion, m_length, m_tokens, m_tokens_len, caption, sent_len, "_".join(tokens), word_embeddings, pos_one_hots, all_captions, cot, task
        return name, None, None, m_tokens, m_tokens_len, caption, None, None, None, None, all_captions, cot, task

def unit_test():
    test_dataset = Text2MotionDatasetTrain(
        motion_token_path = 'datasets/motionx/data/TOKENS',
        text_path = 'datasets/motionx/data/texts/semantic_labels',
        cot_path = 'datasets/motionx/data/texts/cot/v3',
        split_path = 'datasets/motionx/data/split',
        split = 'test',
        mean = 0.0,
        std = 1.0,
        unit_length = 4,
        fps = 30,
        task_path = './datasets/motionx/data/instructions/template_pretrain_orig.json',    
        std_text = True,
        debug=True,
    )
    print (len(test_dataset))
    print (test_dataset[10])
    
if __name__ == "__main__":
    unit_test()