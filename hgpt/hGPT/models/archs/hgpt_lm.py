import os
import time
import math
import random

from typing import List, Union, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

import heapq

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer
)

from .utils.token_emb import NewTokenEmb

class MLM(nn.Module):

    def __init__(
        self,
        model_path: str,
        model_type: str,
        stage: str,
        new_token_type: str,
        motion_codebook_size: int,
        framerate: float,
        down_t: int,
        predict_ratio: float,
        inbetween_ratio: float,
        max_length: int,
        lora: bool,
        quota_ratio: float,
        # noise_density: float,
        # mean_noise_span_length: int,
        **kwargs,
    ) -> None:

        super().__init__()

        # Parameters
        self.m_codebook_size = motion_codebook_size
        self.max_length = max_length
        self.framerate = framerate
        self.down_t = down_t
        self.predict_ratio = predict_ratio
        self.inbetween_ratio = inbetween_ratio
        # self.noise_density = noise_density
        # self.mean_noise_span_length = mean_noise_span_length
        self.quota_ratio = quota_ratio
        self.stage = stage

        # Instantiate language model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
        if model_type == "t5":
            self.language_model = T5ForConditionalGeneration.from_pretrained(
                model_path)
            self.lm_type = 'encdec'
        elif model_type == "gpt2":
            self.language_model = GPT2LMHeadModel.from_pretrained(model_path)
            self.lm_type = 'dec'
        elif model_type == "llama":
            self.language_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            self.lm_type = 'dec'
        else:
            raise ValueError("type must be either seq2seq or conditional")

        if self.lm_type == 'dec':
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add motion tokens
        self.tokenizer.add_tokens(
            [f'<motion_id_{i}>' for i in range(self.m_codebook_size + 3)])
        
        if new_token_type == "insert":
            self.language_model.resize_token_embeddings(len(self.tokenizer))
        elif new_token_type == "mlp":
            shared = NewTokenEmb(self.language_model.shared,
                                 self.m_codebook_size + 3)
            # lm_head = NewTokenEmb(self.language_model.lm_head,
            #   self.m_codebook_size + 3)
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            self.language_model.shared = shared
            # self.language_model.lm_head = lm_head
        
        # Lora
        # TODO: only tune motion tokens.
        # TODO: save only lora
        # Ref: https://github.com/huggingface/peft/blob/c1fe8105a5a4a612a6178699e1def5c66c2638d2/src/peft/utils/save_and_load.py
        if lora:
            from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
            # peft_config = LoraConfig(
            #     bias="none",
            #     task_type="CAUSAL_LM",
            #     #  inference_mode=False,
            #     r=8,
            #     lora_alpha=16,
            #     lora_dropout=0.05)
            peft_config = LoraConfig(
                bias="none",
                task_type="CAUSAL_LM",
                # modules_to_save: List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
                modules_to_save=["lm_head","embed_tokens"],
                # target_modules: The names of the modules to apply the adapter to
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "down_proj", "gate_proj", "up_proj"],
                # MotionGPT2
                r=32,
                lora_alpha=16,
                lora_dropout=0.05)
            self.language_model = get_peft_model(self.language_model,
                                                 peft_config)
    
    ########################################### Model Training ##########################################
    
    def forward(self, texts: List[str], motion_tokens: Tensor,
                motion_tokens_len: List[int], tasks: dict, cot: List[str]):
        if self.lm_type == 'encdec':
            return self.forward_encdec(texts, motion_tokens, motion_tokens_len, tasks, cot)
        elif self.lm_type == 'dec':
            return self.forward_dec(texts, motion_tokens, motion_tokens_len, tasks, cot)
        else:
            raise NotImplementedError("Only conditional_multitask supported")

    def forward_encdec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks: dict,
        cot: List[str]
    ):
        raise NotImplementedError
    
    def forward_dec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        motion_tokens_len: List[int],
        tasks: dict,
        cot: List[str]
    ):  
        
        # print ("* " * 20)
        # print (texts)
        # print (motion_tokens)  
        # print (motion_tokens_len)
        # print (tasks)                      
        # print (cot)
        # print ("* " * 20)
        
        self.tokenizer.padding_side = "right"

        motion_strings = self.motion_token_to_string(motion_tokens, motion_tokens_len)
        # print (motion_strings)
        
        # Supervised or unsupervised
        # condition = random.choice(
        #     ['text', 'motion', 'supervised', 'supervised', 'supervised'])
        condition = random.choice(
            ['supervised', 'supervised', 'supervised'])
        
        if condition == 'text':
            labels = texts
        elif condition == 'motion':
            labels = motion_strings
        else:
            inputs, outputs = self.template_fulfill(tasks, motion_tokens_len,
                                                    motion_strings, texts, cot)
            labels = []
            for i in range(len(inputs)):
                labels.append(inputs[i] + '\n' + outputs[i] +
                              self.tokenizer.eos_token)
        # print (labels)
        # print ("* " * 20)
        # # exit(0)
        inputs = self.tokenizer(labels,
                                padding=True,
                                # padding='max_length',
                                max_length=self.max_length,
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors="pt")
        # print (inputs)
        # print ("* " * 20)
        # input_ids = inputs['input_ids']  # 获取 input_ids
        # # 将每个样本的 token IDs 转换回字符串
        # decoded_strings = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        # # 打印结果
        # for i, text in enumerate(decoded_strings):
        #     print(f"Sample {i+1}: {text}")
        
        labels_input_ids = inputs.input_ids.to(motion_tokens.device)
        
        labels_attention_mask = inputs.attention_mask.to(motion_tokens.device)
        outputs = self.language_model(input_ids=labels_input_ids,
                                      attention_mask=labels_attention_mask,
                                      labels=inputs["input_ids"])
        
        # print (outputs.keys())
        # # print (outputs)
        # # print ("* " * 20)
        # exit(0)
        # print(f"{outputs.loss=}")
        # if torch.isnan(outputs.loss):
        #     print(f"{labels=}")
        #     print(f"{labels_input_ids=}")
        #     print(f"{labels_input_ids.shape=}")
        #     print(f"{torch.isnan(labels_input_ids).any()=}")
        #     print(f"{torch.isnan(outputs.logits).any()=}")
        #     exit(0)
        return outputs

    ########################################### Model Test ##########################################

    def generate_conditional(self,
                             texts: Optional[List[str]] = None,
                             motion_tokens: Optional[Tensor] = None,
                             lengths: Optional[List[int]] = None,
                             cot: Optional[List[str]] = None,
                             task: str = "t2m",
                             with_len: bool = False,
                             stage: str = 'train',
                             tasks: dict = None):

        self.device = self.language_model.device

        # for test
        # texts = ['a figure scrapes their shoes off on a mat.']
        # print (texts)
        
        if task in ["t2m", "m2m", "pred", "inbetween"]:

            if task == "t2m":
                assert texts is not None
                if with_len:
                    raise NotImplementedError
                cot = [''] * len(texts)
                motion_strings = [''] * len(texts)
                lengths = [0] * len(texts)
            
            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts, cot)
            # print ("* " * 10)
            # print ("inputs: ")
            # print (inputs)
            # print ("outputs: ")
            # print (outputs)
            # print ("* " * 10)
            
            outputs_tokens, cleaned_text = self.generate_direct(inputs,
                                                                max_length=self.max_length,
                                                                num_beams=1,
                                                                do_sample=True) # TODO: check here, do sample = False

            return outputs_tokens, cleaned_text
    
    def generate_direct(self,
                        texts: List[str],
                        max_length: int,
                        num_beams: int = 1,
                        do_sample: bool = True,
                        bad_words_ids: List[int] = None):

        # Device
        self.device = self.language_model.device

        # Tokenize
        if self.lm_type == 'dec':
            texts = [text + "\n" for text in texts] # "\n<soc>"
            self.tokenizer.padding_side = 'left'
        
        # print ("texts: ")
        # print (texts)
        
        source_encoding = self.tokenizer(texts,
                                         padding=True,
                                         # padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")
        
        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)

        if self.lm_type == 'encdec':
            outputs = self.language_model.generate(
                source_input_ids,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                bad_words_ids=bad_words_ids,
            )
        elif self.lm_type == 'dec':
            outputs = self.language_model.generate(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=do_sample,
                max_new_tokens=max_length)
            
        outputs_string = self.tokenizer.batch_decode(outputs,
                                                     skip_special_tokens=True)
        
        # print ("outputs_string: ")
        # print (outputs_string)
        # exit(0)
        
        outputs_tokens, cleaned_text = self.motion_string_to_token(
            outputs_string)
        return outputs_tokens, cleaned_text
    
    
    ########################################### Motion Token <==> String ###################################
    
    def motion_token_to_string(self, motion_token: Tensor, motion_tokens_len: List[int]):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu(
            ) if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()[:motion_tokens_len[i]]
            motion_string.append(
                (f'<motion_id_{self.m_codebook_size}>' +
                 ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
                 f'<motion_id_{self.m_codebook_size + 1}>'))
        return motion_string

    def motion_string_to_token(self, motion_string: List[str]):
        motion_tokens = []
        output_string = []
        for i in range(len(motion_string)):
            string = self.get_middle_str(
                motion_string[i], f'<motion_id_{self.m_codebook_size}>',
                f'<motion_id_{self.m_codebook_size + 1}>')
            string_list = string.split('><')
            
            token_list = []
            for token_i in string_list[1:-1]:
                tmp_i = token_i.split('_')[-1].replace('>', '')
                try:
                    tmp_i = int(tmp_i)
                    token_list.append(tmp_i)
                except:
                    print ("Decoder Error During Motion_string_to_token.")
                    continue

            if len(token_list) == 0:
                token_list = [0]
            token_list_padded = torch.tensor(token_list,
                                             dtype=int).to(self.device)
            motion_tokens.append(token_list_padded)
            output_string.append(motion_string[i].replace(
                string, '<Motion_Placeholder>'))

        return motion_tokens, output_string

    def get_middle_str(self, content, startStr, endStr):
        try:
            startIndex = content.index(startStr)
            if startIndex >= 0:
                startIndex += len(startStr)
            endIndex = content.index(endStr)
        except:
            return f'<motion_id_{self.m_codebook_size}><motion_id_0><motion_id_{self.m_codebook_size+1}>'

        return f'<motion_id_{self.m_codebook_size}>' + content[
            startIndex:endIndex] + f'<motion_id_{self.m_codebook_size+1}>'

    ########################################### Placeholder Fulfill ###################################
    
    def template_fulfill(self,
                         tasks,
                         motion_tokens_len,
                         motion_strings,
                         texts,
                         cot):
        inputs = []
        outputs = []
        for i in range(len(motion_tokens_len)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = motion_tokens_len[i]
            inputs.append(
                self.placeholder_fulfill(input_template, length,
                                         motion_strings[i], texts[i], cot[i]))
            outputs.append(
                self.placeholder_fulfill(output_template, length,
                                         motion_strings[i], texts[i], cot[i]))

        return inputs, outputs
    
    def placeholder_fulfill(self, prompt: str, length: int, motion_string: str,
                            text: str, cot: str):
        
        seconds = math.floor(length / self.framerate)
        motion_splited = motion_string.split('>')
        token_length = length / self.down_t
        predict_head = int(token_length * self.predict_ratio + 1)
        masked_head = int(token_length * self.inbetween_ratio + 1)
        masked_tail = int(token_length * (1 - self.inbetween_ratio) + 1)
        
        motion_predict_head = '>'.join(
            motion_splited[:predict_head]
        ) + f'><motion_id_{self.m_codebook_size+1}>'
        motion_predict_last = f'<motion_id_{self.m_codebook_size}>' + '>'.join(
            motion_splited[predict_head:])

        motion_masked = '>'.join(
            motion_splited[:masked_head]
        ) + '>' + f'<motion_id_{self.m_codebook_size+2}>' * (
            masked_tail - masked_head) + '>'.join(motion_splited[masked_tail:])

        if random.random() < self.quota_ratio:
            text = f'\"{text}\"'

        prompt = prompt.replace('<Caption_Placeholder>', text).replace(
            '<Motion_Placeholder>',
            motion_string).replace('<Frame_Placeholder>', f'{length}').replace(
                '<Second_Placeholder>', '%.1f' % seconds).replace(
                    '<Motion_Placeholder_s1>', motion_predict_head).replace(
                        '<Motion_Placeholder_s2>',
                        motion_predict_last).replace(
                            '<Motion_Placeholder_Masked>', motion_masked).replace(
                                '<CoT_Placeholder>', cot
                            )

        return prompt