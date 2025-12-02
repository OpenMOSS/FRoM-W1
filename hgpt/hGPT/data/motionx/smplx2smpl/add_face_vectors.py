# coding=utf-8
# Copyright 2022 The IDEA Authors (Shunlin Lu and Ling-Hao Chen). All rights reserved.
#
# For all the datasets, be sure to read and follow their license agreements,
# and cite them accordingly.
# If the unifier is used in your research, please consider to cite as:
#
# @article{humantomato,
#   title={HumanTOMATO: Text-aligned Whole-body Motion Generation},
#   author={Lu, Shunlin and Chen, Ling-Hao and Zeng, Ailing and Lin, Jing and Zhang, Ruimao and Zhang, Lei and Shum, Heung-Yeung},
#   journal={arxiv:2310.12978},
#   year={2023}
# }
#
#
# Licensed under the IDEA License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/IDEA-Research/HumanTOMATO/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. We provide a license to use the code, 
# please read the specific details carefully.
#

"""
Fix https://github.com/IDEA-Research/Motion-X/issues/55 first!!
"""

import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import numpy as np

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path
    
def process_file_parallel(idx, file_path):
    if idx % 1000 == 0:
        print (f"processing file {idx}: {file_path} ...")
    
    data_623 = np.load(file_path)
    face_path = file_path.replace('vectors_623', 'smplx_322')
    if os.path.exists(face_path):
        smplx_322 = np.load(face_path)
        face_expr = smplx_322[:, 159:159+50]
        if len(face_expr) > len(data_623):
            face_expr = face_expr[:len(data_623),:]
            
        motion_w_face = np.concatenate((data_623, face_expr), axis=1)
        output_path = file_path.replace("vectors_623", "vectors_673")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, motion_w_face)
    else:
        print ("no face path: ", file_name)
    
if __name__ == "__main__":
    # transfer_to_body_only_humanml
    # change your folder path here
    base_path = ''
    vectors_623_path = base_path + 'vectors_623'
    
    file_names = findAllFile(vectors_623_path)
    print ("file_names: ", len(file_names))
    inputs = [(idx, file_path) for idx, file_path in enumerate(file_names) if not os.path.exists(file_path.replace("vectors_623", "vectors_673"))]
    print ("inputs: ", len(inputs))

    # debug
    for item in inputs:
        process_file_parallel(*item)
        
    # max_workers = 500
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     frame_num = executor.map(lambda p: process_file_parallel(*p), inputs)
        
    print ("Good job!")