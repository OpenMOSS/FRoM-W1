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

import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path

def process_file_parallel(idx, file_path, body_joints, joints):
    print (f"processing file {idx}: {file_path} ...")
    data = np.load(file_path)
    data_263 = np.concatenate((data[:, :4+(body_joints - 1)*3], data[:, 4+(joints - 1)*3:4+(joints - 1)*3+(body_joints - 1)*6], data[:, 4 + (joints - 1)*9: 4 + (joints - 1) *9 + body_joints *3], data[:, -4:]), axis=1)
    assert data_263.shape[1] == 263
    output_path = file_path.replace("vectors_623", "vectors_263")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, data_263)
    
if __name__ == "__main__":
    # transfer_to_body_only_humanml
    
    joints = 52
    body_joints = 22

    # change your folder path here
    base_path = ''
    folder_path = base_path + 'motion_data/vectors_623'
    file_names = findAllFile(folder_path)
    inputs = [(idx, file_path, body_joints, joints) for idx, file_path in enumerate(file_names) if not os.path.exists(file_path.replace("vectors_623", "vectors_263"))]
    print ("input num: ", len(inputs))
    
    max_workers = 300
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        frame_num = executor.map(lambda p: process_file_parallel(*p), inputs)

    print ("Good job!")