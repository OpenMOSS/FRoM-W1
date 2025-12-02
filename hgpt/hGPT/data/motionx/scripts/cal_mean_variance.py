import numpy as np
import sys
import os
from os.path import join as pjoin
from tqdm import tqdm

def findAllFile(base):
    """
    Recursively find all files in the specified directory.

    Args:
        base (str): The base directory to start the search.

    Returns:
        list: A list of file paths found in the directory and its subdirectories.
    """
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path

# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)

def mean_variance(data_dir, save_dir, joints_num, add_face=False):
    # TODO: add face
    error_list = []
    file_list = findAllFile(data_dir)
    data_list = []

    for file in tqdm(file_list):
        try:
            data = np.load(pjoin(data_dir, file))
            if np.isnan(data).any():
                print("isnan: ", file)
                error_list.append(pjoin(data_dir, file))
                continue
            data_list.append(data)
        except:
            # �NUMPYv{'descr': '<f4', 'fortran_order': False, 'shape': (299, 623), }
            print ("error: ", pjoin(data_dir, file))
            error_list.append(pjoin(data_dir, file))
            continue

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
    Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0
    Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
    Std[4 + (joints_num - 1) * 9 + joints_num * 3: ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: ].mean() / 1.0

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]

    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    print ("error list: ")
    print (error_list)
    return Mean, Std

if __name__ == '__main__':
    base_path = ''
    data_dir = f'{base_path}/motion_data/vectors_263'
    save_dir = f'{base_path}/mean_std/vectors_263'
    mean, std = mean_variance(data_dir, save_dir, 22, add_face=False)