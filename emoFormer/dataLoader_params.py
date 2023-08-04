import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import librosa    
import argparse
import time

import os
import random
import numpy as np
from tqdm import tqdm
import torch

    
def get_data(file_name):
    # 03-01-03-01-01-01-01
    emo_list = ["01", "02", "03", "04", "05", "06", "07", "08"]
    con_list = ["01", "02"]
    
    file_name_split = file_name.split("-")
    
    c1 = file_name_split[4]
    e1 = file_name_split[2]
    itensity = file_name_split[3]
    
    c2 = " "
    e2 = " "
    
    if c1 == "01":
        c2 = "02"
    else:
        c2 = "01"

    e2 = random.choice(emo_list)
    if itensity == "02":
        while (e2 == e1 or e2 == "01"): 
            e2 = random.choice(emo_list)
    else:
        while e2 == e1:
            e2 = random.choice(emo_list)
    
    con_file_name = new_name(file_name_split, c2, e1)
    emo_file_name = new_name(file_name_split, c1, e2)

    
    return con_file_name, emo_file_name

def new_name(split, con, emo):
    
    split[4] = con
    split[2] = emo
    
    return "-".join(split)

def find_idx_by_name(file_name, my_list):
    
    for i in range(len(my_list)):
        if my_list[i]["name"] == file_name:
            return i
        
    return None

def unpack_npz(npz_file):
    flame_params = np.load(npz_file)
    shape_params = torch.from_numpy(flame_params["shape"])
    expression_params = torch.from_numpy(flame_params["expr"])
    # pose_params = torch.from_numpy(flame_params["rotation"])
    # neck_pose = torch.from_numpy(flame_params["neck_pose"])
    # eye_pose = torch.from_numpy(flame_params["eyes_pose"])
    # transl = torch.from_numpy(flame_params["translation"])
    # jaw_pose = torch.from_numpy(flame_params["jaw_pose"])
    # pose_params_1 = torch.cat((pose_params, jaw_pose),dim=1)
    pose_params = torch.from_numpy(flame_params["pose"])
    pose_params[:,:3] = 0
    shape_params = torch.from_numpy(np.zeros_like(shape_params))
    return shape_params, expression_params, pose_params

def process_audio(wav_file, processor):
    # speech_array, sampling_rate = librosa.load(wav_file, sr=16000)
    # wav = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
    # return wav
    return wav_file

def load_data(npz_dir, wav_dir, n):

    # Get a list of all NPZ files
    npz_files = [file for file in os.listdir(npz_dir) if file.endswith(".npz")]
    npz_files = npz_files[:n]

    # Shuffle the list of NPZ files
    random.seed(42)
    random.shuffle(npz_files)

    # Calculate the split sizes based on the ratios
    total_files = len(npz_files)
    train_size = int(total_files * 0.8)
    val_size = int(total_files * 0.1)
    if total_files == 1:
        train_size = 1
        val_size = 0


    print("Loading data:")

    all_data = []
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    for npz_file in tqdm(npz_files):
        npz_path = os.path.join(npz_dir, npz_file)
        npz_name = npz_file.replace(".npz", "")
        npz_parts = npz_name.split("-")[1:]  # Remove the first two parts ("01")
        wav_file = "03-" + "-".join(npz_parts) + ".wav"
        wav_path = os.path.join(wav_dir, wav_file)
        # print(npz_path, wav_path)
        shape, expr, pose = unpack_npz(npz_path)
        audio = process_audio(wav_path, processor)
        
        all_data.append({
            "name": npz_name,
            "audio": audio,
            "shape": shape,
            "expr": expr,
            "pose": pose,
        })
        
    
    print(f"Loaded {total_files} data, {train_size} for training, {val_size} for validation and {total_files - train_size - val_size} for testing")
    train_set = Dataset(all_data[:train_size], data_type="train")
    val_set = Dataset(all_data[train_size:train_size + val_size], data_type="val")
    test_set = Dataset(all_data[train_size + val_size:], data_type="test")

    data_loader = {}

    data_loader["train"] = data.DataLoader(dataset = train_set, batch_size = 1, shuffle=True)
    data_loader["val"] = data.DataLoader(dataset = val_set, batch_size = 1, shuffle=False)
    data_loader["test"] = data.DataLoader(dataset = test_set, batch_size = 1, shuffle=False)
    
    return data_loader

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, data_type="train", cross=True):
        self.data = data
        self.len = len(self.data)
        self.data_type = data_type
        self.cross = cross
        
    def check_file_pair(self, index):
        
        file_name = self.data[index]["name"]
        
        con_file_name, emo_file_name = get_data(file_name)
        con_idx = find_idx_by_name(con_file_name, self.data)
        emo_idx = find_idx_by_name(emo_file_name, self.data)
        
        if con_idx == None:
            return "other index", None
        
        if emo_idx == None:
            return "other emo", None
        
        else:
            return "ok", index
        
        

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        
        file_name = self.data[index]["name"]
            
        audio = self.data[index]["audio"]
        shape = self.data[index]["shape"]
        expr = self.data[index]["expr"]
        pose = self.data[index]["pose"]
        params = np.concatenate((shape, expr, pose), axis = 1)
            
        return file_name, audio, torch.FloatTensor(params)
        
    def __len__(self):
        return self.len
    