import librosa
import numpy as np
import os
import random

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, AutoModelForAudioClassification

def pad_audio(audio, max_length):
    
    max = max_length
    padding = int(max - len(audio))
    audio = np.pad(audio, pad_width=(0, padding), mode='constant')
    
    return audio

def check_audio(speech):
    
    if speech.shape[0]>1 and speech.ndim > 1: # torchaudio unknown error, dual channel?
            speech = speech[0, :]

    else:
        pass
    
    return speech

def new_name(split, con, emo):
    
    split[4] = con
    split[2] = emo
    
    return "-".join(split)

def label_map(gt):
    
    # For some pre-trained models, the label are not consistent with RAVDESS
    # Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    if gt == "01":
        gt = "06"
    elif gt == "02":
        gt = "02"
    elif gt == "03":
        gt = "05"
    elif gt == "04":
        gt = "07"
    elif gt == "05":
        gt = "01"
    elif gt == "06":
        gt = "04"
    elif gt == "07":
        gt = "03"
    elif gt == "08":
        gt = "08"
    
    return gt

def one_hot(e):

    label = int(e)
    onehot = torch.zeros(8)
    onehot[label-1] = 1

    return onehot

def speech_file_to_array_fn(path, sampling_rate):
    
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    
    return speech

class DisentangleDataset(Dataset):

    def __init__(self, dataset_path, stage, max_length=5.3*16000):

        self.dataset_dir = dataset_path
        self.stage = stage
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        self.sampling_rate = self.feature_extractor.sampling_rate

        # Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
        # Emotional intensity (01 = normal, 02 = strong)
        self.emo = ["01", "02", "03", "04", "05", "06", "07", "08"]
        self.emo_intensity = ["01", "02"]
        self.con = ["01", "02"]
        
        self.max_length = max_length

        if self.stage == 'train':
            self.data_dir = [(os.path.join(self.dataset_dir, f'Actor_{i:02d}')) for i in range(1,21)]
            
        elif self.stage == 'test':
            self.data_dir = [(os.path.join(self.dataset_dir, f'Actor_{i:02d}')) for i in range(21,25)]
        
    def __len__(self):
        # directory * number of files
        return len(self.data_dir) * len(os.listdir(self.data_dir[0]))

    def __getitem__(self, idx):
        
        if self.stage == 'train':
            actor_idx = idx // 60 + 1
        elif self.stage == 'test':
            actor_idx = idx // 60 + 1 + 20
        
        wav_idx = idx % 60
        
        if self.stage == 'train':
            files = os.listdir(self.data_dir[actor_idx-1])
        elif self.stage == 'test':
            files = os.listdir(self.data_dir[actor_idx-21])
        files.sort()

        file = files[wav_idx] 
        # 03       -      01       -   02    -         01          -    01     -     01     -01    .wav
        # Modality - Vocal channel - Emotion - Emotional intensity - Statement - Repetition - Actor 
        
        file_split = file.split("-")

        c1 = file_split[4]
        e1 = file_split[2]
        itensity = file_split[3]

        # Choose a new content
        c2 = c1
        while c2 == c1:
            c2 = random.choice(self.con)

        # Choose a new emotion
        e2 = e1

        if itensity == "02":
            while (e2 == e1 or e2 == "01"): 
                e2 = random.choice(self.emo)
        
        else:
            while e2 == e1:
                e2 = random.choice(self.emo)

        actor_dir = os.path.join(self.dataset_dir, f'Actor_{actor_idx:02d}')

        wav_11 = os.path.join(actor_dir, "-".join(file_split))
        wav_12 = os.path.join(actor_dir, new_name(file_split, c1, e2))
        wav_21 = os.path.join(actor_dir, new_name(file_split, c2, e1))
        wav_22 = os.path.join(actor_dir, new_name(file_split, c2, e2))

        # audio_11, _ = librosa.load(wav_11, sr = 16000)
        # audio_12, _ = librosa.load(wav_12, sr = 16000)
        # audio_21, _ = librosa.load(wav_21, sr = 16000)
        # audio_22, _ = librosa.load(wav_22, sr = 16000)
        
        speech_11 = speech_file_to_array_fn(wav_11, self.sampling_rate)
        speech_12 = speech_file_to_array_fn(wav_12, self.sampling_rate)
        speech_21 = speech_file_to_array_fn(wav_21, self.sampling_rate)
        speech_22 = speech_file_to_array_fn(wav_22, self.sampling_rate)
        
        speech_11 = check_audio(speech_11)
        speech_12 = check_audio(speech_12)
        speech_21 = check_audio(speech_21)
        speech_22 = check_audio(speech_22)
    
        # audio_11 = pad_audio(audio_11, self.max_length)
        # audio_12 = pad_audio(audio_12, self.max_length)
        # audio_21 = pad_audio(audio_21, self.max_length)
        # audio_22 = pad_audio(audio_22, self.max_length)
        
        inputs_11 = self.feature_extractor(speech_11, sampling_rate=self.sampling_rate, return_tensors="pt", padding="max_length", max_length=84800)
        inputs_12 = self.feature_extractor(speech_12, sampling_rate=self.sampling_rate, return_tensors="pt", padding="max_length", max_length=84800)
        inputs_21 = self.feature_extractor(speech_21, sampling_rate=self.sampling_rate, return_tensors="pt", padding="max_length", max_length=84800)
        inputs_22 = self.feature_extractor(speech_22, sampling_rate=self.sampling_rate, return_tensors="pt", padding="max_length", max_length=84800)
        
        e1 = label_map(e1)
        e2 = label_map(e2)
        
        emo_1 = int(e1)-1
        emo_2 = int(e2)-1
        
        return {"sp_11": inputs_11, 
                "sp_12": inputs_12, 
                "sp_21": inputs_21, 
                "sp_22": inputs_22, 
                "emo_1": emo_1, "emo_2": emo_2
                }

