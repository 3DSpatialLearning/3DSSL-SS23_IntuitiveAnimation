import librosa
import numpy as np
import os
import random
from transformers import Wav2Vec2Model,Wav2Vec2Processor, AutoModelForAudioClassification
from transformers import Wav2Vec2FeatureExtractor

import tensorboardX
import torchaudio
import numpy as np
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def label_map(gt):
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

def pad_audio(audio, max_length):
    max = max_length
    padding = int(max - len(audio))
    audio = np.pad(audio, pad_width=(0, padding), mode='constant')
    
    return audio

class TestDataset(Dataset):

    def __init__(self, dataset_path, stage, max_length=5.3*16000):

        # root_dir = /home/yuxinguo/data/RAVDESS/Audio_Speech_Actors_01-24/
        # file_dir = /home/yuxinguo/data/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/.wav
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
            self.data_dir = [(os.path.join(self.dataset_dir, f'Actor_{i:02d}')) for i in range(1,22)]
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

        actor_dir = os.path.join(self.dataset_dir, f'Actor_{actor_idx:02d}')

        wav_11 = os.path.join(actor_dir, "-".join(file_split))
        
        speech = speech_file_to_array_fn(wav_11, self.sampling_rate)
        
        if speech.shape[0]>1 and speech.ndim > 1: # torchload unknown error
            speech = speech[0, :]
        # speech_1 = librosa.load(wav_11, sr = self.sampling_rate)
        inputs = self.feature_extractor(speech, sampling_rate=self.sampling_rate, return_tensors="pt", padding="max_length", max_length=84800)
        # audio = pad_audio(audio, self.max_length)
        
        # if inputs.input_values.shape[0] == 2: 
        #     print(inputs.input_values[:, 0] == inputs.input_values[:, 1])
        
        # load with librosa
        # audio_11, _ = librosa.load(wav_11, sr = 16000) # return np.ndarray
        # audio_11 = pad_audio(audio_11, self.max_length)
        # print(np.array_equal(audio, audio_11))
        # print(audio.shape, audio_11.shape)
        # print(type(audio), type(audio_11))
        
        e1 = label_map(e1)
        emo_1 = int(e1)-1
        
        return {"wav_11": wav_11,
                "sp_11": inputs, 
                "emo_1": emo_1
                }

model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
model.projector = nn.Linear(1024, 1024, bias=True)
model.classifier = nn.Linear(1024, 8, bias=True)

torch_state_dict = torch.load('/home/yuxinguo/EmoFormer/SER/pytorch_model.bin', map_location=torch.device('cpu'))
model.projector.weight.data = torch_state_dict['classifier.dense.weight']
model.projector.bias.data = torch_state_dict['classifier.dense.bias']
model.classifier.weight.data = torch_state_dict['classifier.output.weight']
model.classifier.bias.data = torch_state_dict['classifier.output.bias']
model = model.cuda()

print(model)

train_data = TestDataset("/home/yuxinguo/data/RAVDESS/Audio_Speech_Actors_01-24/", 'train')
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=True)

test_data = TestDataset("/home/yuxinguo/data/RAVDESS/Audio_Speech_Actors_01-24/", 'test')
test_loader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=True)

epoch_number = 0
EPOCHS = 20

# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9, 12, 15, 18], gamma=0.5)

writer = tensorboardX.SummaryWriter(comment='ser')

def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    losses = []
    pbar = tqdm(enumerate(train_loader),total=len(train_loader))
    for i, data in pbar:
        
        inputs = data["sp_11"] 
        inputs["input_values"] = inputs["input_values"].squeeze(dim=1)
        inputs = {key: inputs[key].to("cuda") for key in inputs} # input_values，attention_mask
        
        labels = data["emo_1"]
        labels = labels.to("cuda")
        
        optimizer.zero_grad()
        
        logits = model(**inputs).logits
        loss = F.cross_entropy(logits, labels)
        # pred = F.softmax(logits, dim=1)
        # loss = F.cross_entropy(pred*100, labels)
        
        loss.backward()

        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        losses.append(loss.item())
    
        pbar.set_description(f"(Epoch {epoch+1}, iteration {i}) loss:{np.mean(losses):.7f}")
    avg_tloss = running_loss / (i+1)
    
    return avg_tloss

for epoch in range(EPOCHS):
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print("now:", timestamp)
    
    avg_tloss = 0.
    avg_vloss = 0.
    print('EPOCH {}:'.format(epoch_number + 1))

    model.train(True)
    avg_tloss = train_one_epoch(epoch_number)
    
    # eval
    running_vloss = 0.
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            
            inputs = data["sp_11"] 
            inputs["input_values"] = inputs["input_values"].squeeze(dim=1)
            inputs = {key: inputs[key].to("cuda") for key in inputs} # input_values，attention_mask
        
            labels = data["emo_1"]
            labels = labels.to("cuda")

            logits = model(inputs["input_values"]).logits
            # pred = F.softmax(logits, dim=1)
            # Compute the loss and its gradients
            vloss = F.cross_entropy(logits, labels)
            
            running_vloss += vloss.item()
    
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_tloss, avg_vloss))
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_tloss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()
    
    model_path = 'model_{}_{}'.format(timestamp, epoch_number) 
    
    if epoch_number % 5 == 0:
        torch.save(model.state_dict(), model_path) 
        
    scheduler.step() 

    epoch_number += 1