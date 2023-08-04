import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, AutoProcessor, AutoModelForAudioClassification, Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import IPython.display as ipd
import numpy as np
import pandas as pd

def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    
    if speech.shape[0]>1 and speech.ndim > 1: # torchload unknown error
            speech = speech[0, :]
    
    # inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding="max_length", max_length=84800)
    print(inputs.input_values)
    inputs = {key: inputs[key].to("cuda") for key in inputs}
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True)
        logits = model(**inputs).logits
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    s = F.softmax(logits, dim=1).cpu()
    return output, s

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

model_name_or_path = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
sampling_rate = feature_extractor.sampling_rate

model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition") 
model.projector = nn.Linear(1024, 1024, bias=True)
model.classifier = nn.Linear(1024, 8, bias=True)
torch_state_dict = torch.load('/home/yuxinguo/EmoFormer/SER/pytorch_model.bin', map_location=torch.device('cpu'))
model.projector.weight.data = torch_state_dict['classifier.dense.weight']
model.projector.bias.data = torch_state_dict['classifier.dense.bias']
model.classifier.weight.data = torch_state_dict['classifier.output.weight']
model.classifier.bias.data = torch_state_dict['classifier.output.bias']

model.cuda()

path = "/home/yuxinguo/data/RAVDESS/Audio_Speech_Actors_01-24/Actor_02/03-01-01-01-01-02-02.wav"
outputs, logits = predict(path, sampling_rate)

print(logits)