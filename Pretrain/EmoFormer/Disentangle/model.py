from transformers import Wav2Vec2Model,Wav2Vec2Processor
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, AutoModelForAudioClassification

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def speech_file_to_array_fn(path, sampling_rate):
    
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    
    return speech

def compute_acc(y_logits, y_true):
    
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

class ContentEncoder(nn.Module):

    def __init__(self):
        
        super(ContentEncoder, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 6)
        self.fc = nn.Linear(1024, 512)

    def forward(self, x):

        x = self.transformer_encoder(x)
        # x = torch.mean(x, dim=1)
        x = self.fc(x)

        return x

class EmotionEncoder(nn.Module):

    def __init__(self, type='fc'):
        super(EmotionEncoder, self).__init__()
        self.type = type
        
        self.projector = nn.Linear(1024, 1024, bias=True)
        self.classifier = nn.Linear(1024, 8, bias=True)
        # self.relu = nn.ReLU()

    def forward(self, x):
        
        x = torch.mean(x, dim=1)
        x = self.projector(x)
        x = self.classifier(x)
        
        return x

class Decoder(nn.Module):
    pass
    #     def __init__(self):
    #         super(Decoder, self).__init__()
    #     def forward(self, con, emo): 
    #         pass

class Classify(nn.Module):
    pass

class DisentangleNet(nn.Module):

    def __init__(self, config, optim='Adam'):
        super(DisentangleNet, self).__init__()
        
        self.audio_encoder = Wav2Vec2Model.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        self.con_encoder = ContentEncoder()
        self.emo_encoder = EmotionEncoder()
        # self.decoder = Decoder()
        # self.classify = Classify()
        
        self.optim = optim
        
        self.l1loss = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss()

        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(list(self.con_encoder.parameters())+list(self.emo_encoder.parameters()),
                                           #+list(self.classify.parameters())
                                           config.lr, betas=(config.beta1, config.beta2))
            scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[15,30, 45, 60, 75, 90], gamma=0.5)
            
        
        else:
            self.optimizer = torch.optim.SGD(list(self.con_encoder.parameters())+list(self.emo_encoder.parameters()),
                                           #+list(self.classify.parameters())
                                           lr=0.005)

    def update_network(self, loss_dict):
        
        loss = loss_dict['con_loss']+ loss_dict['cla1_loss'] + loss_dict['cla2_loss']

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_func(self, data):
        
        self.con_encoder.train()
        self.emo_encoder.train()
        # self.decoder.train()
        # self.classify.train()

        loss, acc = self.process(data)

        self.update_network(loss)

        return loss, acc

    def val_func(self, data):

        self.con_encoder.eval()
        self.emo_encoder.eval()
        # self.decoder.eval()
        # self.classify.eval()

        with torch.no_grad():
            loss, acc = self.process(data)

        return loss, acc

    def process(self, data):

        loss = {}
        acc = {}
        
        x11 = data["sp_11"]
        x12 = data["sp_12"]
        x21 = data["sp_21"]
        x22 = data["sp_22"]

        x11 = x11["input_values"].squeeze(dim=1)
        x12 = x12["input_values"].squeeze(dim=1)
        x21 = x21["input_values"].squeeze(dim=1)
        x22 = x22["input_values"].squeeze(dim=1)
        
        x11 = x11.cuda()
        x12 = x12.cuda()
        x21 = x21.cuda()
        x22 = x22.cuda()
        
        gt1 = data["emo_1"].cuda()
        gt2 = data["emo_2"].cuda()
        
        out_11 = self.audio_encoder(x11)
        out_12 = self.audio_encoder(x12)
        out_21 = self.audio_encoder(x21)
        out_22 = self.audio_encoder(x22)
        
        feat_11 = out_11.last_hidden_state
        feat_12 = out_12.last_hidden_state
        feat_21 = out_21.last_hidden_state
        feat_22 = out_22.last_hidden_state
        
        c1_11 = self.con_encoder(feat_11) 
        c1_12 = self.con_encoder(feat_12) 
        c2_21 = self.con_encoder(feat_21) 
        c2_22 = self.con_encoder(feat_22)
        
        e1_11 = self.emo_encoder(feat_11)
        e2_12 = self.emo_encoder(feat_12)
        
        loss["con_loss"] = self.l1loss(c1_11, c1_12) + self.l1loss(c2_21, c2_22)
        loss["cla1_loss"] = self.cross_entropy(e1_11, gt1)
        loss["cla2_loss"] = self.cross_entropy(e2_12, gt2)

        acc["cla1_acc"] = compute_acc(e1_11 ,gt1)
        acc["cla2_acc"] = compute_acc(e2_12 ,gt2)

        return loss, acc


        
        
        