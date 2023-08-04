import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from wav2vec import Wav2Vec2Model, linear_interpolation
from encoder import DisentangleNet
from FLAME import FLAME

def normalization(input_tensor):
    if (torch.max(input_tensor) - torch.min(input_tensor)) != 0:
        return (input_tensor - torch.min(input_tensor)) / (torch.max(input_tensor) - torch.min(input_tensor))
    else:
        return input_tensor
    
def get_emo_label(file_path):
    file_split = file_path.split("/")
    file_name = file_split[-1].split("-")

    content = file_name[4]
    emotion = file_name[2]
    emotion = int(emotion) - 1
    return content, emotion

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)


class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    

class EmoFormer(nn.Module):
    def __init__(self, args):
        super(EmoFormer, self).__init__()
        self.args = args
        self.device = args.device
        self.dataset = "vocaset"
        self.encoder = DisentangleNet()
        # ckpt = torch.load('/home/yuxinguo/EmoFormer/Disentangle/train/model/9_pretrain.pt', map_location=torch.device('cpu'))
        # self.encoder.load_state_dict(ckpt["model"])
        self.encoder.train(True)
        self.feature_dim = args.feature_dim
        self.parameter_dim = args.parameter_dim
        self.emo = args.emo
        # self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # wav2vec 2.0 weights initialization
        # self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(512, args.feature_dim)
        self.emotion_vector = nn.Linear(8, args.feature_dim, bias=False)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.biased_mask = init_biased_mask(n_head = 4, max_seq_len = 600, period=args.period)
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period)
        self.motion_encoder = nn.Linear(args.parameter_dim, args.feature_dim)
        self.motion_decoder = nn.Linear(args.feature_dim, args.parameter_dim)
        
        if self.emo == True:
            self.emo_loss = nn.CrossEntropyLoss()

        nn.init.constant_(self.motion_decoder.weight, 0)
        nn.init.constant_(self.motion_decoder.bias, 0)
        
    
    def forward(self, audio, params, criterion, teacher_forcing = True):
        # print(f"input.shape{vertice.shape}")
        frame_num = params.shape[1]
        content_feature, emotion_feature = self.encoder(audio[0])
        emotion_embedding = self.emotion_vector(emotion_feature)
        content_feature_int = linear_interpolation(content_feature, 50, 30, frame_num)
        content_feature_int = self.audio_feature_map(content_feature_int)
        if teacher_forcing:
            decoder_input_emb = emotion_embedding.unsqueeze(1)
            emotion_emb = decoder_input_emb
            params_input = torch.cat((torch.zeros((1,1,self.parameter_dim), device=self.device), params[:, :-1, :]), dim=1)
            decoder_input = self.motion_encoder(params_input)
            decoder_input += emotion_emb
            decoder_input = self.PPE(decoder_input)
            tgt_mask = self.biased_mask[:, :decoder_input.shape[1], :decoder_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, decoder_input.shape[1], content_feature_int.shape[1])
            decoder_output = self.transformer_decoder(decoder_input, content_feature_int, tgt_mask=tgt_mask, memory_mask=memory_mask)
            params_output = self.motion_decoder(decoder_output)
        else:

            for t in range(frame_num):
                if t == 0:
                    decoder_input_emb = emotion_embedding.unsqueeze(1)
                    emotion_emb = decoder_input_emb
                    decoder_input = self.PPE(emotion_emb)
                else:
                    decoder_input = self.PPE(decoder_input_emb)
                tgt_mask = self.biased_mask[:, :decoder_input.shape[1], :decoder_input.shape[1]].clone().detach().to(device=self.device)
                memory_mask = enc_dec_mask(self.device, self.dataset, decoder_input.shape[1], content_feature_int.shape[1])
                decoder_output = self.transformer_decoder(decoder_input, content_feature_int, tgt_mask=tgt_mask, memory_mask=memory_mask)
                params_output = self.motion_decoder(decoder_output)
                new_out = self.motion_encoder(params_output[:, -1, :]).unsqueeze(1)
                new_out += emotion_emb
                decoder_input_emb = torch.cat((decoder_input_emb, new_out), 1)

        params_output = params_output.squeeze(0)
        params_gt = params.squeeze(0)
        shape_gt = params_gt[:, :100]
        expr_gt = params_gt[:,100: 150]
        pose_gt = params_gt[:, 150: 156]
        self.args.batch_size = frame_num
        if self.parameter_dim == 56:
            expr_out = params_output[:, 0: 50]
            jaw_pose_out = params_output[:, 53:56]
            
            expr_out_norm = normalization(expr_out)
            jaw_pose_out_norm = normalization(jaw_pose_out)

            expr_gt_norm = normalization(expr_gt)
            jaw_pose_gt_norm = normalization(pose_gt[:, 3:6])
            loss = criterion(expr_out_norm, expr_gt_norm) + 5 * criterion(jaw_pose_out_norm, jaw_pose_gt_norm)
        else:
            flame = FLAME(self.args)
            flame.to(self.device)
            mesh_gt, _ = flame(shape_gt, expr_gt, pose_gt)
            mesh_gt = mesh_gt.reshape(frame_num, 15069)
            loss = criterion(params_output, mesh_gt)
            
        if self.emo == True:
            _, emo = get_emo_label(audio[0])
            emo = torch.tensor(emo)
            emo = emo.unsqueeze(0)
            emo = emo.cuda()
            # print(emotion_feature.get_device(), emo.get_device())
            
            loss_emo = self.emo_loss(emotion_feature, emo)
        
        else:
            loss_emo = torch.tensor(0)
            
        loss_dict = {}
        loss_dict["loss"] = loss
        loss_dict["loss_emo"] = loss_emo
        
        return loss_dict
    
    def forward_en(self, audio, params, criterion):
        # print(f"input.shape{vertice.shape}")
        frame_num = params.shape[1]
        content_feature, emotion_feature = self.encoder(audio[0])
        emotion_embedding = self.emotion_vector(emotion_feature)
        content_feature_int = linear_interpolation(content_feature, 50, 30, frame_num)
        content_feature_int = self.audio_feature_map(content_feature_int)
        
        return content_feature_int, emotion_embedding
        
    def forward_de(self, audio, params, criterion, content_feature_int, emotion_embedding):
        frame_num = params.shape[1]
        for t in range(frame_num):
            if t == 0:
                decoder_input_emb = emotion_embedding.unsqueeze(1)
                emotion_emb = decoder_input_emb
                decoder_input = self.PPE(emotion_emb)
            else:
                decoder_input = self.PPE(decoder_input_emb)
            tgt_mask = self.biased_mask[:, :decoder_input.shape[1], :decoder_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, decoder_input.shape[1], content_feature_int.shape[1])
            decoder_output = self.transformer_decoder(decoder_input, content_feature_int, tgt_mask=tgt_mask, memory_mask=memory_mask)
            params_output = self.motion_decoder(decoder_output)
            new_out = self.motion_encoder(params_output[:, -1, :]).unsqueeze(1)
            new_out += emotion_emb
            decoder_input_emb = torch.cat((decoder_input_emb, new_out), 1)

        params_output = params_output.squeeze(0)
        params_gt = params.squeeze(0)
        shape_gt = params_gt[:, :100]
        expr_gt = params_gt[:,100: 150]
        pose_gt = params_gt[:, 150: 156]
        self.args.batch_size = frame_num
        if self.parameter_dim == 56:
            expr_out = params_output[:, 0: 50]
            jaw_pose_out = params_output[:, 53:56]
            
            expr_out_norm = normalization(expr_out)
            jaw_pose_out_norm = normalization(jaw_pose_out)

            expr_gt_norm = normalization(expr_gt)
            jaw_pose_gt_norm = normalization(pose_gt[:, 3:6])
            loss = criterion(expr_out_norm, expr_gt_norm) + 5 * criterion(jaw_pose_out_norm, jaw_pose_gt_norm)
        else:
            flame = FLAME(self.args)
            flame.to(self.device)
            mesh_gt, _ = flame(shape_gt, expr_gt, pose_gt)
            mesh_gt = mesh_gt.reshape(frame_num, 15069)
            loss = criterion(params_output, mesh_gt)
            
        if self.emo == True:
            _, emo = get_emo_label(audio[0])
            emo = torch.tensor(emo)
            emo = emo.unsqueeze(0)
            emo = emo.cuda()
            # print(emotion_feature.get_device(), emo.get_device())
            
            loss_emo = self.emo_loss(emotion_feature, emo)
        
        else:
            loss_emo = torch.tensor(0)
            
        loss_dict = {}
        loss_dict["loss"] = loss
        loss_dict["loss_emo"] = 1e-3 * loss_emo
        
        return loss_dict
        
    def predict(self, audio):
        content_feature, emotion_feature = self.encoder(audio[0])
        content_feature_int = linear_interpolation(content_feature, 50, 30)
        emotion_embedding = self.emotion_vector(emotion_feature)
        frame_num = content_feature_int.shape[1]
        content_feature_int = self.audio_feature_map(content_feature_int)
        decoder_input = torch.zeros((1, 1, self.parameter_dim), device=self.device)
        decoder_input = self.motion_encoder(decoder_input)
        
        for t in range(frame_num):
            if t == 0:
                decoder_input_emb = emotion_embedding.unsqueeze(1)
                emotion_emb = decoder_input_emb
                decoder_input = self.PPE(emotion_emb)
            else:
                decoder_input = self.PPE(decoder_input_emb)
            tgt_mask = self.biased_mask[:, :decoder_input.shape[1], :decoder_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, decoder_input.shape[1], content_feature_int.shape[1])
            decoder_output = self.transformer_decoder(decoder_input, content_feature_int, tgt_mask=tgt_mask, memory_mask=memory_mask)
            params_output = self.motion_decoder(decoder_output)
            new_out = self.motion_encoder(params_output[:, -1, :]).unsqueeze(1)
            new_out += emotion_emb
            decoder_input_emb = torch.cat((decoder_input_emb, new_out), 1)
        
        if self.parameter_dim == 56:
            params_output = params_output.squeeze(0)
            shape = torch.zeros((frame_num, 100),device=self.device)
            expr_out = params_output[:, :50]
            pose_out = params_output[:, 50:56]
            self.args.batch_size = frame_num
            flame = FLAME(self.args)
            flame.to(self.device)
            mesh, _ = flame(shape, expr_out, pose_out)
            mesh.reshape(1, frame_num, 15069)

            return mesh
        
        return params_output
    
    def predict_emo(self, audio, emo_vector=None, emo_mid_vector=None):
        
        predict = {}   
        
        content_feature, emotion_feature = self.encoder(audio[0])
        content_feature_int = linear_interpolation(content_feature, 50, 30)
        
        if emo_vector != None:
            emotion_feature = emo_vector
        
        if self.emo == True:
            predict["emotion"] = emotion_feature
        
        emotion_embedding = self.emotion_vector(emotion_feature)
        
        if emo_mid_vector != None:
            emotion_embedding_mid = self.emotion_vector(emo_mid_vector)
        
        frame_num = content_feature_int.shape[1]
        content_feature_int = self.audio_feature_map(content_feature_int)
        decoder_input = torch.zeros((1, 1, self.parameter_dim), device=self.device)
        decoder_input = self.motion_encoder(decoder_input)
        
        for t in range(frame_num):
            if t == 0:
                decoder_input_emb = emotion_embedding.unsqueeze(1)
                emotion_emb = decoder_input_emb
                decoder_input = self.PPE(emotion_emb)
            else:
                decoder_input = self.PPE(decoder_input_emb)
            tgt_mask = self.biased_mask[:, :decoder_input.shape[1], :decoder_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, decoder_input.shape[1], content_feature_int.shape[1])
            decoder_output = self.transformer_decoder(decoder_input, content_feature_int, tgt_mask=tgt_mask, memory_mask=memory_mask)
            params_output = self.motion_decoder(decoder_output)
            new_out = self.motion_encoder(params_output[:, -1, :]).unsqueeze(1)
            
            if t == frame_num // 2 and emo_mid_vector != None:
                print(f"change vector after frame {t}")
                emotion_emb = emotion_embedding_mid.unsqueeze(1)
            
            new_out += emotion_emb
            decoder_input_emb = torch.cat((decoder_input_emb, new_out), 1)
        
        if self.parameter_dim == 56:
            params_output = params_output.squeeze(0)
            shape = torch.zeros((frame_num, 100),device=self.device)
            expr_out = params_output[:, :50]
            pose_out = params_output[:, 50:56]
            self.args.batch_size = frame_num
            flame = FLAME(self.args)
            flame.to(self.device)
            mesh, _ = flame(shape, expr_out, pose_out)
            mesh.reshape(1, frame_num, 15069)
            
            predict["mesh"] = mesh

            return predict
            # return mesh
            
        predict["params_output"] = params_output
        return predict
        # return params_output
