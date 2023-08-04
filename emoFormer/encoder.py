import torch.nn as nn
import torch
import torchaudio
from transformers import Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor

def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

class EmotionEncoder(nn.Module):

    def __init__(self):
        super(EmotionEncoder, self).__init__()
        self.type = type
        
        self.projector = nn.Linear(1024, 1024, bias=True)
        self.classifier = nn.Linear(1024, 8, bias=True)
        # self.relu = nn.ReLU()

    def forward(self, x):
        
        x = torch.mean(x, dim=1)
        x = self.projector(x)
        x = torch.tanh(x)
        x = self.classifier(x)
        
        return x
    
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
    
class DisentangleNet(nn.Module):

    def __init__(self):
        super(DisentangleNet, self).__init__()

        # audio feature extractor - Wav2Vec2.0
        
        self.audio_encoder = Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.sampling_rate = self.feature_extractor.sampling_rate
        self.con_encoder = ContentEncoder()
        self.emo_encoder = EmotionEncoder()
        # self.decoder = Decoder()
        # self.classify = Classify()

    def forward(self, path):
        
        
        speech = speech_file_to_array_fn(path, self.sampling_rate)
        if speech.shape[0]>1 and speech.ndim > 1: # torchload unknown error
            speech = speech[0, :]
        
        inputs = self.feature_extractor(speech, sampling_rate=self.sampling_rate, return_tensors="pt", padding=False) # 84800
        input = inputs.input_values.to(self.audio_encoder.device)
        output = self.audio_encoder(input)
        feat = output.last_hidden_state
        
        con = self.con_encoder(feat) 
        emo = self.emo_encoder(feat)

        return con, emo
    
if __name__ == "__main__":

    model = DisentangleNet()
    path = "/home/yuxinguo/data/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"

    ckpt = torch.load('/home/yuxinguo/EmoFormer/Disentangle/train/model/9_pretrain.pt', map_location=torch.device('cpu'))
    model.load_state_dict(ckpt["model"])

    con, _ = model(path)

    print(con.shape)