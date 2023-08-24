from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model 
)

import torchaudio
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor


def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

path = "/home/yuxinguo/data/RAVDESS/Audio_Speech_Actors_01-24/Actor_02/03-01-02-02-01-02-02.wav"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
sampling_rate = feature_extractor.sampling_rate
speech = speech_file_to_array_fn(path, sampling_rate)
if speech.shape[0]>1 and speech.ndim > 1: # torchload unknown error
            speech = speech[0, :]
            
inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding="max_length", max_length=84800) # 84800
input = inputs.input_values

print(input.shape)

model1 = Wav2Vec2Model.from_pretrained("lighteternal/wav2vec2-large-xlsr-53-greek")
model2 = Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
model3 = Wav2Vec2Model.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
model4 = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition") 

audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

output = model3(input)
code = audio_encoder(input)

print(type(output), output.last_hidden_state.shape)
print(type(code), code.last_hidden_state.shape)



# print("1", model1)
# print("2", model2)
# print("3", model3)
# print("4", model4) 

