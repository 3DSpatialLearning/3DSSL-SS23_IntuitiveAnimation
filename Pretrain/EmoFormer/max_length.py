# 获得RAVDESS最长的文件

import os
import librosa

path = "/home/yuxinguo/data/RAVDESS/Video_Speech_"
path_list = [path+"Actor_"+str(i).zfill(2)+"/Actor_"+str(i).zfill(2) for i in range(1, 25)]

max_length = 0.0

for p in path_list:
    print("Nos is", p)
    for item in os.listdir(p):
        file = os.path.join(p, item)
        print(file)
        audio, samplingrate = librosa.load(file, sr=16000)
        length = librosa.get_duration(audio, sr = samplingrate)
    
        if length > max_length:
            max_length = length
        
print(max_length)
