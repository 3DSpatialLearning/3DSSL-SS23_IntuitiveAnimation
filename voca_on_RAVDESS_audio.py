import os
import glob
import argparse
from utils.inference import inference

def str2bool(val):
    if isinstance(val, bool):
        return val
    elif isinstance(val, str):
        if val.lower() in ['true', 't', 'yes', 'y']:
            return True
        elif val.lower() in ['false', 'f', 'no', 'n']:
            return False
    return False

audiodata = sorted(glob.glob(os.path.join("training_data/audio/Actor_01", '*.wav')))

for count, audio in enumerate(audiodata):

    file_name = audio.split("/")[-1].split(".")[0]
    os.makedirs(f'voca_out/{file_name}', exist_ok=True)
    os.system("python run_voca.py --tf_model_fname './model/gstep_52280.model' --ds_fname './ds_graph/output_graph.pb' --audio_fname {} --template_fname './template/FLAME_sample.ply' --condition_idx 3 --out_path 'voca_out/{}'".format(audio, file_name))
