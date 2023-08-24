import argparse
import os
from utils.add_emotions import add_emotions
from run_voca import str2bool

parser = argparse.ArgumentParser(description='Parser for adding Emotions to voca model')
parser.add_argument('--source_path', default='', help='input sequence path')
parser.add_argument('--out_path', default='', help='output path')
parser.add_argument('--flame_model_path', default='./flame/generic_model.pkl', help='path to the FLAME model')
parser.add_argument('--uv_template_fname', default='', help='Path of a FLAME template with UV coordinates')
parser.add_argument('--texture_img_fname', default='', help='Path of the texture image')
parser.add_argument('--emotion', default='happy', help='select emotion')

args = parser.parse_args()
print(f'args:::: {args}')
source_path = args.source_path
out_path = args.out_path
flame_model_fname = args.flame_model_path
uv_template_fname = args.uv_template_fname
texture_img_fname = args.texture_img_fname
emotion = args.emotion

if not os.path.exists(out_path):
    os.makedirs(out_path)

add_emotions(source_path, out_path, flame_model_fname, emotion, uv_template_fname=uv_template_fname, texture_img_fname=texture_img_fname)