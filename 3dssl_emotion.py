import os
import glob
from psbody.mesh import Mesh
from utils.add_emotions import add_emotions, createMesh, flame_forward
from pathlib import Path
import numpy as np
import chumpy as ch
import torch
from scipy.sparse.linalg import cg
from dataProcessing.config import get_config
from dataProcessing.flameMesh import mesh_from_params
from dataProcessing.FLAME import FLAME
import pickle
import wave
from scipy.io.wavfile import read
import io
import argparse
from smpl_webuser.serialization import load_model

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

def compute_FLAME_params(source_path, params_out_fname, flame_model_fname, template_fname):
    '''
    Load a template and an existing animation sequence in "zero pose" and compute the FLAME shape, jaw pose, and expression paramters. 
    Outputs one set of shape paramters for the entire sequence, and pose and expression parameters for each frame.
    :param source_path:         path of the animation sequence (files must be provided in OBJ file format)
    :param params_out_fname     output path of the FLAME paramters file
    :param flame_model_fname:   path of the FLAME model
    :param template_fname       "zero pose" template used to generate the sequence
    '''

    if not os.path.exists(os.path.dirname(params_out_fname)):
        os.makedirs(os.path.dirname(params_out_fname))
    
    # Load sequence files
    sequence_fnames = sorted(glob.glob(os.path.join(source_path, '*.obj')))
    num_frames = len(sequence_fnames)
    if num_frames == 0:
        print('No sequence meshes found')
        return

    model = load_model(flame_model_fname)

    print('Optimize for template identity parameters')
    template_mesh = Mesh(filename=template_fname)
    ch.minimize(template_mesh.v - model, x0=[model.betas[:300]], options={'sparse_solver': lambda A, x: cg(A, x, maxiter=2000)[0]})

    betas = model.betas.r[:300].copy()
    model.betas[:] = 0.

    model.v_template[:] = template_mesh.v
    
    model_pose = np.zeros((num_frames, model.pose.shape[0]))
    model_exp = np.zeros((num_frames, 100))

    for frame_idx in range(num_frames):
        print('Process frame %d/%d' % (frame_idx+1, num_frames))
        model.betas[:] = 0.
        model.pose[:] = 0.
        frame_vertices = Mesh(filename=sequence_fnames[frame_idx]).v
        # Optimize for jaw pose and facial expression
        ch.minimize(frame_vertices - model, x0=[model.pose[6:9], model.betas[300:]], options={'sparse_solver': lambda A, x: cg(A, x, maxiter=2000)[0]})
        model_pose[frame_idx] = model.pose.r.copy()
        model_exp[frame_idx] = model.betas.r[300:].copy()

    np.save(params_out_fname, {'shape': betas, 'pose': model_pose, 'expression': model_exp})

source_path = "./animation_output/meshes"
out_path = "./emotion_output"
flame_model_fname = "./flame/generic_model.pkl"
emotions = ['angry', 'exited', 'fear', 'sad', 'surprised', 'frustrated', 'happy', 'disappointed', 'neutral']
template ='./fear_template.ply'
emoca_path = '/mnt/hdd/datasets/RAVDESS/Video_Speech_Actor_01/Actor_01/01-01-01-01-01-02-01/EMOCA_v2_lr_mse_20'

#path = "/home/finnschaefer/voca/training_data/targets/Actor_01/new_target_Actor_01/"
#folder = os.listdir(path)
#print(folder)
#
#for path_ in folder:
#
#    compute_FLAME_params(path+path_  + "/meshes", "./training_data/targets/Autor_01/" + path_, flame_model_fname, "/home/finnschaefer/voca/template/FLAME_sample.ply")
#
#exit()

if not os.path.exists(out_path):
    os.makedirs(out_path)

#print(f'test_path: {Path(out_path)}')

#path_to_template = './training_data/test/'
#sort_list = ['01-01-01-01-01-01-01.npy','01-01-01-01-01-01-02.npy', '01-01-01-01-01-01-03.npy', '01-01-01-01-01-01-04.npy', '01-01-01-01-01-01-05.npy', '01-01-01-01-01-01-06.npy', '01-01-01-01-01-01-08.npy', '01-01-01-01-01-01-09.npy']
#template_dict = {}
#for i in sort_list:
#    vert = np.load(path_to_template + i)
#    #vert.reshape(-1, 5023, 3)
#    if i[-6:] == '01.npy':
#        template_dict['Actor_01'] = vert[0]
#    if i[-6:] == '02.npy':
#        template_dict['Actor_02'] = vert[0]
#    if i[-6:] == '03.npy':
#        template_dict['Actor_03'] = vert[0]
#    if i[-6:] == '04.npy':
#        template_dict['Actor_04'] = vert[0]
#    if i[-6:] == '05.npy':
#        template_dict['Actor_05'] = vert[0]
#    if i[-6:] == '06.npy':
#        template_dict['Actor_06'] = vert[0]
#    if i[-6:] == '08.npy':
#        template_dict['Actor_08'] = vert[0]
#
##print(template_dict)
#with open("./training_data/template_RAVDESS.pkl", "wb") as template:
#    pickle.dump(template_dict, template)

#for i in emotions:
#    print(i)
#    params, emotion = createMesh(i)
#    print(f"calling flame forward pass\n{params['shape_params'].shape}")
#    #flame_forward(params['shape_params'], params['expression_params'], params['pose_params'], i)
#
#emotion = 'happy'
#add_emotions(source_path, out_path, flame_model_fname, emotion, template)

#directory = '/mnt/hdd/datasets/RAVDESS/Video_Speech_Actor_05/Actor_05/'  # Replace with the path to your directory


#####IMPORTANT####
#audio_directory = "./training_data/audio/"
#actor_list = [folder for folder in os.listdir(audio_directory) if os.path.isdir(os.path.join(audio_directory, folder))]
#actor_list = sorted(actor_list)
#actor_paths = [os.path.join(audio_directory, actor_list[i]) for i in range(len(actor_list))]
#actor_dict = {}
#audio_dict = {}
#for i in range(len(actor_paths)):
#    current_files = sorted(os.listdir(actor_paths[i]))
#    if actor_list[i] == 'Actor_07':
#        continue
#    if actor_list[i] == 'Actor_09':
#        break
#    for j in range(len(current_files)):
#        #if current_files[j] == '03-01-02-01-01-02-01.wav' or '03-01-08-01-02-02-01.wav':
#        #    continue
#        wav_file = actor_paths[i] + '/' + current_files[j]
#        sample_rate, audio_data = read(wav_file)
#        audio_array = np.array(audio_data, dtype=np.int16)
#        audio_dict[current_files[j]] = {'audio': audio_array, 'sample_rate': sample_rate}
#    
#    actor_dict[actor_list[i]] = audio_dict
#
#print(actor_dict)
#with open('./training_data/raw_audio_fixed_RAVDESS.pkl', 'wb') as file:
#    pickle.dump(actor_dict, file)
#exit()
directory = "/home/finnschaefer/voca/training_data/targets/training/"

folder_list = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
file_names = sorted(os.listdir(directory))
folder_list = sorted(folder_list)
#print(file_names)
#print(f'folder_list: {folder_list}')
init_shape = np.zeros((len(file_names), 100))
ctr = 0
seq_dict = {}
ctr_ = 0
for file in file_names:
    config = get_config()
    current = np.load(directory + file, allow_pickle=True).item()
    shape = torch.from_numpy(current['shape'])
    exp = torch.from_numpy(current['expression'])
    pose = torch.from_numpy(current['pose'])
    print(shape.shape)
    print(exp.shape)
    print(pose.shape)
    init_shape[ctr] = current['shape'][0]
    ctr+=1  
    num_frames = shape.shape[0]
    config.batch_size = num_frames
    config.shape_params = shape.shape[1]
    config.expression_params = exp.shape[1]
    config.pose_params = pose.shape[1]
    model = FLAME(config)
    pose[:,:3] = 0
    shape = torch.from_numpy(np.zeros_like(shape))
    vertices, _ = model(shape, exp, pose)
    vertices_template = vertices
    vertices.reshape(num_frames, 15069).numpy().squeeze
    frame_dict = {}
    for i in range(num_frames):
        ctr_ +=1
        frame_dict[i] = ctr_
    seq_dict["03" + file.replace(".npy", ".wav")[2:]] = frame_dict
    faces = model.faces
    np.save(f'./training_data/test/{file.split(".")[0]}.npy', vertices)
#print(seq_dict.keys())
exit()
audio_directory = "./training_data/audio/"
actor_list = [folder for folder in os.listdir(audio_directory) if os.path.isdir(os.path.join(audio_directory, folder))]
actor_list = sorted(actor_list)
actor_paths = [os.path.join(audio_directory, actor_list[i]) for i in range(len(actor_list))]

array2seq_dict = {}

for i in range(len(actor_paths)):
    current_files = sorted(os.listdir(actor_paths[i]))
    seq_dict_ = {}

    for j in range(len(current_files)):
        if current_files[j][-6:] == '07.wav' or current_files[j] == '03-01-07-02-01-01-13.wav' or current_files[j][-6:] == '18.wav' or current_files[j][-6:] == '19.wav' or current_files[j][-6:] == '20.wav' or current_files[j][-6:] == '21.wav' or current_files[j][-6:] == '22.wav' or current_files[j][-6:] == '23.wav' or current_files[j][-6:] == '24.wav':
            continue

        seq_dict_[current_files[j]] = seq_dict["03" + current_files[j][2:]]

    array2seq_dict[actor_list[i]] = dict(seq_dict_)  # Create a new dictionary for each key

with open("./training_data/subj_seq_to_idx_RAVDESS.pkl", "wb") as file:
    pickle.dump(array2seq_dict, file)

print(array2seq_dict['Actor_01'].keys())

directory = '/home/finnschaefer/voca/training_data/test/'
folder_list = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
file_names = sorted(os.listdir(directory))
folder_list = sorted(folder_list)
stacked_array = None
overall_dict = {}
dict_dicht = {}
for file in file_names:
    array = np.load(directory + file)
    if stacked_array is None:
        stacked_array = array
    else:
        stacked_array = np.vstack((stacked_array, array))

np.save('data_verts_training.npy', stacked_array)

exit()
for j in range(len(folder_list)):
    folder_path = directory + folder_list[j]# + '/EMOCA_v2_lr_mse_20/'
    print(f'folder_path: {folder_path}')
    exit()
    if folder_path == '/mnt/hdd/datasets/RAVDESS/Video_Speech_Actor_01/Actor_01/01-01-01-01-01-02-01/EMOCA_v2_lr_mse_20/':
        continue
    file_count = [folder for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]
    print(f'file_count: {file_count}')
    for i in range(len(file_count)):
        path = emoca_path + f'/000{str(i+1).zfill(3)}_000/'
        #print(f'path: {path}')
        if path == '/mnt/hdd/datasets/RAVDESS/Video_Speech_Actor_01/Actor_01/01-01-01-01-01-02-01/EMOCA_v2_lr_mse_20/000100_000/':
            break
        shapepath = path + 'shape.npy'
        posepath = path + 'pose.npy'
        exppath = path + 'exp.npy'
        shapeparams = torch.zeros((1, 100))
        expressionparams = torch.tensor(np.reshape(np.load(exppath), (1,50)))
        poseparams = torch.zeros((1, 6))
        #print(f'shapeparams: {shapeparams.shape}')
        print(f"Processing Frame: {i+1}")
        flame_forward(shape_params=shapeparams, expression_params=expressionparams, pose_params=poseparams, emotion='Actor5/' + str(folder_list[j]), i=i)