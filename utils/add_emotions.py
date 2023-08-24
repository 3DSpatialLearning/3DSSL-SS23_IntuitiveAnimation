import os
import glob
import numpy as np
from scipy.sparse.linalg import cg
from smpl_webuser.serialization import load_model
import chumpy as ch
import torch

from psbody.mesh import Mesh

from utils.FLAME_net import FLAME
from utils.config import get_config
import pyvista as pv
from pathlib import Path


#if torch.cuda.is_available():
#    device = torch.device("cuda:0")  # Specify the GPU device index (e.g., 0)
#else:
device = torch.device("cpu")


config = get_config()
radian = np.pi/180.0
flamelayer = FLAME(config)
flamelayer = flamelayer.to(device)

def add_emotions(source_path, out_path, flame_model_fname, emotion, template, uv_template_fname='', texture_img_fname=''):
    '''
    Function to add emotions to the voca output, which is basicly a merge of two meshes
    '''
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    sequence_fnames = sorted(glob.glob(os.path.join(source_path, '*.obj')))
    num_frames = len(sequence_fnames)
    if num_frames == 0:
        print('No sequence meshes found')
        return

    model = load_model(flame_model_fname)

    print('Optimize for template identity parameters')
    template_mesh = Mesh(filename=template)
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

    np.save(out_path, {'shape': betas, 'pose': model_pose, 'expression': model_exp})

    params = np.load("./emotion_output/emotion_output.npy", allow_pickle=True).item()

    shape = params['shape']
    pose = params['pose']
    exp = params['expression']

    model.betas[:300] = shape
    num_frames = pose.shape[0]
    for frame_idx in range(num_frames):
        model.pose[:] = pose[frame_idx, :]
        model.betas[300:] = exp[frame_idx, :]
        out_fname = Path("/home/finnschaefer/voca/emotion_output/meshes/%05d_FLAME.obj" % frame_idx)
        Mesh(model.r, model.f).write_obj(out_fname)
    
    #output_sequence_meshes(predicted_vertices, Mesh(model.v_template, model.f), out_path, uv_template_fname=uv_template_fname, texture_img_fname=texture_img_fname)
    
def createMesh(emotion):
    params = {}
    print(f'emotion: {emotion}')
    if emotion == 'happy':
        print(f'entering happy emotion')
        expression_params = torch.tensor([[
            1.62,
            0.02,
            0.0,
            1.57,
            1.71,
            1.04,
            1.67,
            1.69,
            -1.47,
            2.0,
        ]])
    elif emotion == 'sad':
        print(f'entering sad emotion')
        expression_params = torch.tensor([[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.22,
            -1.73,
            -1.33,
            0.22,
            -2.0,
            ]])
    elif emotion == 'surprised':
        print(f'entering surprised emotion')
        expression_params = torch.tensor([[
            -1.11,
            0.0,
            0.0,
            1.80,
            1.0,
            -0.60,
            -0.87,
            -0.98,
            -0.89,
            -1.24,
            ]])
    elif emotion == 'angry':
        print(f'entering angry emtotion')
        expression_params = torch.tensor([[
            2.0,
            2.0,
            0.0,
            1.67,
            -2.0,
            -2.0,
            -1.04,
            2.0,
            -1.16,
            2.0,
            ]])
    elif emotion == 'exited':
        print(f'entering exited emotion')
        expression_params = torch.tensor([[
            2.0,
            2.0,
            0.0,
            0.96,
            2.0,
            1.16,
            2.0,
            2.0,
            -1.07,
            2.0,
            ]])
    elif emotion == 'fear':
        print(f'entering fear emotion')
        expression_params = torch.tensor([[
            -1.53,
            -0.44,
            0.0,
            0.60,
            2.0,
            1.93,
            -2.00,
            -0.51,
            2.0,
            -0.49,
            ]])
    elif emotion == 'disappointed':
        print(f'entering disappointed emotion')
        expression_params = torch.tensor([[
            0.0,
            2.0,
            0.0,
            0.60,
            -2.0,
            -1.51,
            2.00,
            0.67,
            2.00,
            -2.00,
            ]])
    elif emotion == 'frustrated':
        print(f'entering frustrated emotion')
        expression_params = torch.tensor([[
            2.0,
            2.0,
            0.0,
            1.82,
            -2.0,
            -2.00,
            1.31,
            2.00,
            2.00,
            -2.00,
            ]])
    elif emotion == 'neutral':
        print(f'entering neutral emotion')
        expression_params = torch.tensor([[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            ]])
    

    pose_params = np.array([[
            0.0 * radian,
            0.0 * radian,
            0.0 * radian,
            0.0 * radian,
            0.0 * radian,
            0.0 * radian,
    ]])

    shape_params = torch.tensor([[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
    ]], dtype=torch.float32)

    #if torch.cuda.is_available():
    #    device = torch.device("cuda:0")  # Specify the GPU device index (e.g., 0)
    #else:
    device = torch.device("cpu")

    shape_params = torch.hstack((shape_params, torch.zeros(1,90)))
    pose_params = torch.tensor(pose_params, dtype=torch.float32)
    expression_params = torch.hstack((expression_params, torch.zeros(1, 40, dtype=torch.float32)))

    shape_params = shape_params.to(device)
    pose_params = pose_params.to(device)
    expression_params = expression_params.to(device)
    params['shape_params'] = shape_params
    params['pose_params'] = pose_params
    params['expression_params'] = expression_params

    return params, emotion



def flame_forward(shape_params, expression_params, pose_params, emotion, i):
    print("generating first vertice and landmark")
    vertice, _ = flamelayer(shape_params, expression_params, pose_params)
    print("creating faces")
    faces = np.hstack(np.insert(flamelayer.faces, 0, values=3, axis=1))
    print("cerating vertices")
    vertices = vertice[0].detach().cpu().numpy().squeeze()
    print("creating mesh")
    mesh = pv.PolyData(vertices, faces)
    if not os.path.exists('./emoca_template/' + emotion):
        os.makedirs('./emoca_template/' + emotion)
    output_path = './emoca_template/' + emotion + f'/{i+1}_template.ply'
    print(f'output_path: {output_path}')
    mesh.save(output_path)



