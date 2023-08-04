import os, shutil
import cv2
import tempfile
import numpy as np
from subprocess import call
import argparse
os.environ['PYOPENGL_PLATFORM'] = 'osmesa' #egl
import pyrender
import trimesh
from psbody.mesh import Mesh
import glob
from emoFormer import EmoFormer
import torch
import random

def file_with_other_emo(abs_path):
    "/home/yuxinguo/data/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-01-03-01-01-01-01.wav"
    emo_list = ["01", "02", "03", "04", "05", "06", "07", "08"]
    
    file_path_split = abs_path.split("/")
    name_split = file_path_split[-1].split("-")
    
    e1 = name_split[2]
    itensity = name_split[3]
    
    e2 = e1

    if itensity == "02":
        while (e2 == e1 or e2 == "01"): 
            e2 = random.choice(emo_list)
        
    else:
        while e2 == e1:
            e2 = random.choice(emo_list)
    
    e2 = "03"
    
    name_split[2] = e2
    name = "-".join(name_split)
    
    file_path_split[-1] = name
    file_path = "/".join(file_path_split)
    
    return file_path, e1, e2
    
def render_mesh_helper(args,mesh, t_center, rot=np.zeros(3), tex_img=None,  z_offset=0):
    if args.dataset == "BIWI":
        camera_params = {'c': np.array([400, 400]),
                         'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                         'f': np.array([4754.97941935 / 8, 4754.97941935 / 8])}
    elif args.dataset == "vocaset":
        camera_params = {'c': np.array([400, 400]),
                         'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                         'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
    
    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8 
            )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)

    if args.background_black:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])#[0, 0, 0] black,[255, 255, 255] white
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])#[0, 0, 0] black,[255, 255, 255] white
    
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def render_sequence_meshes(args,sequence_vertices, template, out_path,filename,vt, ft ,tex_img):
    num_frames = sequence_vertices.shape[0]
    file_name_pred = filename
    tmp_video_file_pred = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path)
    writer_pred = cv2.VideoWriter(tmp_video_file_pred.name, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (800, 800), True)

    center = np.mean(sequence_vertices[0], axis=0)
    video_fname_pred = os.path.join(out_path, file_name_pred+'.mp4')
    for i_frame in range(num_frames):
        render_mesh = Mesh(sequence_vertices[i_frame], template.f)
        if vt is not None and ft is not None:
            render_mesh.vt, render_mesh.ft = vt, ft
        pred_img = render_mesh_helper(args,render_mesh, center, tex_img=tex_img)
        pred_img = pred_img.astype(np.uint8)
        img = pred_img
        writer_pred.write(img)

    writer_pred.release()
    cmd = ('ffmpeg' + ' -i {0} -pix_fmt yuv420p -qscale 0 {1}'.format(
       tmp_video_file_pred.name, video_fname_pred)).split()
    call(cmd)

def main():
    parser = argparse.ArgumentParser(description='EmoFormer: Transformer based audio driven face animation network with emotion')
    # FLAME arguments
    parser.add_argument('--flame_model_path', type=str, default='../model/generic_model.pkl', help='flame model path')
    parser.add_argument('--static_landmark_embedding_path', type=str, default='../model/flame_static_embedding.pkl', help='Static landmark embeddings path for FLAME')
    parser.add_argument('--dynamic_landmark_embedding_path', type=str, default='../model/flame_dynamic_embedding.npy', help='Dynamic contour embedding path for FLAME')
    parser.add_argument('--shape_params', type=int, default=100, help='the number of shape parameters')
    parser.add_argument('--expression_params', type=int, default=50, help='the number of expression parameters')
    parser.add_argument('--pose_params', type=int, default=6, help='the number of pose parameters')
    parser.add_argument('--use_face_contour', default=True, type=bool, help='If true apply the landmark loss on also on the face contour.')
    parser.add_argument('--use_3D_translation', default=True, type=bool, help='If true apply the landmark loss on also on the face contour.')
    parser.add_argument('--optimize_eyeballpose', default=True, type=bool, help='If true optimize for the eyeball pose.')
    parser.add_argument('--optimize_neckpose', default=True, type=bool, help='If true optimize for the neck pose.')
    parser.add_argument('--num_worker', type=int, default=4, help='pytorch number worker.')
    parser.add_argument('--batch_size', type=int, default=328, help='Training batch size.')
    parser.add_argument('--ring_margin', type=float, default=0.5, help='ring margin.')
    parser.add_argument('--ring_loss_weight', type=float, default=1.0, help='weight on ring loss.')

    # EmoFormer arguments
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--parameter_dim", type=int, default=5023*3, help='number of parameters - 5023*3 for vertices output, 56 for FLAME parameters output')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')

    # Test arguments
    parser.add_argument("--wav_path", type=str, default="wav/*.wav", help="path to the test audio, for multiple inputs, use *.wav")
    parser.add_argument("--model_path", type=str, default="save/vertices_output_01/25_model.pth", help="path to the saved model")
    parser.add_argument("--result_path", type=str, default="result", help='path to save the predictions')
    parser.add_argument("--save_videos", type=bool, default=True, help="If true save rendered video")
    parser.add_argument("--save_mesh", type=bool, default=True, help="If true save predicted mesh")
    parser.add_argument("--device", type=str, default="cuda", help="cuda for gpu, cpu for cpu")

    # Render arguments
    parser.add_argument("--render_template_path", type=str, default="../model/FLAME_sample.ply", help='path of the mesh in FLAME/BIWI topology')
    parser.add_argument('--background_black', type=bool, default=True, help='whether to use black background')
    parser.add_argument('--fps', type=int,default=30, help='frame rate - 30 for vocaset; 25 for BIWI')
    
    parser.add_argument("--emo", type=bool, default=False, help="test see if the model capture emotion")
    parser.add_argument("--emo_switch", type=bool, default=False, help="emotion switch test")

    args = parser.parse_args()

    model = EmoFormer(args)
    model.to(args.device)
    model.eval()
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    wavs = glob.glob(args.wav_path)
    print(f"Loading render template {args.render_template_path}")
    template = Mesh(filename=args.render_template_path)
    result_path = args.result_path
    
    if args.emo_switch == True:
        
        wavs_list = []
        wavs_list.append(wavs[0])
        
        wav_other, e1, e2 = file_with_other_emo(wavs[0])
        
        wavs_list.append(wav_other)
        wavs = wavs_list
        
        subname = f"{e1}-{e2}"
        
        result_path = os.path.join(result_path, subname)
        
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    
    tmp = {}
    tmp["file"] = []
    tmp["wav"] = []
    tmp["emo"] = []
    
    print("2 files:", wavs)
    
    for wav in wavs:
        
        filename = os.path.basename(wav).split(".")[0]
        
        if args.emo == False:
            pred = model.predict_emo((wav,)).detach().cpu().squeeze(0).numpy()
            pred = pred.reshape(-1,5023,3)
        else:
            pred = model.predict_emo((wav,))
            pred_emo = pred["emotion"]
            pred = pred["params_output"].detach().cpu().squeeze(0).numpy()
            pred = pred.reshape(-1,5023,3)
            
            if args.emo_switch == True:
                tmp["file"].append(filename)
                tmp["wav"].append(wav)
                tmp["emo"].append(pred_emo)

        if args.save_mesh:
            np.save(os.path.join(result_path, f"{filename}.npy"), pred)

        if args.save_videos:
            vt, ft = None, None
            tex_img = None

            render_sequence_meshes(args, pred, template, result_path, filename, vt, ft, tex_img)
            
    if args.emo_switch == True:
            
        pred_0c_1e = model.predict_emo((tmp["wav"][0],), tmp["emo"][1])
        # pred_0c_1e = model.predict_emo((tmp["wav"][0],), tmp["emo"][1], tmp["emo"][0])
        pred_1c_0e = model.predict_emo((tmp["wav"][1],), tmp["emo"][0])
        
        pred_0c_1e = pred_0c_1e["params_output"].detach().cpu().squeeze(0).numpy()
        pred_0c_1e = pred_0c_1e.reshape(-1,5023,3)
        
        pred_1c_0e = pred_1c_0e["params_output"].detach().cpu().squeeze(0).numpy()
        pred_1c_0e = pred_1c_0e.reshape(-1,5023,3)
        
        filename_0 = tmp['file'][0] + "_swicth"
        filename_1 = tmp['file'][1] + "_switch"
        
            
        if args.save_mesh:
            np.save(os.path.join(result_path,f"{filename_0}.npy"), pred_0c_1e)
            np.save(os.path.join(result_path, f"{filename_1}.npy"), pred_1c_0e)
                
        if args.save_videos:
            vt, ft = None, None
            tex_img = None
            
            print("filename_0:", filename_0)
            render_sequence_meshes(args, pred_0c_1e, template, result_path, filename_0, vt, ft, tex_img)
            
            vt, ft = None, None
            tex_img = None
            
            print("filename_1:", filename_1)
            render_sequence_meshes(args, pred_1c_0e, template, result_path, filename_1, vt, ft, tex_img)
            
        

if __name__ == "__main__":
    main()