from dataProcessing.config import get_config
from dataProcessing.FLAME import FLAME
import numpy as np
import torch
# import pyvista as pv

def mesh_from_params(params_file):
    config = get_config()
    flame_params = np.load(params_file)
    shape_params = torch.from_numpy(flame_params["shape"])
    expression_params = torch.from_numpy(flame_params["expr"])
    # pose_params = torch.from_numpy(flame_params["rotation"])
    # neck_pose = torch.from_numpy(flame_params["neck_pose"])
    # eye_pose = torch.from_numpy(flame_params["eyes_pose"])
    # transl = torch.from_numpy(flame_params["translation"])
    # jaw_pose = torch.from_numpy(flame_params["jaw_pose"])
    # pose_params_1 = torch.cat((pose_params, jaw_pose),dim=1)
    pose_params = torch.from_numpy(flame_params["pose"])
    frames = shape_params.shape[0]
    config.batch_size = frames
    config.shape_params = shape_params.shape[1]
    config.expression_params = expression_params.shape[1]
    config.pose_params = pose_params.shape[1]
    model = FLAME(config)
    pose_params[:,:3] = 0
    shape_params = torch.from_numpy(np.zeros_like(shape_params))
    vertices, _ = model(shape_params, expression_params, pose_params)
    faces = model.faces

    return vertices.reshape(frames, 15069).numpy().squeeze(), faces

vertices, faces = mesh_from_params("/mnt/hdd/datasets/RAVDESS/npzs/01-01-06-02-01-01-01.npz")
print(vertices.shape)

np.save("test_neutral_shape.npy", vertices)
# faces = np.hstack(np.insert(faces, 0, values=3, axis=1))

# for i in range(vertices.shape[0]):
#     mesh = pv.PolyData(vertices[i], faces)
#     print(f"saving: test/frame{i:03d}.ply")
#     mesh.save(f"test/frame{i:03d}.ply")