import numpy as np
import torch
from FLAME import FLAME
import pyvista as pv
from config import get_config

config = get_config()
radian = np.pi/180.0
flamelayer = FLAME(config)
flamelayer.cuda()

# forward pass of FLAME, from parameters to mesh
def createMesh(shape_params, expression_params, pose_params):
    vertice, _ = flamelayer(shape_params, expression_params, pose_params)
    faces = np.hstack(np.insert(flamelayer.faces, 0, values=3, axis=1))
    vertices = vertice[0].detach().cpu().numpy().squeeze()
    mesh = pv.PolyData(vertices, faces)
    return mesh

class ModelControl:
    def __init__(self, mesh):
        self.output = mesh

        self.kwargs = {
            'rotation_1': 0.0,
            'rotation_2': 0.0,
            'rotation_3': 0.0,
            'jaw_1': 0.0,
            'jaw_2': 0.0,
            'jaw_3': 0.0,
            'expression_1': 0.0,
            'expression_2': 0.0,
            'expression_3': 0.0,
            'expression_4': 0.0,
            'expression_5': 0.0,
            'expression_6': 0.0,
            'expression_7': 0.0,
            'expression_8': 0.0,
            'expression_9': 0.0,
            'expression_10': 0.0,
            'shape_1': 0.0,
            'shape_2': 0.0,
            'shape_3': 0.0,
            'shape_4': 0.0,
            'shape_5': 0.0,
            'shape_6': 0.0,
            'shape_7': 0.0,
            'shape_8': 0.0,
            'shape_9': 0.0,
            'shape_10': 0.0,
        }


    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def update(self):
        # This is where you call your simulation
        self.customShape = torch.tensor([[
            self.kwargs['shape_1'], 
            self.kwargs['shape_2'], 
            self.kwargs['shape_3'], 
            self.kwargs['shape_4'], 
            self.kwargs['shape_5'], 
            self.kwargs['shape_6'],
            self.kwargs['shape_7'], 
            self.kwargs['shape_8'], 
            self.kwargs['shape_9'], 
            self.kwargs['shape_10']
            ]])
        
        self.customExpression = torch.tensor([[
            self.kwargs['expression_1'], 
            self.kwargs['expression_2'], 
            self.kwargs['expression_3'], 
            self.kwargs['expression_4'], 
            self.kwargs['expression_5'], 
            self.kwargs['expression_6'],
            self.kwargs['expression_7'], 
            self.kwargs['expression_8'], 
            self.kwargs['expression_9'], 
            self.kwargs['expression_10']
            ]])

        # Creating shape parameters
        self.shape_params =torch.hstack((self.customShape, torch.zeros(1, 90))).cuda()

        # Creating global pose
        pose_params_numpy = np.array([[
            self.kwargs['rotation_1']*radian,
            self.kwargs['rotation_2']*radian,
            self.kwargs['rotation_3']*radian,
            self.kwargs['jaw_1']*radian,
            self.kwargs['jaw_2']*radian,
            self.kwargs['jaw_3']*radian,
            ]], dtype=np.float32)
        self.pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).cuda()

        # Cerating expression parameters
        self.expression_params = torch.hstack((self.customExpression, torch.zeros(1, 40, dtype=torch.float32))).cuda()
        result = createMesh(self.shape_params, self.expression_params, self.pose_params)
        self.output.copy_from(result)
        return