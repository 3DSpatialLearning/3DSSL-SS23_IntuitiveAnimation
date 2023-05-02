import numpy as np
import torch
import pyvista as pv
from fit_lmk3d import run_fitting

mesh = pv.read("./data/fit_lmk3d_result.obj")

def refit(point, i):
    lmks[i] = point
    mesh_v, mesh_f = run_fitting(lmks)
    faces = np.hstack(np.insert(mesh_f, 0, values=3, axis=1))
    mesh.points = mesh_v
    mesh.faces = faces

p = pv.Plotter()

lmks = np.load("./data/scan_lmks.npy")

# lmks = np.fromfile("./data/startingLmks.txt", sep = " ")
# lmks = np.reshape(lmks, (68,3))


p.camera_position = "xy"
p.add_sphere_widget(refit, center=lmks, radius=0.001, color="red", test_callback=False)
p.add_mesh(mesh)
p.show()