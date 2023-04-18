import numpy as np
import torch
import pyvista as pv
import modelControl


# callback function used to iterativly add sliders
def updateParameter(parameter, value):
    engine(parameter, value)

# creating initial parameters
shape_params_init = torch.zeros(1, 100).cuda()
pose_params_numpy = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
pose_params_init = torch.tensor(pose_params_numpy, dtype=torch.float32).cuda()
expression_params_init = torch.zeros(1, 50, dtype=torch.float32).cuda()

# creating initail mesh
starting_mesh = modelControl.createMesh(shape_params_init, expression_params_init, pose_params_init)
engine = modelControl.ModelControl(starting_mesh)

# setting up plotter and loading initial mesh
p = pv.Plotter()
p.camera_position = 'xy'
p.add_mesh(starting_mesh, show_edges=False)

# setting slider width
pv.global_theme.slider_styles.modern.slider_width = 0.02
pv.global_theme.slider_styles.modern.tube_width = 0.02

# adding sliders for jaws
p.add_slider_widget(
    callback=lambda value: engine('jaw_1', value),
    rng=[0, 30],
    value=0,
    title="Jaw 1",
    pointa=(0.01, 0.95),
    pointb=(0.31, 0.95),
    style='modern'
)
p.add_slider_widget(
    callback=lambda value: engine('jaw_2', value),
    rng=[-45, 45],
    value=0,
    title="Jaw 2",
    pointa=(0.01, 0.85),
    pointb=(0.31, 0.85),
    style='modern'
)
p.add_slider_widget(
    callback=lambda value: engine('jaw_3', value),
    rng=[-45, 45],
    value=0,
    title="Jaw 3",
    pointa=(0.01, 0.75),
    pointb=(0.31, 0.75),
    style='modern'
)

# adding sliders for expressions and shapes
for i in range(1, 8):
    parameter = 'expression_{}'.format(i)
    title = 'Expression {}'.format(i)
    pointa = (0.01, 0.65 - 0.1*(i-1))
    pointb = (0.31, 0.65 - 0.1*(i-1))
    p.add_slider_widget(
        callback=lambda value, param = parameter: updateParameter(param, value),
        rng=[-2, 2],
        value=0,
        title=title,
        pointa=pointa,
        pointb=pointb,
        style='modern'
    )

for i in range(1, 11):
    parameter = 'shape_{}'.format(i)
    title = 'Shape {}'.format(i)
    pointa = (0.69, 0.95 - 0.1*(i-1))
    pointb = (0.99, 0.95 - 0.1*(i-1))
    p.add_slider_widget(
        callback=lambda value, param = parameter: updateParameter(param, value),
        rng=[-2, 2],
        value=0,
        title=title,
        pointa=pointa,
        pointb=pointb,
        style='modern'
    )


p.show()
        