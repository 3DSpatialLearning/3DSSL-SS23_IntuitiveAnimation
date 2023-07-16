import numpy as np
import os
import glob

def stack_and_save_npz(base_dir, output_dir):
    filename = os.path.basename(os.path.dirname(base_dir))
    directories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    directories.sort()
    exp_list = []
    shape_list = []
    pose_list = []
    for directory in directories:
        if directory.endswith("000"):
            
            exp_path = os.path.join(base_dir, directory, "exp.npy")
            shape_path = os.path.join(base_dir, directory, "shape.npy")
            pose_path = os.path.join(base_dir, directory, "pose.npy")
            print(f"loading: {exp_path}")
            exp = np.load(exp_path)
            shape = np.load(shape_path)
            pose = np.load(pose_path)
            exp_list.append(exp)
            shape_list.append(shape)
            pose_list.append(pose)
    if len(exp_list) != 0:
        print(f"stack and saving: {os.path.join(output_dir, filename)}.npz")
        exp_out = np.stack(exp_list, axis=0)
        shape_out = np.stack(shape_list, axis=0)
        pose_out = np.stack(pose_list, axis=0)

        np.savez(f"{os.path.join(output_dir, filename)}.npz", shape = shape_out, expr = exp_out, pose = pose_out)

for i in range(1, 19):
    base = f"/mnt/hdd/datasets/RAVDESS/Video_Speech_Actor_{i:02d}/Actor_{i:02d}/"
    if os.path.exists(base):
        sequences = os.listdir(base)
        for sequence in sequences:
            stack_and_save_npz(os.path.join(base, sequence, "EMOCA_v2_lr_mse_20"), "/mnt/hdd/datasets/RAVDESS/npzs")