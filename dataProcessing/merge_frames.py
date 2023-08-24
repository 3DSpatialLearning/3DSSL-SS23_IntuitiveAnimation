import numpy as np
import os
import glob

dataset_dir = "/mnt/hdd/datasets/RAVDESS/"
npz_dir = "/mnt/hdd/datasets/RAVDESS/npzs"

# for filename in glob.glob(r"/mnt/hdd/datasets/RAVDESS/*/*"):
#     print(os.path.basename(filename))

frames = [filename for filename in glob.glob("/mnt/hdd/datasets/RAVDESS/*/*/*/*/*/")]
frames.sort()

for frame in frames:
    if frame.split("/")[-2].startswith("p"):
        frames.remove(frame)

for i, frame in enumerate(frames):
    try:
        if i == 0:
            exp_list = []
            shape_list = []
            pose_list = []
            exp_path = os.path.join(frame, "exp.npy")
            shape_path = os.path.join(frame, "shape.npy")
            pose_path = os.path.join(frame, "pose.npy")
            print(f"loading {exp_path}")
            exp = np.load(exp_path)
            shape = np.load(shape_path)
            pose = np.load(pose_path)
            exp_list.append(exp)
            shape_list.append(shape)
            pose_list.append(pose)
        elif frame.split("/")[-4] == frames[i-1].split("/")[-4]:
            filename = frame.split("/")[-4]
            exp_path = os.path.join(frame, "exp.npy")
            shape_path = os.path.join(frame, "shape.npy")
            pose_path = os.path.join(frame, "pose.npy")
            print(f"loading {exp_path}")
            if os.path.exists(exp_path):
                exp = np.load(exp_path)
                shape = np.load(shape_path)
                pose = np.load(pose_path)
                exp_list.append(exp)
                shape_list.append(shape)
                pose_list.append(pose)
        else:
            last_filename = frames[i-1].split("/")[-4]
            print(f"saving {os.path.join(npz_dir, last_filename)}.npz")
            exp_out = np.stack(exp_list, axis=0)
            shape_out = np.stack(shape_list, axis=0)
            pose_out = np.stack(pose_list, axis=0)
            np.savez(f"{os.path.join(npz_dir, last_filename)}.npz", shape = shape_out, expr = exp_out, pose = pose_out)
            print("clear")
            exp_list = []
            shape_list = []
            pose_list = []
            exp_path = os.path.join(frame, "exp.npy")
            shape_path = os.path.join(frame, "shape.npy")
            pose_path = os.path.join(frame, "pose.npy")
            print(f"loading {exp_path}")
            if os.path.exists(exp_path):
                exp = np.load(exp_path)
                shape = np.load(shape_path)
                pose = np.load(pose_path)
                exp_list.append(exp)
                shape_list.append(shape)
                pose_list.append(pose)
    except:
        print(filename)
        
    

    # current_filename = frame.split("/")[-4]
    # if current_filename != last_filename:
    #     exp_out = np.stack(exp_list, axis=0)
    #     shape_out = np.stack(shape_list, axis=0)
    #     pose_out = np.stack(pose_list, axis=0)
    #     transl = np.repeat(np.zeros((3,)), exp_out.shape[0]).reshape(exp_out.shape[0], 3)
    #     print(f"saving {last_filename}.npz")
    #     print("clear")
    #     exp_list = []
    #     shape_list = []
    #     pose_list = []
    #     tex_list = []
    #     exp_path = os.path.join(frame, "exp.npy")
    #     shape_path = os.path.join(frame, "shape.npy")
    #     pose_path = os.path.join(frame, "pose.npy")
    #     print(f"adding {exp_path}")
    # else:
    #     exp_path = os.path.join(frame, "exp.npy")
    #     shape_path = os.path.join(frame, "shape.npy")
    #     pose_path = os.path.join(frame, "pose.npy")
    #     print(f"adding {exp_path}")
    #     # exp = np.load(exp_path)
    #     # shape = np.load(shape_path)
    #     # pose = np.load(pose_path)
    #     # exp_list.append(exp)
    #     # shape_list.append(shape)
    #     # pose_list.append(pose)

    # last_filename = frame.split("/")[-4]

    # print(actor)


# for actor in actors:
#     actor_name = os.path.basename(actor)
#     sentences = [frame for frame in glob.glob(os.path.join(actor, "*/"))]
#     sentences.sort()
#     for sentence 
#     print(frames)
# base_dir = "/mnt/hdd/datasets/gigaMove/emoca_pred/024_SEN-01-cramp_small_danger_cam_222200037/EMOCA_v2_lr_mse_20"

# directories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
# directories.sort()
# exp_list = []
# shape_list = []
# pose_list = []
# tex_list = []
# for directory in directories:
#     exp_path = os.path.join(base_dir, directory, "exp.npy")
#     shape_path = os.path.join(base_dir, directory, "shape.npy")
#     pose_path = os.path.join(base_dir, directory, "pose.npy")
#     exp = np.load(exp_path)
#     shape = np.load(shape_path)
#     pose = np.load(pose_path)
#     exp_list.append(exp)
#     shape_list.append(shape)
#     pose_list.append(pose)

# exp_out = np.stack(exp_list, axis=0)
# shape_out = np.stack(shape_list, axis=0)
# pose_out = np.stack(pose_list, axis=0)
# transl = np.repeat(np.zeros((3,)), exp_out.shape[0]).reshape(exp_out.shape[0], 3)

# print(exp_out.shape)
# print(shape_out.shape)
# print(pose_out.shape)
# print(transl.shape)