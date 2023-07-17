import numpy as np
import pickle
import os
import glob
import argparse

def changeTemplate(faceFormer_pred_path, save_path):
    with open("/home/haifanzhang/3DSSL-SS23_IntuitiveAnimation/model/generic_model.pkl", 'rb') as f:
        new_template = pickle.load(f, encoding='latin1')
    
    filename = os.path.basename(faceFormer_pred_path)
    ff_pred = np.load(faceFormer_pred_path)
    num_frames = ff_pred.shape[0]
    new_template_vertices = new_template["v_template"]
    new_template_vertices = new_template_vertices.reshape(1, 15069)
    all_new_template_vertices = np.tile(new_template_vertices, (num_frames,1))

    with open("/home/haifanzhang/FaceFormer/vocaset/templates.pkl", 'rb') as f:
        old_template = pickle.load(f, encoding='latin1')

    old_template_vertices = old_template["FaceTalk_170809_00138_TA"]
    old_template_vertices = old_template_vertices.reshape(1, 15069)

    all_old_template_vertices = np.tile(old_template_vertices, (num_frames,1))

    ff_pred = np.load(faceFormer_pred_path)

    new_vertices = ff_pred - all_old_template_vertices + all_new_template_vertices

    np.save(os.path.join(save_path, filename), new_vertices)

def main():
    parser = argparse.ArgumentParser(description='Change the FaceFormer prediction with FLAME generic model')
    parser.add_argument("--pred_path", type=str, help="Path of FaceFormer prediction")
    parser.add_argument("--save_path", type=str, help="Path for saving mesh with changed template")

    args = parser.parse_args()

    faceformer_pred_path = args.pred_path
    save_path = args.save_path

    preds = glob.glob(faceformer_pred_path)

    for pred in preds:
        changeTemplate(pred, save_path)


if __name__ == "__main__":
    main()