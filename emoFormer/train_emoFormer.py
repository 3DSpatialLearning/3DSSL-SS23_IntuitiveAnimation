from dataLoader_params import load_data
from emoFormer import EmoFormer
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import os
import argparse
from torch.utils.tensorboard import SummaryWriter

def train(args, model, train_loader, val_loader, optimizer, criterion):
    writer = SummaryWriter(f"runs/{args.run_name}")
    if not os.path.exists(os.path.join(args.save_path, args.run_name)):
        os.makedirs(os.path.join(args.save_path, args.run_name))

    iteration = 0
    
    cross = False
    if cross == True:
        for e in range(args.max_epochs):
            model.train()
            loss_org_log = []
            loss_con_log = []
            loss_emo_log = []
            pbar = tqdm(enumerate(train_loader),total=len(train_loader))
            
            for i, (name_cross, audio_cross, params_cross) in pbar:
                iteration += 1
                print(type(audio_cross), type(params_cross))
                for audio, params in zip(audio_cross, params_cross):
                    audio, params = audio.to(args.device), params.to(args.device)
                    
                    # aa, ba, ab - org, con, emo
                    
                ca_aa, ea_aa = model.forward_en(audio[0], params[0], criterion)
                cb_ba, ea_ba = model.forward_en(audio[1], params[1], criterion)
                ca_ab, eb_ab = model.forward_en(audio[2], params[2], criterion)
                    
                loss_org = model.forward_de(_, params[0], criterion, ca_aa, ea_aa)
                loss_con = model.forward_de(_, params[1], criterion, cb_ba, ea_aa)
                loss_emo = model.forward_de(_, params[2], criterion, ca_aa, eb_ab)
                    
                loss_sum = sum(loss_org.values()) + sum(loss_con.values()) + sum(loss_emo.values())
                loss_sum.backward()
                    
                loss_org_log.append(loss_org.item())
                loss_con_log.append(loss_con.item())
                loss_emo_log.append(loss_emo.item())
                    
                if i % args.gradient_accumulation_steps==0:
                    optimizer.step()
                    optimizer.zero_grad()
                        
                pbar.set_description(f"(Epoch {e+1}, iteration {iteration}), con_loss:{np.mean(loss_con_log):.7f}, emo_loss:{np.mean(loss_emo_log):.7f}, org_loss:{np.mean(loss_emo_log):.7f}")
                
            writer.add_scalar("con_loss", np.mean(loss_con_log), e+1)
            writer.add_scalar("emo_loss", np.mean(loss_emo_log), e+1)
            writer.add_scalar("org_loss", np.mean(loss_org_log), e+1)
                
                # validation
            val_loss_log = []
            val_emo_loss_log = []
            
            model.eval()
            for name, audio, params in val_loader:
                audio, params = audio, params.to(args.device)
                loss_dict = model(audio, params, criterion, teacher_forcing = False)
                
                loss = loss_dict["loss"]
                loss_emo = loss_dict["loss_emo"]
                
                val_loss_log.append(loss.item())
                val_emo_loss_log.append(loss_emo.item())
            
            val_loss = np.mean(val_loss_log)
            val_emo_loss = np.mean(val_emo_loss_log)

            if (e > 0 and (e + 1) % args.save_epoch == 0) or e == args.max_epochs - 1:
                torch.save(model.state_dict(), os.path.join(args.save_path, args.run_name, f"{e+1}_model.pth"))

            print(f"Epoch: {e+1}, val_loss: {val_loss:.7f}")
            print(f"Epoch: {e+1}, val_emo_loss: {val_emo_loss:.7f}")

            writer.add_scalar("validation_loss", val_loss, e+1)
            writer.add_scalar("validation_emo_loss", val_emo_loss, e+1)
                    
    else:
        for e in range(args.max_epochs):
            model.train()
            loss_log = []
            loss_emo_log = []
            pbar = tqdm(enumerate(train_loader),total=len(train_loader))
            # train
            for i, (name, audio, params) in pbar:
                iteration += 1
                audio, params = audio, params.to(args.device)
                loss_dict = model(audio, params, criterion, teacher_forcing = False)
                loss_sum = sum(loss_dict.values())
                loss = loss_dict["loss"]
                loss_emo = loss_dict["loss_emo"]
                
                loss_sum.backward()
                loss_log.append(loss.item())
                loss_emo_log.append(loss_emo.item())
                
                if i % args.gradient_accumulation_steps==0:
                    optimizer.step()
                    optimizer.zero_grad()

                pbar.set_description(f"(Epoch {e+1}, iteration {iteration}) Train_emo_loss:{np.mean(loss_emo_log):.7f}, Train_loss:{np.mean(loss_log):.7f}")

            writer.add_scalar("train_loss", np.mean(loss_log), e+1)
            writer.add_scalar("train_emo_loss", np.mean(loss_emo_log), e+1)

            # validation
            val_loss_log = []
            val_emo_loss_log = []
            
            model.eval()
            for name, audio, params in val_loader:
                audio, params = audio, params.to(args.device)
                loss_dict = model(audio, params, criterion, teacher_forcing = False)
                
                loss = loss_dict["loss"]
                loss_emo = loss_dict["loss_emo"]
                
                val_loss_log.append(loss.item())
                val_emo_loss_log.append(loss_emo.item())
            
            val_loss = np.mean(val_loss_log)
            val_emo_loss = np.mean(val_emo_loss_log)

            if (e > 0 and (e + 1) % args.save_epoch == 0) or e == args.max_epochs - 1:
                torch.save(model.state_dict(), os.path.join(args.save_path, args.run_name, f"{e+1}_model.pth"))

            print(f"Epoch: {e+1}, val_loss: {val_loss:.7f}")
            print(f"Epoch: {e+1}, val_emo_loss: {val_emo_loss:.7f}")

            writer.add_scalar("validation_loss", val_loss, e+1)
            writer.add_scalar("validation_emo_loss", val_emo_loss, e+1)
        
    writer.close()
    return model

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

    # Train arguments
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epochs", type=int, default=100, help='number of epochs')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--run_name", type=str, default="overfit4", help="name of this run")
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--npz_path", type=str, default="/mnt/hdd/datasets/RAVDESS/npzs", help='path to the npz files')
    parser.add_argument("--wav_path", type=str, default="/mnt/hdd/datasets/RAVDESS/wavs", help='path to the wav files')
    parser.add_argument("--num_data", type=int, default=1000, help="numbers of data used for dataloader")
    parser.add_argument("--device", type=str, default="cuda", help="cuda for gpu, cpu for cpu")
    parser.add_argument("--save_epoch", type=int, default=5, help="save the model after every n epochs")
    parser.add_argument("--emo", type=bool, default=False, help="test see if the model capture emotion")
    
    args = parser.parse_args()

    dataloader = load_data(args.npz_path, args.wav_path, args.num_data)
    train_loader = dataloader["train"]
    val_loader = dataloader["val"]
    test_loader = dataloader["test"]

    model = EmoFormer(args)
    model.to(args.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
    criterion = nn.MSELoss()

    model = train(args, model, train_loader, val_loader, optimizer, criterion)

    

if __name__=="__main__":
    
    main()
        
