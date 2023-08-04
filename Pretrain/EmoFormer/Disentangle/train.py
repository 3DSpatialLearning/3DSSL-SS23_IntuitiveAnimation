import argparse
import time
import tensorboardX
import os
import numpy as np

from tqdm import tqdm

from model import DisentangleNet
from dataset import DisentangleDataset

import torch
from torch.utils.data import DataLoader
from torch.nn import init



def initialize_weights( net, init_type='kaiming', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def my_collate(batch):
    ## data is a list
    sp_11 = [ item['sp_11'] for item in batch]
    sp_12 = [ item['sp_12'] for item in batch]
    sp_21 = [ item['sp_21'] for item in batch]
    sp_22 = [ item['sp_11'] for item in batch]
    emo_1 = [ item['emo_1'] for item in batch]
    emo_2 = [ item['emo_2'] for item in batch]

    return [sp_11, sp_12, sp_21, sp_22, emo_1, emo_2]

def main():

    parser = argparse.ArgumentParser(description='disentangle')

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda1", type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=4e-4)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--num_thread", type=int, default=0)
    parser.add_argument("--cuda", type=bool, default=True)   

    parser.add_argument("--model_dir", type=str, default="./train/model/")
    parser.add_argument("--log_dir", type=str, default="./log/")
    parser.add_argument("--datatset_dir", type=str, default="/home/yuxinguo/data/RAVDESS/Audio_Speech_Actors_01-24/")
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--run_name", type=str, default="dis_weight", help="name of this run")
    parser.add_argument("--emo_only", type=bool, default=False, help="only classification")
    parser.add_argument("--pretrain", type=bool, default=False, help="use pretrain")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    model = DisentangleNet(args)
    print("model parameters: ", count_parameters(model))
    if args.cuda:
        model = model.cuda()

    if args.pretrain == True:
        # load from pretrain
        state_dict = torch.load('/home/yuxinguo/EmoFormer/SER/model_20230719_200139_15.pt', map_location=torch.device('cpu'))
    
    # load wav2vec2
        for name, param in state_dict.items():
            name_my = "audio_encoder." + name
            if name_my in model.state_dict().keys():
                name_my = "model." + name_my + ".data"
                vars()[name_my] = state_dict[name]
        
    
    # load emo
        model.emo_encoder.projector.weight.data = state_dict['projector.weight']
        model.emo_encoder.projector.bias.data = state_dict['projector.bias']
        model.emo_encoder.classifier.weight.data = state_dict['classifier.weight']
        model.emo_encoder.classifier.bias.data = state_dict['classifier.bias']
        
    model.cuda()
    # load data
    print('loading train data & test data')

    train_set = DisentangleDataset(args.datatset_dir, 'train')
    val_set = DisentangleDataset(args.datatset_dir, 'test')

    # train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_thread, shuffle=True, drop_last=True, collate_fn=my_collate )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_thread, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=args.num_thread, shuffle=True, drop_last=True)
    
    print('end loading')

    # training
    writer = tensorboardX.SummaryWriter(comment='dis')

    
    iteration = 0
    for epoch in range(0, args.max_epochs):
        model.train()
        loss_con_log = []
        loss_emo_log = []
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))

        for i, data in pbar:
            iteration += 1
            
            losses, acces = model.train_func(data)
            loss_con = losses["con_loss"]
            if args.emo_only == True:
                loss_con *= 0
            
            loss_emo = losses["cla1_loss"] + losses["cla2_loss"]
            loss_emo *= 1
            
            loss_con_log.append(loss_con.item())
            loss_emo_log.append(loss_emo.item())
            
            pbar.set_description(f"(Epoch {epoch+1}, iteration {iteration}) con_loss:{np.mean(loss_con_log):.7f}, cla_loss:{np.mean(loss_emo_log):.7f}")

        writer.add_scalar("train_con_loss", np.mean(loss_con_log), epoch+1)
        writer.add_scalar("train_emo_loss", np.mean(loss_emo_log), epoch+1)
        
        val_con_loss_log = []
        val_emo_loss_log = []
        
        model.eval()
        with torch.no_grad():
            
            for i, data in enumerate(val_loader):

                losses, acces = model.val_func(data)
                loss_con = losses["con_loss"]
                loss_emo = losses["cla1_loss"] + losses["cla2_loss"]

                val_con_loss_log.append(loss_con.item())
                val_emo_loss_log.append(loss_emo.item())
            
            val_con_loss = np.mean(val_con_loss_log)
            val_emo_loss = np.mean(val_emo_loss_log)
            
            if (epoch > 0 and (epoch + 1) % 5 == 0):
                torch.save(model.state_dict(), os.path.join(args.save_path, args.run_name, f"{epoch+1}_model.pth"))
            
            print(f"Epoch: {epoch+1}, val_con_loss: {val_con_loss:.7f}")
            print(f"Epoch: {epoch+1}, val_emo_loss: {val_emo_loss:.7f}")
            
            writer.add_scalar("val_con_loss", val_con_loss, epoch+1)
            writer.add_scalar("val_emo_loss", val_emo_loss, epoch+1)

if __name__ == "__main__":
    main()