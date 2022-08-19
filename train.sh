#!/bin/bash
gpus_list=("0,1" "2,3" "4,5")
lr_list=(0.00001)
#lr_dis_list=(0.00001 0.0001)
#先只修改w_loss_weight_list吧
#w_loss_weight_list=(0.001 0.0001)
#python train_ae.py --gpus 0,1 --lr 1e-3 --virtual_scale 4 &
#sleep 60
#python train_ae.py --gpus 2,5 --lr 1e-3 --virtual_scale 6 &
#wait
#python train_ae.py --gpus 0,1 --lr 1e-3 --virtual_scale 8 &
#sleep 60
#python train_ae.py --gpus 2,5 --lr 1e-4 --virtual_scale 4 &
##python train_ae.py --gpus 0,1 --lr 1e-5 --mse_loss_weight 2

#python train_ae.py --gpus 0,1 --lr 1e-5 --lr_d 1e-5 --wgan_optim Adam --w_loss_weight 1e-5 --d_loss_real_weight 3 &

python train_ae.py --gpus 0,1 --w_loss_weight 1e-5 &
sleep 60
python train_ae.py --gpus 2,3 --w_loss_weight 1e-4 &
sleep 60
python train_ae.py --gpus 4,5 --w_loss_weight 1e-3 &

