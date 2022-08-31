#!/bin/bash
#gpus_list=("0,1" "2,3" "4,5")
#lr_list=(0.00001)
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
#1e-4 1e-3 1e-2 1e-1 1 1e1 1e2 1e3 1e4
#python train_wgan.py --gpus 0 --lr_dis 1e4 --w_loss_weight 1e-4 &
#sleep 60
#python train_wgan.py --gpus 1 --lr_dis 1e4 --w_loss_weight 1e-3 &
#sleep 60
#python train_wgan.py --gpus 2 --lr_dis 1e4 --w_loss_weight 1e-2 &
#sleep 60
#python train_wgan.py --gpus 3 --lr_dis 1e4 --w_loss_weight 1e-1 &
#sleep 60
#python train_wgan.py --gpus 4 --lr_dis 1e4 --w_loss_weight 1 &
#sleep 60
#python train_wgan.py --gpus 5 --lr_dis 1e4 --w_loss_weight 1e1 &
##
#wait
#python train_wgan.py --gpus 0 --lr_dis 1e4 --w_loss_weight 1e2 &
#sleep 60
#python train_wgan.py --gpus 1 --lr_dis 1e4 --w_loss_weight 1e3 &
#sleep 60
#python train_wgan.py --gpus 2 --lr_dis 1e4 --w_loss_weight 1e4 &
#sleep 60
##从这里开始是试一下不用sigmoid
#python train_wgan.py --gpus 3 --lr_dis 1e4 --set_sigmoid False &
#sleep 60
#python train_wgan.py --gpus 4 --le_g 1e-5 --lr_dis 1e4 --set_sigmoid False &
#sleep 60
##微调lr_dis的参数
#python train_wgan.py --gpus 5 --lr_dis 1e4 &
#wait
#python train_wgan.py --gpus 0 --lr_dis 2e4 &
#sleep 60
#python train_wgan.py --gpus 1 --lr_dis 4e4 &
#sleep 60
##换一个更简单的discriminator
#python train_ae_with_3loss_chamfer_w_cross.py --gpus 0 --cross_loss_weight 1e-1 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross.py --gpus 1 --cross_loss_weight 1e-2 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross.py --gpus 2 --cross_loss_weight 1e-3 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross.py --gpus 3 --cross_loss_weight 1e-4 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross.py --gpus 4 --cross_loss_weight 1e-5 &
#sleep 60
#python train_ae_with_3loss_chamfer_w_cross.py --gpus 5 --cross_loss_weight 1 &
#sleep 60
#python train_ae.py --gpus 0 --lr 0.1 &
#sleep 60
#python train_ae.py --gpus 1 --lr 0.01 &
#sleep 60
#python train_ae.py --gpus 2 --lr 0.001 &
#sleep 60
#python train_ae.py --gpus 3 --lr 0.0001 &
#sleep 60
#python train_ae.py --gpus 4 --lr 0.00001 &
#sleep 60
#python train_ae.py --gpus 5 --lr 0.000001 &
#sleep 60
#wait
#python train_ae.py --gpus 0 --lr 0.001 --loss_virtual_weight 2 &
#sleep 60
#python train_ae.py --gpus 1 --lr 0.001 --loss_virtual_weight 4 &
#sleep 60
#python train_ae.py --gpus 2 --lr 0.006 --loss_virtual_weight 2 &
#sleep 60
#python train_ae.py --gpus 3 --lr 0.006 --loss_virtual_weight 4 &
#sleep 60
#python train_ae.py --gpus 4 --lr 0.0001 --loss_virtual_weight 2 &
#sleep 60
#python train_ae.py --gpus 5 --lr 0.0001 --loss_virtual_weight 4 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 0 --loss_virtual_weight 0.0009 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 1 --loss_virtual_weight 0.0008 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 2 --loss_virtual_weight 0.0007 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 3 --loss_virtual_weight 0.0006 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 4 --loss_virtual_weight 0.0005 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 5 --loss_virtual_weight 0.0004 &
#sleep 60
#wait
#python train_resnet_with_virtual.py --gpus 0 --loss_virtual_weight 0.0003 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 1 --loss_virtual_weight 0.0002 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 2 --loss_virtual_weight 0.0001 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 3 --loss_virtual_weight 0 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 4 --loss_virtual_weight 0 &
#sleep 60
#python train_resnet_with_virtual.py --gpus 5 --loss_virtual_weight 1e-5 &
#sleep 60
#python train_ae_chamfer_and_crossgan.py --gpus 4 --w_loss_weight 1e-4 &
#sleep 60
#python train_ae_chamfer_and_crossgan.py --gpus 5 --w_loss_weight 1e-5 &
#wait
#python train_ae_chamfer_and_crossgan.py --gpus 4 --w_loss_weight 1e-6 &
#sleep 60
#python train_ae_chamfer_and_crossgan.py --gpus 5 --w_loss_weight 1e-5 --lr_g 2e-5 --lr_d 1e-5 &
#wait
#python train_ae_chamfer_and_crossgan.py --gpus 4 --w_loss_weight 1e-5 --lr_g 2e-5 --lr_d 0 --lr_dis 1e3 &
#sleep 60
#python train_ae_chamfer_and_crossgan.py --gpus 5 --w_loss_weight 1e-5 --lr_g 2e-5 --lr_d 0 --lr_dis 1e4 &
#wait
#python train_ae_chamfer_and_crossgan.py --gpus 4 --w_loss_weight 1e-5 --lr_g 2e-5 --lr_d 0 --lr_dis 1e5 &
#sleep 60
#python train_ae_chamfer_and_crossgan.py --gpus 5 --w_loss_weight 1e-5 --lr_g 0.0006 --lr_d 0 --lr_dis 1e3 &
#wait
#python train_ae_chamfer_and_crossgan.py --gpus 4 --w_loss_weight 1e-5 --lr_g 0.0006 --lr_d 0 --lr_dis 1e3 &
#sleep 60
#python train_ae_chamfer_and_crossgan.py --gpus 5 --w_loss_weight 1e-5 --lr_g 0.0006 --lr_d 0 --lr_dis 1e4 &
#wait
#python train_ae_chamfer_and_crossgan.py --gpus 4 --w_loss_weight 1e-4 --real_cross_d_weight 0.1 &
#sleep 60
#python train_ae_chamfer_and_crossgan.py --gpus 5 --w_loss_weight 1e-5 --real_cross_d_weight 0.001 &
#wait
python train_resnet_with_virtual.py --gpus 0 --loss_virtual_weight 1 --crossweight 3 &
sleep 60
python train_resnet_with_virtual.py --gpus 1 --loss_virtual_weight 1 --crossweight 4 &
sleep 60
python train_resnet_with_virtual.py --gpus 2 --loss_virtual_weight 1 --crossweight 5 &
sleep 60
python train_resnet_with_virtual.py --gpus 3 --loss_virtual_weight 0.1 --crossweight 3 &
sleep 60
python train_resnet_with_virtual.py --gpus 4 --loss_virtual_weight 0.1 --crossweight 4 &
sleep 60
python train_resnet_with_virtual.py --gpus 5 --loss_virtual_weight 0.1 --crossweight 5 &
sleep 60
wait
python train_resnet_with_virtual.py --gpus 0 --loss_virtual_weight 5e-3 --crossweight 3 &
sleep 60
python train_resnet_with_virtual.py --gpus 1 --loss_virtual_weight 5e-3 --crossweight 4 &
sleep 60
python train_resnet_with_virtual.py --gpus 2 --loss_virtual_weight 5e-3 --crossweight 5 &
sleep 60
python train_resnet_with_virtual.py --gpus 3 --loss_virtual_weight 5e-4 --crossweight 3 &
sleep 60
python train_resnet_with_virtual.py --gpus 4 --loss_virtual_weight 5e-4 --crossweight 4 &
sleep 60
python train_resnet_with_virtual.py --gpus 5 --loss_virtual_weight 5e-4 --crossweight 5 &
sleep 60
wait
python train_resnet_with_virtual.py --gpus 0 --loss_virtual_weight 1e-3 --crossweight 3 &
sleep 60
python train_resnet_with_virtual.py --gpus 1 --loss_virtual_weight 1e-3 --crossweight 4 &
sleep 60
python train_resnet_with_virtual.py --gpus 2 --loss_virtual_weight 1e-3 --crossweight 5 &
sleep 60
python train_resnet_with_virtual.py --gpus 3 --loss_virtual_weight 1e-4 --crossweight 3 &
sleep 60
python train_resnet_with_virtual.py --gpus 4 --loss_virtual_weight 1e-4 --crossweight 4 &
sleep 60
python train_resnet_with_virtual.py --gpus 5 --loss_virtual_weight 1e-4 --crossweight 5 &
sleep 60
