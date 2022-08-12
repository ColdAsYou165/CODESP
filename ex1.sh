# gpus=("1,2" "3,4" "5,6")
## python train_ae.py --gpus=${gpus} --lr=${lr}
#for i in ${gpus[@]}
#do
#  if [ "${i}" = "3,4" ]
#  then
#    echo ${i}
#  fi
#done
#w_loss_weight
w=0.00001
let "b=w+1"
echo ${w+1}
echo $b