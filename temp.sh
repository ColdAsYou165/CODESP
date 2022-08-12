#net_name_array=(resnet18 resnet32 resnet56 lenet)
net_name_array=(resnet18 resnet32 resnet56)
#net_name_array=(resnet18)
#resnet_dataset_array=(cifar10 cifar100 svhn)
resnet_dataset_array=(svhn)
lenet_dataset_array=(mnist fmnist)
noise_type_array=("random_pixel" "gaussian_noise")
attack_method_array=("fgsm" "i_fgsm")
#attack_method="mi_fgsm"
targeted=None
loss_type=None
class_nums_100="cifar100"
date="6_5"
folder_name="svhn_new"
python_file="May_test_temp"
batch_size=100
num=10000
transferability="False"
for attack_method in ${attack_method_array[@]}
do
  for noise_type in ${noise_type_array[@]}
  do
    for net_name in ${net_name_array[@]}
    do
    if [ "${net_name}" = "lenet" ]
    then
      for dataset in ${lenet_dataset_array[@]}
      do
      weight="./checkpoint/${net_name}_${dataset}.pth"
      folder_dir="${date}/${folder_name}/${attack_method}/num=${num}/${noise_type}"
      class_nums=10
      echo python ./${python_file}.py --net_name ${net_name} --batch_size ${batch_size} --weights ${weight} --train_dataset ${dataset} --class_nums ${class_nums} --noise_type ${noise_type} --attack_method ${attack_method} --num ${num} --loss_type ${loss_type} --gpu 1 --targeted ${targeted} --folder_name ${folder_dir}
      done
    else
      for dataset in ${resnet_dataset_array[@]}
      do
      if [ "${dataset}" = "svhn" ];
      then
        weight="./checkpoint/${net_name}_${dataset}_new.pth"
      else
        weight="./checkpoint/${net_name}_${dataset}.pth"
      fi
      folder_dir="${date}/${folder_name}/${attack_method}/num=${num}/${noise_type}"
      if [ "${dataset}" = "${class_nums_100}" ];
      then
        class_nums=100
        echo "class_nums=100"
      else
        class_nums=10
        echo "class_nums=10"
      fi
      echo python ./${python_file}.py --net_name ${net_name} --batch_size ${batch_size} --weights ${weight} --train_dataset ${dataset} --class_nums ${class_nums} --noise_type ${noise_type} --attack_method ${attack_method} --num ${num} --loss_type ${loss_type} --gpu 1 --targeted ${targeted} --folder_name ${folder_dir}
      done
    fi
      #python ./buffer.py --net_name ${net_name} --batch_size 100 --weights "weight" --train_dataset cifar10 --class_nums 10 --noise_type "gaussian noise" --attack_method "PPBA" --num 10000 --loss_type margin_loss --gpu 1 --targeted "targeted_top1" --folder_name "fn"
    done
  done
done


