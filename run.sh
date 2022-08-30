#Baseline
#python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device tpu-8 --experiment resnet-50-start_from_scratch --batch-size 512 --epochs 200

#python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device cpu --experiment resnet-50-test --batch-size 512 --epochs 200 --pretrained

#python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device tpu-8 --experiment resnet-50-start_from_pretrained_50 --batch-size 512 --pretrained --start-epoch 150 --epochs 200

#Li
#python3 train.py /mnt/ssd/imagenet/ --device tpu-8 --experiment resnet-50-from_scratch --batch-size 512  --scale 0.8 1.0 -Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_supervised_imagenet.yaml -target_entropy 3.0

python3 train.py /mnt/ssd/imagenet/ --device cpu --experiment resnet-50-test --batch-size 512  --scale 0.8 1.0 -Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_supervised_imagenet.yaml -target_entropy 3.0 -entropy_parameter 0.1