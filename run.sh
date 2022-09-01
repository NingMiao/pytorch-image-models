#Baseline
#python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device tpu-8 --experiment resnet-50-start_from_scratch --batch-size 512 --epochs 200

#python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device tpu-8 --experiment resnet-50-test --batch-size 512 --epochs 200 --start-epoch 170 --pretrained

#python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device tpu-8 --experiment resnet-50-test --batch-size 512 --epochs 1 -eval_only  --pretrained

#Li
#python3 train.py /mnt/ssd/imagenet/ --device tpu-8 --experiment resnet-50-from_scratch --batch-size 512  --scale 0.8 1.0 -Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_supervised_imagenet.yaml -target_entropy 3.0

#python3 train.py /mnt/ssd/imagenet/ --device tpu-8 --experiment resnet-50-test --batch-size 512  --scale 0.8 1.0 -Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_supervised_imagenet.yaml -target_entropy 3.0 -entropy_parameter 0.3  --pretrained --epochs 200 --start-epoch 170

python3 train.py /mnt/ssd/imagenet/ --device tpu-8 --experiment resnet-50-Li-30-ep0.1_lr0.001-te3.0 --batch-size 512  --scale 0.8 1.0 -Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_supervised_imagenet.yaml -target_entropy 3.0 -entropy_parameter 0.1  --pretrained --epochs 200 --start-epoch 170 -Li_lr 0.001

#Apply Li
#python3 train.py /mnt/ssd/imagenet/ --device tpu-8 --experiment resnet-test --batch-size 64  --scale 0.8 1.0 -Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_supervised_imagenet.yaml -eval_only --resume output/train/resnet-50-Li-30-new/model182.ckpt -resume_Li output/train/resnet-50-Li-30-new/Li182.ckpt --epochs 1 --cooldown-epochs 0 --start-epoch 0