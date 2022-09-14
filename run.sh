#Baseline
#python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device tpu-8 --experiment resnet-50-start_from_scratch --batch-size 512 --epochs 200

#python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device tpu-8 --experiment resnet-50-test --batch-size 512 --epochs 200 --start-epoch 170 --pretrained

#python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device tpu-8 --experiment resnet-50-test --batch-size 512 --epochs 1 -eval_only  --pretrained


#Apply Li
python3 train.py /mnt/ssd/imagenet/ --device tpu-8 --experiment resnet-test --batch-size 256  --scale 0.8 1.0  -eval_only --pretrained --epochs 1 --cooldown-epochs 0 --start-epoch 0 --no-aug -Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_supervised_imagenet.yaml

#-Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_supervised_imagenet.yaml

#Only train Li
#python3 train.py /mnt/ssd/imagenet/ --device tpu-8 --experiment resnet-50-Li-train_on_eval --batch-size 256 -Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_supervised_imagenet.yaml -target_entropy 1.3 -entropy_parameter 3.0  --pretrained --epochs 200 -Li_lr 0.01 -Li_loss crossentropy -train_sample_amount 256 -train_only --no-aug --warmup-epochs 0 --lr 0.0 --lr-noise-pct 0.0 --lr-noise-std 0.0  -save_every 3 --start-epoch 0 -train_Li_only 

#Train model and Li
#python3 train.py /mnt/ssd/imagenet/ --device tpu-8 --experiment resnet-50-Li-_te_3.0_new --batch-size 256 -Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_supervised_imagenet.yaml -target_entropy 3.0 -entropy_parameter 0.03  --pretrained --epochs 200 -Li_lr 0.1 -Li_loss crossentropy -batch_amount 0 -train_only --scale 0.8 1.0 --warmup-epochs 0 -save_every 3 --start-epoch 170