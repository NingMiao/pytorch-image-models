#Apply Li
#python3 train.py /mnt/ssd/imagenet/ --device tpu-8 --experiment resnet-test --batch-size 256  --scale 0.8 1.0  -eval_only --pretrained --epochs 1 --cooldown-epochs 0 --start-epoch 0 --no-aug -Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_supervised_imagenet.yaml

#-Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_supervised_imagenet.yaml

#Only train Li
python3 train_Li_only.py /mnt/ssd/imagenet/ --device cpu --experiment resnet-50-Li-test --batch-size 256 -Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_supervised_imagenet.yaml -target_entropy 1.3 -entropy_parameter 3.0  --epochs 200 -Li_lr 0.001 -Li_loss crossentropy -train_sample_amount 256 -train_only --no-aug --warmup-epochs 0 --lr 0.0 --lr-noise-pct 0.0 --lr-noise-std 0.0  -save_every 3 --start-epoch 0 -train_Li_only -mode 1 -n_copies 1