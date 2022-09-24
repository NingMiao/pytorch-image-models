#Baseline
#python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device tpu-8 --experiment resnet-50-no_aug_from_scratch --batch-size 512 --epochs 90 --lr 0.2 --warmup-epoch 0 --cooldown-epochs 0 --sched step --no-aug

#python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device tpu-8 --experiment resnet-50-random_crop_from_scratch_new_transformation_fixed_with_ratio --batch-size 512 --epochs 90 --lr 0.2 --warmup-epoch 0 --cooldown-epochs 0 --sched step --decay-epochs 30 -save_every 5 --color-jitter 0.0

#python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device tpu-8 --experiment resnet-50-aa_from_scratch --batch-size 512 --epochs 90 --lr 0.2 --warmup-epoch 0 --cooldown-epochs 0 --sched step --aa original --decay-epochs 30

#python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device tpu-8 --experiment resnet-50-ra_aug_from_scratch --batch-size 512 --epochs 90 --lr 0.2 --warmup-epoch 0 --cooldown-epochs 0 --sched step --aa rand-m9-mstd0.5

#python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device tpu-8 --experiment resnet-50-aa_from_scratch_continued --batch-size 512 --epochs 90 --lr 0.2 --warmup-epoch 0 --cooldown-epochs 0 --sched step --aa original --decay-epochs 30 --resume output/train/resnet-50-aa_from_scratch/model28.ckpt --start-epoch 29

#python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device tpu-8 --experiment resnet-50-test --batch-size 512 --epochs 90 --lr 0.2 --warmup-epoch 0 --cooldown-epochs 0 --sched step --decay-epochs 30 --color-jitter 0.0 -save_every 5  --start-epoch 40 --no-aug -Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_supervised_imagenet_new_param.yaml -Li_lr 0.1 -target_entropy 0.0 -entropy_parameter 0.3 -vrm global

#Get patch logit
#on eval
#python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device tpu-8 --experiment resnet-50-test --batch-size 16 --lr 0.2 --warmup-epoch 0 --cooldown-epochs 0 --sched step --decay-epochs 30 --color-jitter 0.0 -save_every 3 --pretrained --start-epoch 40 --cooldown-epochs 0 --scale 0.8 1.0 --epochs 41  -Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_supervised_imagenet_new_param.yaml -eval_only -sample_save output/sample_resnet50_eval/

#on train
python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device tpu-8 --experiment resnet-50-test --batch-size 16 --lr 0.2 --warmup-epoch 0 --cooldown-epochs 0 --sched step --decay-epochs 30 --color-jitter 0.0 -save_every 3 --pretrained --start-epoch 40 --cooldown-epochs 0 --scale 0.8 1.0 --epochs 41  -Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_supervised_imagenet_new_param.yaml -eval_only --no-aug -eval_on_train -eval_max_batch 14000 -sample_save output/sample_resnet50_train/

#Only train Li
#python3 train.py /mnt/ssd/imagenet/ --device tpu-8 --experiment resnet-50-Li-only_Li_te_3.0_new --batch-size 256 -Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_supervised_imagenet.yaml -target_entropy 3.0 -entropy_parameter 0.03  --pretrained --epochs 200 -Li_lr 0.1 -Li_loss crossentropy -batch_amount 0 -train_only --pretrained --no-aug --warmup-epochs 0 --lr 0.0 --lr-noise-pct 0.0 --lr-noise-std 0.0  -save_every 3 --start-epoch 170 -train_Li_only

#Train model and Li
#python3 train.py /mnt/ssd/imagenet/ --model resnet50 --device tpu-8 --experiment resnet-50-test --batch-size 512 --epochs 90 --lr 0.2 --warmup-epoch 0 --cooldown-epochs 0 --sched step --decay-epochs 30 --color-jitter 0.0 --scale 1.0 1.0 -Li_config_path ../InstaAug/InstaAug_module/configs/config_crop_new_param_supervised_imagenet.yaml -target_entropy 3.0 -entropy_parameter 0.03 --start-epoch 40 --resume output/train/resnet-50-aa_from_scratch_continued/model39.ckpt -Li_lr 0.01 -Li_loss crossentropy --aa original

