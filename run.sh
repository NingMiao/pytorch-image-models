#Baseline
#python3 train.py /mnt/ssd/imagenet/ --device tpu-8 --experiment resnet-50-from_scratch --resume output/train/resnet-50-from_scratch/model.ckpt --batch-size 512

#Li
python3 train.py /mnt/ssd/imagenet/ --device tpu-8 --experiment resnet-50-from_scratch --resume output/train/resnet-50-from_scratch/model.ckpt --batch-size 512  --scale [0.8, 1.0]