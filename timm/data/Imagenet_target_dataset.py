import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader

class Imagenet_target(Dataset):
    def __init__(self, annotations_file, img_dir=None, transform=None, target_transform=None):
        data=np.load(annotations_file)
        ind=np.arange(len(data))
        #np.random.shuffle(ind)
        self.target=torch.tensor(data[ind]).to(torch.float32)
        #self.target=torch.arange(50000).unsqueeze(1).to(torch.float32)
    def __len__(self):
        return self.target.shape[0]
    
    def __getitem__(self, idx):
        return self.target[idx]
    

if __name__=='__main__':
    IT=Imagenet_target('../../output/sample_resnet50_eval/correct_flag.npy')
    IT2=Imagenet_target('../../output/sample_resnet50_eval/correct_flag.npy')
    test_dataloader = DataLoader(IT, batch_size=64, shuffle=False)
    test_dataloader2 = DataLoader(IT2, batch_size=64, shuffle=False)

    for i, (data, data2) in enumerate(zip(test_dataloader, test_dataloader2)):
        print(data)
        print(data2)
        print(haha)