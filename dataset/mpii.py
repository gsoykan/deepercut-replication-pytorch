from torch.utils.data.dataset import T_co

from dataset.pose_dataset import PoseDataset
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class MPII(PoseDataset):
    def __init__(self, cfg):
        cfg.all_joints = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9], [12], [13]]
        cfg.all_joints_names = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'chin', 'forehead']
        cfg.num_joints = 14
        super().__init__(cfg)

    def mirror_joint_coords(self, joints, image_width):
        joints[:, 1] = image_width - joints[:, 1]
        return joints

    def get_pose_segments(self):
        return [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13]]


class MPIIDataset(Dataset):
    def __init__(self, cfg):
        self.mpii = MPII(cfg)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx) -> T_co:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.mpii.fetch_item_at_index(idx)
        img = sample['image']
        img_tensor = self.transforms(img)
        sample['image'] = img_tensor  # .unsqueeze(0)

        if torch.cuda.is_available():
            sample['image'] = sample['image'].to('cuda', dtype=torch.float)
            for key, value in sample.items():
                if key == 'image' or key == 'data_item':
                    continue
                value = value.astype('float32')
                value = torch.Tensor(value)
                sample[key] = value.to('cuda', dtype=torch.float)

        sample['data_item'] = sample['data_item'].__dict__
        sample['data_item']['im_size'] = sample['data_item']['im_size'].astype('float32')

        return sample

    def __len__(self):
        num_samples = self.mpii.num_training_samples()
        return num_samples

    def make_dataloader(self):
        DataLoader(self, batch_size=1, shuffle=False, )
