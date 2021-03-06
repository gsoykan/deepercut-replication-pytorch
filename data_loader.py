from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from scipy.io import loadmat
import config
import dataset.mpii
from dataset.pose_dataset import ActivityMode


# TODO: WRITE a COLLATE FUNCTION FOR BATCH SIZE > 1
def create_dataloader(shuffle=False,
                      activity_mode=ActivityMode.training,
                      should_transform=True):
    mpii_dataset = dataset.mpii.MPIIDataset(config, activity_mode, should_transform)
    dataloader = DataLoader(mpii_dataset, batch_size=1, shuffle=shuffle)
    return dataloader