import numpy as np


def convert_tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()
