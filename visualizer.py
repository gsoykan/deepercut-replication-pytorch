import matplotlib.pyplot as plt
import torch
import numpy as np

def show_tensor_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

# TODO: Not working as expected, FIX IT
def show_marks(image, marks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(marks[:, 0], marks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.cpu().numpy()
    inp = np.squeeze(inp, 0)
    inp = inp.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
