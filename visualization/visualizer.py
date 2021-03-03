import math
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import io, transform
import config


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


def show_heatmaps(img, scmap, pose, cmap="jet"):
    scmap = np.squeeze(scmap).transpose((1, 2, 0))
    interp = "bilinear"
    all_joints = config.all_joints
    all_joints_names = config.all_joint_names
    subplot_width = 3
    subplot_height = math.ceil((len(all_joints) + 1) / subplot_width)
    f, axarr = plt.subplots(subplot_height, subplot_width)
    for pidx, part in enumerate(all_joints):
        plot_j = (pidx + 1) // subplot_width
        plot_i = (pidx + 1) % subplot_width
        scmap_part = np.sum(scmap[:, :, part], axis=2)
        scmap_part = transform.resize(scmap_part, np.multiply(8, scmap_part.shape))
        scmap_part = np.lib.pad(scmap_part, ((4, 0), (4, 0)), 'minimum')
        curr_plot = axarr[plot_j, plot_i]
        curr_plot.set_title(all_joints_names[pidx])
        curr_plot.axis('off')
        curr_plot.imshow(img, interpolation=interp)
        curr_plot.imshow(scmap_part, alpha=.5, cmap=cmap, interpolation=interp)

    curr_plot = axarr[0, 0]
    curr_plot.set_title('Pose')
    curr_plot.axis('off')
    curr_plot.imshow(visualize_joints(img, pose))
    plt.show()


def visualize_joints(image, pose):
    marker_size = 8
    minx = 2 * marker_size
    miny = 2 * marker_size
    maxx = image.shape[1] - 2 * marker_size
    maxy = image.shape[0] - 2 * marker_size
    num_joints = pose.shape[0]

    visim = image.copy()
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
              [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
              [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    for p_idx in range(num_joints):
        cur_x = pose[p_idx, 0]
        cur_y = pose[p_idx, 1]
        if check_point(cur_x, cur_y, minx, miny, maxx, maxy):
            _npcircle(visim,
                      cur_x, cur_y,
                      marker_size,
                      colors[p_idx],
                      0.0)
    return visim


def check_point(cur_x, cur_y, minx, miny, maxx, maxy):
    return minx < cur_x < maxx and miny < cur_y < maxy


def _npcircle(image, cx, cy, radius, color, transparency=0.0):
    """Draw a circle on an image using only numpy methods."""
    radius = int(radius)
    cx = int(cx)
    cy = int(cy)
    y, x = np.ogrid[-radius: radius, -radius: radius]
    index = x ** 2 + y ** 2 <= radius ** 2
    image[cy - radius:cy + radius, cx - radius:cx + radius][index] = (
            image[cy - radius:cy + radius, cx - radius:cx + radius][index].astype('float32') * transparency +
            np.array(color).astype('float32') * (1.0 - transparency)).astype('uint8')
