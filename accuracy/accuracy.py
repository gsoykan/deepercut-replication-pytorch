import numpy as np
import utils.utils as  utils
import config
import math


# Takes scmap and locref
def argmax_pose_predict(scmap, offmat, stride):
    """Combine scoremat and offsets to the final pose."""
    scmap = utils.convert_tensor_to_numpy(scmap)
    if offmat is not None:
        offmat = utils.convert_tensor_to_numpy(offmat)
    num_joints = scmap.shape[1]
    pose = []
    for joint_idx in range(num_joints):
        maxloc = np.unravel_index(np.argmax(scmap[0, joint_idx, :, :]),
                                  scmap[0, joint_idx, :, :].shape)

        offset = np.array(offmat[maxloc][joint_idx])[::-1] if offmat is not None else 0
        pos_f8 = (np.array(maxloc).astype('float') * stride + 0.5 * stride +
                  offset)
        pose.append(np.hstack((pos_f8[::-1],
                               [scmap[0][joint_idx][maxloc]])))
    return np.array(pose)


def convert_pose_to_prediction(pose, global_scale):
    pose_refscale = np.copy(pose)
    pose_refscale[:, 0:2] /= global_scale
    return pose_refscale


def get_head_size(x1, y1, x2, y2):
    head_size = config.sc_bias * math.dist([x1, y1], [x2, y2])
    return head_size


def get_distance_in_PCKh(pred, gt, head_size):
    pckh = np.sqrt(np.sum((pred - gt) ** 2)) / head_size
    return pckh


def compare_predictions_with_joints(predictions, joints, head_rect):
    acc_map = np.zeros(config.num_joints)
    head_size = get_head_size(head_rect['x1'],
                              head_rect['y1'],
                              head_rect['x2'],
                              head_rect['y2'])
    acc_map[:] = -1
    for joint in joints:
        joint_id = joint[0]
        gt = joint[1:3]
        pred = predictions[joint_id][:2] # list(reversed(predictions[joint_id][:2]))
        pckh_distance = get_distance_in_PCKh(pred, gt, head_size)
        if pckh_distance <= config.PCKh_limit:
            acc_map[joint_id] = 1
        else:
            acc_map[joint_id] = 0
    return acc_map


def compute_accuracy_percentage_from_running_accuracy(running_accuracy):
    acc = np.zeros(config.num_joints + 1)
    acc_sum = 0
    for j_id in range(config.num_joints):
        number_of_available_joints = sum(1 for e in running_accuracy[:, j_id] > -1 if e == True)
        number_of_correct_joints = sum(1 for e in running_accuracy[:, j_id] > 0 if e == True)
        joint_percentage = (number_of_correct_joints / number_of_available_joints) * 100.0
        acc_sum += joint_percentage
        acc[j_id] = joint_percentage
    acc[config.num_joints] = acc_sum / config.num_joints
    return acc
