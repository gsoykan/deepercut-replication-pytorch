import logging
import random as rand
from enum import Enum

import numpy as np
from numpy import array as arr
from numpy import concatenate as cat

import scipy.io as sio
from skimage import io, transform
import copy

from data_models.data_models import DataItem
from utils.annotator_json_reader import read_comic_dataset


def extend_crop(crop, crop_pad, image_size):
    crop[0] = max(crop[0] - crop_pad, 0)
    crop[1] = max(crop[1] - crop_pad, 0)
    crop[2] = min(crop[2] + crop_pad, image_size[2] - 1)
    crop[3] = min(crop[3] + crop_pad, image_size[1] - 1)
    return crop


def mirror_joints_map(all_joints, num_joints):
    res = np.arange(num_joints)
    symmetric_joints = [p for p in all_joints if len(p) == 2]
    for pair in symmetric_joints:
        res[pair[0]] = pair[1]
        res[pair[1]] = pair[0]
    return res


def load_pairwise_stats(cfg):
    mat_stats = sio.loadmat(cfg.pairwise_stats_fn)
    pairwise_stats = {}
    for id in range(len(mat_stats['graph'])):
        pair = tuple(mat_stats['graph'][id])
        pairwise_stats[pair] = {"mean": mat_stats['means'][id], "std": mat_stats['std_devs'][id]}
    for pair in pairwise_stats:
        pairwise_stats[pair]["mean"] *= cfg.global_scale
        pairwise_stats[pair]["std"] *= cfg.global_scale
    return pairwise_stats


def get_pairwise_index(j_id, j_id_end, num_joints):
    return (num_joints - 1) * j_id + j_id_end - int(j_id < j_id_end)


def data_to_input(data):
    return np.expand_dims(data, axis=0).astype(float)

class ActivityMode(Enum):
    training = 1
    validation = 2
    test = 3


class PoseDataset:
    def __init__(self, cfg, activity_mode=ActivityMode.training):
        self.activity_mode = activity_mode
        self.cfg = cfg

        if cfg.use_comic_data:
            self.data = self.load_comic_dataset() if cfg.comic_annotations_json_location else []
        else:
            self.data = self.load_mpii_dataset() if cfg.dataset else []

        self.num_images = len(self.data)
        if activity_mode is ActivityMode.training:
            self.mirror = self.cfg.mirror
        else:
            self.mirror = False

        if self.mirror:
            self.symmetric_joints = mirror_joints_map(cfg.all_joints, cfg.num_joints)
        self.curr_img = 0
        if activity_mode is ActivityMode.training:
            self.set_shuffle(cfg.shuffle)
        else:
            self.set_shuffle(False)

        self.set_pairwise_stats_collect(cfg.pairwise_stats_collect)
        # if self.cfg.pairwise_predict:
        # self.pairwise_stats = load_pairwise_stats(self.cfg)


    def load_comic_dataset(self):
        cfg = self.cfg
        whole_dataset = read_comic_dataset()
        max_num_images = len(whole_dataset)
        if self.activity_mode == ActivityMode.training:
            start_index = 0
            end_index = cfg.train_dataset_length

        if self.activity_mode == ActivityMode.validation:
            start_index = cfg.train_dataset_length
            end_index = start_index + cfg.validation_dataset_length

        if self.activity_mode == ActivityMode.test:
            start_index = cfg.validation_dataset_length + cfg.train_dataset_length
            end_index = start_index + cfg.test_dataset_length

        if end_index > max_num_images:
            assert True, "End index should not be greater than max num images"
        data = []
        has_gt = True
        for i in range(start_index, end_index):
            item = whole_dataset[i]
            data.append(item)
        self.has_gt = has_gt
        return data


    def load_mpii_dataset(self):
        cfg = self.cfg
        file_name = cfg.dataset
        mlab = sio.loadmat(file_name)

        if cfg.dataset_type == 'single':
            cropped_annotations = sio.loadmat(cfg.annolist_single_person)['annolist']
            annotation_single_paths = [cropped_annotations[0, i][0][0][0][0][0] for i in
                                       range(cropped_annotations.shape[1])]

        self.raw_data = mlab
        mlab = mlab['dataset']
        max_num_images = mlab.shape[1]

        if self.activity_mode == ActivityMode.training:
            start_index = 0
            end_index = cfg.train_dataset_length

        if self.activity_mode == ActivityMode.validation:
            start_index = cfg.train_dataset_length
            end_index = start_index + cfg.validation_dataset_length

        if self.activity_mode == ActivityMode.test:
            start_index = cfg.validation_dataset_length + cfg.train_dataset_length
            end_index = start_index + cfg.test_dataset_length

        if end_index > max_num_images:
            assert True, "End index should not be greater than max num images"

        data = []
        has_gt = True

        for i in range(start_index, end_index):
            sample = mlab[0, i]
            im_path = sample[0][0]

            if cfg.dataset_type == 'single' and im_path not in annotation_single_paths:
                continue
            else:
                annotation_index = annotation_single_paths.index(im_path)

            item = DataItem()
            item.image_id = i
            item.im_path = im_path
            item.im_size = sample[1][0]

            if cropped_annotations is not None:
                item.head_rect = {}
                head_rect_info = cropped_annotations['annorect'][0, annotation_index][0, 0]
                item.head_rect['x1'] = float(head_rect_info['x1'][0, 0])
                item.head_rect['x2'] = float(head_rect_info['x2'][0, 0])
                item.head_rect['y1'] = float(head_rect_info['y1'][0, 0])
                item.head_rect['y2'] = float(head_rect_info['y2'][0, 0])

            if len(sample) >= 3:
                joints = sample[2][0][0]
                joint_id = joints[:, 0]
                # make sure joint ids are 0-indexed
                if joint_id.size != 0:
                    assert ((joint_id < cfg.num_joints).any())
                joints[:, 0] = joint_id
                item.joints = [joints]
            else:
                has_gt = False
            if cfg.crop:
                crop = sample[3][0] - 1
                item.crop = extend_crop(crop, cfg.crop_pad, item.im_size)
            data.append(item)
        self.has_gt = has_gt
        return data

    def set_pairwise_stats_collect(self, pairwise_stats_collect):
        self.pairwise_stats_collect = pairwise_stats_collect
        if self.pairwise_stats_collect:
            assert self.get_scale() == 1.0

    def get_scale(self):
        cfg = self.cfg
        scale = cfg.global_scale
        if hasattr(cfg, 'scale_jitter_lo') and hasattr(cfg, 'scale_jitter_up'):
            scale_jitter = rand.uniform(cfg.scale_jitter_lo, cfg.scale_jitter_up)
            scale *= scale_jitter
        return scale

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle
        if not shuffle:
            self.image_indices = np.arange(self.num_images)
            #  assert not self.mirror

    # BATCH RELATED
    def next_batch(self):
        while True:
            imidx, mirror = self.next_training_sample()
            data_item = self.get_training_sample(imidx)
            scale = self.get_scale()

            if not self.is_valid_size(data_item.im_size, scale):
                continue

            return self.make_batch(data_item, scale, mirror)

    def next_training_sample(self):
        if self.curr_img == 0 and self.shuffle:
            self.shuffle_images()

        curr_img = self.curr_img
        self.curr_img = (self.curr_img + 1) % self.num_training_samples()

        imidx = self.image_indices[curr_img]
        mirror = self.mirror and self.mirrored[curr_img]

        return imidx, mirror

    def fetch_item_at_index(self, idx):
        if self.curr_img == 0 and self.shuffle:
            self.shuffle_images()
        curr_img = idx
        imidx = self.image_indices[curr_img]
        mirror = self.mirror and self.mirrored[curr_img]
        data_item = self.get_training_sample(imidx)
        scale = self.get_scale()
        if not self.is_valid_size(data_item.im_size, scale):
            assert True, "Image is not in valid size"
        batch = self.make_batch(data_item, scale, mirror)
        return batch

    def num_training_samples(self):
        num = self.num_images
        if self.mirror:
            num *= 2
        return num

    def get_training_sample(self, imidx):
        return self.data[imidx]

    def shuffle_images(self):
        num_images = self.num_images
        if self.mirror:
            image_indices = np.random.permutation(num_images * 2)
            self.mirrored = image_indices >= num_images
            image_indices[self.mirrored] = image_indices[self.mirrored] - num_images
            self.image_indices = image_indices
        else:
            self.image_indices = np.random.permutation(num_images)

    def make_batch(self, data_item_original, scale, mirror):
        im_file = data_item_original.im_path
        logging.debug('image %s', im_file)
        logging.debug('mirror %r', mirror)
        image = io.imread(im_file)
        data_item = data_item_original

        if self.has_gt:
            joints = np.copy(data_item.joints)

        if self.cfg.crop:
            crop = data_item.crop
            image = image[crop[1]:crop[3] + 1, crop[0]:crop[2] + 1, :]
            if self.has_gt:
                joints[:, 1:3] -= crop[0:2].astype(joints.dtype)

        img = transform.rescale(image, scale) if scale != 1 else image
        scaled_img_size = arr(img.shape[0:2])

        if mirror:
            img = np.fliplr(img).copy()

        batch = {'image': img}

        if self.has_gt:
            stride = self.cfg.stride
            if mirror:
                joints = [self.mirror_joints(person_joints, self.symmetric_joints, image.shape[1]) for person_joints in
                          joints]
                data_item = copy.deepcopy(data_item)
                data_item.joints = [self.mirror_joints(person_joints, self.symmetric_joints, image.shape[1]) for person_joints in
                          data_item.joints]

            sm_size = np.ceil(scaled_img_size / (stride * 2)).astype(int) * 2
            scaled_joints = [person_joints[:, 1:3] * scale for person_joints in joints]
            joint_id = [person_joints[:, 0].astype(int) for person_joints in joints]
            scmap, scmap_weights, locref_map, locref_mask, pairwise_map, pairwise_mask = self.compute_targets_and_weights(
                joint_id,
                scaled_joints,
                data_item,
                sm_size,
                scale)
            batch['scmap'] = scmap
            batch['scmap_weights'] = scmap_weights
            if self.cfg.location_refinement:
                batch['locref_map'] = locref_map
                batch['locref_mask'] = locref_mask
            if self.cfg.pairwise_predict:
                batch['pairwise_map'] = pairwise_map
                batch['pairwise_mask'] = pairwise_mask

        batch['data_item'] = data_item
        return batch

    def mirror_joints(self, joints, symmetric_joints, image_width):
        # joint ids are 0 indexed
        res = np.copy(joints)
        res = self.mirror_joint_coords(res, image_width)
        # swap the joint_id for a symmetric one
        joint_id = joints[:, 0].astype(int)
        res[:, 0] = symmetric_joints[joint_id]
        return res

    def mirror_joint_coords(self, joints, image_width):
        # horizontally flip the x-coordinate, keep y unchanged
        joints[:, 1] = image_width - joints[:, 1] - 1
        return joints

    def is_valid_size(self, image_size, scale):
        im_width = image_size[2]
        im_height = image_size[1]

        raw_min_input_size = 100
        if im_height < raw_min_input_size or im_width < raw_min_input_size:
            return False

        if hasattr(self.cfg, 'max_input_size'):
            max_input_size = self.cfg.max_input_size
            input_width = im_width * scale
            input_height = im_height * scale
            if input_height * input_width > max_input_size * max_input_size:
                return False

        return True

    def set_locref(self, locref_map, locref_mask, locref_scale, i, j, j_id, dx, dy):
        locref_mask[j, i, j_id * 2 + 0] = 1
        locref_mask[j, i, j_id * 2 + 1] = 1
        locref_map[j, i, j_id * 2 + 0] = dx * locref_scale
        locref_map[j, i, j_id * 2 + 1] = dy * locref_scale

    def set_pairwise_map(self, pairwise_map, pairwise_mask, i, j, j_id, j_id_end, coords, pt_x, pt_y, person_id, k_end):
        num_joints = self.cfg.num_joints
        joint_pt = coords[person_id][k_end, :]
        j_x_end = np.asscalar(joint_pt[0])
        j_y_end = np.asscalar(joint_pt[1])
        pair_id = get_pairwise_index(j_id, j_id_end, num_joints)
        stats = self.pairwise_stats[(j_id, j_id_end)]
        dx = j_x_end - pt_x
        dy = j_y_end - pt_y
        pairwise_mask[j, i, pair_id * 2 + 0] = 1
        pairwise_mask[j, i, pair_id * 2 + 1] = 1
        pairwise_map[j, i, pair_id * 2 + 0] = (dx - stats["mean"][0]) / stats["std"][0]
        pairwise_map[j, i, pair_id * 2 + 1] = (dy - stats["mean"][1]) / stats["std"][1]

    def compute_scmap_weights(self, scmap_shape, joint_id, data_item):
        cfg = self.cfg
        if cfg.weigh_only_present_joints:
            weights = np.zeros(scmap_shape)
            for person_joint_id in joint_id:
                for j_id in person_joint_id:
                    weights[:, :, j_id] = 1.0
        else:
            weights = np.ones(scmap_shape)
        return weights

    def compute_targets_and_weights(self, joint_id, coords, data_item, size, scale):
        stride = self.cfg.stride
        dist_thresh = self.cfg.pos_dist_thresh * scale
        num_joints = self.cfg.num_joints
        half_stride = stride / 2
        scmap = np.zeros(cat([size, arr([num_joints])]))

        locref_shape = cat([size, arr([num_joints * 2])])
        locref_mask = np.zeros(locref_shape)
        locref_map = np.zeros(locref_shape)

        pairwise_shape = cat([size, arr([num_joints * (num_joints - 1) * 2])])
        pairwise_mask = np.zeros(pairwise_shape)
        pairwise_map = np.zeros(pairwise_shape)

        dist_thresh_sq = dist_thresh ** 2

        width = size[1]
        height = size[0]

        for person_id in range(len(coords)):
            for k, j_id in enumerate(joint_id[person_id]):
                joint_pt = coords[person_id][k, :]
                j_x = np.asscalar(joint_pt[0])
                j_y = np.asscalar(joint_pt[1])

                # don't loop over entire heatmap, but just relevant locations
                j_x_sm = round((j_x - half_stride) / stride)
                j_y_sm = round((j_y - half_stride) / stride)
                min_x = round(max(j_x_sm - dist_thresh - 1, 0))
                max_x = round(min(j_x_sm + dist_thresh + 1, width - 1))
                min_y = round(max(j_y_sm - dist_thresh - 1, 0))
                max_y = round(min(j_y_sm + dist_thresh + 1, height - 1))

                for j in range(min_y, max_y + 1):  # range(height):
                    pt_y = j * stride + half_stride
                    for i in range(min_x, max_x + 1):  # range(width):
                        pt_x = i * stride + half_stride
                        dx = j_x - pt_x
                        dy = j_y - pt_y
                        dist = dx ** 2 + dy ** 2

                        if dist <= dist_thresh_sq:
                            locref_scale = 1.0 / self.cfg.locref_stdev
                            current_normalized_dist = dist * locref_scale ** 2
                            prev_normalized_dist = locref_map[j, i, j_id * 2 + 0] ** 2 + \
                                                   locref_map[j, i, j_id * 2 + 1] ** 2
                            update_scores = (scmap[j, i, j_id] == 0) or prev_normalized_dist > current_normalized_dist
                            if self.cfg.location_refinement and update_scores:
                                self.set_locref(locref_map, locref_mask, locref_scale, i, j, j_id, dx, dy)
                            if self.cfg.pairwise_predict and update_scores:
                                for k_end, j_id_end in enumerate(joint_id[person_id]):
                                    if k != k_end:
                                        self.set_pairwise_map(pairwise_map, pairwise_mask, i, j, j_id, j_id_end,
                                                              coords, pt_x, pt_y, person_id, k_end)
                            scmap[j, i, j_id] = 1
        scmap_weights = self.compute_scmap_weights(scmap.shape, joint_id, data_item)
        scmap = scmap.transpose((2, 0, 1))
        scmap_weights = scmap_weights.transpose((2, 0, 1))
        locref_map = locref_map.transpose((2, 0, 1))
        locref_mask = locref_mask.transpose((2, 0, 1))
        pairwise_map = pairwise_map.transpose((2, 0, 1))
        pairwise_mask = pairwise_mask.transpose((2, 0, 1))
        return scmap, scmap_weights, locref_map, locref_mask, pairwise_map, pairwise_mask
