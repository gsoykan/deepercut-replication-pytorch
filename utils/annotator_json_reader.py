import config
import json
import numpy as np
from data_models.data_models import DataItem


def read_comic_dataset():
    json_location = config.comic_annotations_json_location
    with open(json_location) as f:
        data = json.load(f)
        annotations = data['root']
    return create_dataset_from_annotations(annotations)

def create_dataset_from_annotations(annotations):
    dataset = []
    for idx, annotation in enumerate(annotations):
        item = DataItem()
        item.image_id = idx
        item.im_path = annotation['image_path']
        item.im_size = np.array(annotation['im_size'])
        item.head_rect = {}
        item.head_rect['x1'] = float(annotation['head_rect_top_left'][0])
        item.head_rect['x2'] = float(annotation['head_rect_bottom_right'][0])
        item.head_rect['y1'] = float(annotation['head_rect_top_left'][1])
        item.head_rect['y2'] = float(annotation['head_rect_bottom_right'][1])
        joints = create_joints_for_dataitem(annotation)
        item.joints = [joints]
        dataset.append(item)
    return dataset


def create_joints_for_dataitem(annotation):
    joint_keys = [
        "joint_0_(r_ankle)",
        "joint_1_(r_knee)",
        "joint_2_(r_hip)",
        "joint_3_(l_hip)",
        "joint_4_(l_knee)",
        "joint_5_(l_ankle)",
        "joint_6_(r_wrist)",
        "joint_7_(r_elbow)",
        "joint_8_(r_shoulder)",
        "joint_9_(l_shoulder)",
        "joint_10_(l_elbow)",
        "joint_11_(l_wrist)",
        "joint_12_(chin)",
        "joint_13_(top of head)"
    ]
    joints = []
    for key in annotation.keys():
        if key in joint_keys:
            joint_id = int(key.split('_')[1])
            value = np.array(annotation[key]).astype('int32')
            value = np.insert(value, 0, joint_id)
            joints.append(value)
    return np.array(joints)


if __name__ == '__main__':
    read_comic_dataset()
