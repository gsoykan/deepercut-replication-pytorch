dataset_dir = "/home/gsoykan20/Downloads/datasets/mpii_human_pose_v1/cropped/"
dataset_annotations_dir = "/home/gsoykan20/Downloads/datasets/mpii_human_pose_v1/cropped/dataset.mat"

dataset = "/home/gsoykan20/Downloads/datasets/mpii_human_pose_v1/cropped/dataset.mat"
annolist_single_person = "/home/gsoykan20/Downloads/datasets/mpii_human_pose_v1/cropped/annolist-singlePerson-h400.mat"
raw_annolist = "/home/gsoykan20/Downloads/datasets/mpii_human_pose_v1/mpii_human_pose_v1_u12_1.mat"

save_location = "/home/gsoykan20/PycharmProjects/deepercut-pytorch/saved/"

sample_image_path = "/home/gsoykan20/PycharmProjects/deepercut-pytorch/sample_images/image.png"

dataset_type = 'single'
train_dataset_length = 200
validation_dataset_length = 200
test_dataset_length = 100

mirror = True
shuffle = True
# Learn About This Parameters
pairwise_stats_collect = False
pairwise_predict = False
# pairwise_stats_fn = "pairwise_stats.mat"
crop = False

# scale_jitter_lo = 0.85
# scale_jitter_up = 1.15
global_scale = 0.8452830189
max_input_size = 850
stride = 8

pos_dist_thresh = 17
locref_stdev = 7.2801
# used in accuracy head calculation
sc_bias = 0.6
PCKh_limit = 0.5
location_refinement = True
weigh_only_present_joints = False
weigh_part_predictions = False
enable_skip_connections = True

locref_loss_weight = 0.05

num_joints = 14
all_joints = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9], [12], [13]]
all_joint_names = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'chin', 'forehead']
