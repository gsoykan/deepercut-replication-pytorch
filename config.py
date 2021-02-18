dataset_dir = "/home/gsoykan20/Downloads/datasets/mpii_human_pose_v1/cropped/"
dataset_annotations_dir = "/home/gsoykan20/Downloads/datasets/mpii_human_pose_v1/cropped/dataset.mat"

dataset = "/home/gsoykan20/Downloads/datasets/mpii_human_pose_v1/cropped/dataset.mat"

num_joints = 14
all_joints = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9], [12], [13]]
all_joint_names = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'chin', 'forehead']

mirror = False
shuffle = False

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
location_refinement = False
weigh_only_present_joints = False
weigh_part_predictions = False
