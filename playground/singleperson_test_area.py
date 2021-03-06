import config
import torch
from skimage import io
from torchvision import transforms
import accuracy.accuracy
import data_loader
from utils.utils import convert_tensor_to_numpy
from visualization import visualizer
from model.deepercut import DeeperCutHead
import os
import numpy as np


def predict_from_image_and_visualize(filename, model_name):
    input_image = io.imread(filename)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    model = torch.load(config.save_location + model_name + ".pth")
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
        part_detection = output[DeeperCutHead.part_detection]
        locref = output[DeeperCutHead.locref]
    pose = accuracy.accuracy.argmax_pose_predict(part_detection, locref, config.stride)
    visualizer.show_heatmaps(input_image, part_detection.cpu().numpy(), pose)


# https://www.newbedev.com/python/howto/how-to-iterate-over-files-in-a-given-directory/
def iterate_over_files_in_dir_and_visualize(directory,
                                            model_name,
                                            specific_paths
                                            ):
    directory = fr'{directory}'
    available_extensions = [".jpg", ".png", ".JPEG"]
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            filepath = subdir + os.sep + filename
            check = map(lambda ext: filepath.endswith(ext), available_extensions)
            if True in check:
                if (specific_paths is not None and filepath in specific_paths) or specific_paths is None:
                    predict_from_image_and_visualize(filepath, model_name=model_name)
                else:
                    continue


def visualize_ground_truth():
    dataloader = data_loader.create_dataloader(should_transform=False)
    for i_batch, sample_batched in enumerate(dataloader):
        input = sample_batched['image']
        scmap = sample_batched['scmap']
        pose = accuracy.accuracy.argmax_pose_predict(scmap, None, config.stride)
        input = np.squeeze(convert_tensor_to_numpy(input))
        input = np.transpose(input, (1, 2, 0))
        scmap = convert_tensor_to_numpy(scmap)
        visualizer.show_heatmaps(input, scmap, pose)
        print("visualizing" + str(i_batch))


if __name__ == '__main__':
    # predict_from_image_and_visualize("/home/gsoykan20/PycharmProjects/deepercut-pytorch/sample_images/s7.jpg", "resnet152_interm")
    #iterate_over_files_in_dir_and_visualize(config.sample_image_directory,
    #                                        "resnet152_interm",
    #                                        specific_paths=None)
    visualize_ground_truth()
    print("end of visualization")
