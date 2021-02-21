import config
import torch
from skimage import io
from torchvision import transforms
import accuracy.accuracy
import visualizer
from model.deepercut import DeeperCutHead


def predict_from_image_and_visualize(filename):
    input_image = io.imread(filename)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    model = torch.load(config.save_location + "model.pth")
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


if __name__ == '__main__':
    predict_from_image_and_visualize(config.sample_image_path)
