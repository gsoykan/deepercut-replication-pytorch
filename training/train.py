import data_loader
import config
import torch
import time
import model.deepercut
from model.deepercut import DeeperCutHead
from model.deepercut import DeeperCut
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import accuracy.accuracy
import numpy as np
import copy
from dataset.pose_dataset import ActivityMode


def train_model(nn_model, dataloader, validation_dataloader, criterion, loc_ref_criterion, optimizer, scheduler,
                num_epochs=1, model_name="model"):
    since = time.time()
    best_model_wts = copy.deepcopy(nn_model.state_dict())
    best_acc = 0.0

    dataloaders = {
        'train': dataloader,
        'val': validation_dataloader
    }

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:  # 'val']:
            if phase == 'train':
                nn_model.train()  # Set model to training mode
            else:
                nn_model.eval()  # Set model to evaluate mode
            nn_model.freeze_bn()

            running_loss = 0.0
            running_accuracy = np.zeros((len(dataloaders[phase]), config.num_joints))
            # running_corrects = 0

            # Iterate over data.
            for i_batch, sample_batched in enumerate(dataloaders[phase]):
                input = sample_batched['image']
                scmap = sample_batched['scmap']
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = nn_model(input)
                    part_detection_result = output[DeeperCutHead.part_detection]
                    loss = criterion(part_detection_result, scmap)
                    locref_result = None
                    if config.location_refinement:
                        locref_map = sample_batched['locref_map']
                        locref_result = output[DeeperCutHead.locref]
                        locref_loss_weight = torch.as_tensor(config.locref_loss_weight)
                        if torch.cuda.is_available():
                            locref_loss_weight = locref_loss_weight.cuda()
                        raw_locref_loss = loc_ref_criterion(locref_result, locref_map)
                        locref_loss = locref_loss_weight * raw_locref_loss
                        loss += locref_loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                curr_loss = loss.item()
                if i_batch % 50 == 0:
                    print("batch_count" + str(i_batch))
                    print(curr_loss)
                running_loss += loss.item() * input.size(0)

                pose = accuracy.accuracy.argmax_pose_predict(part_detection_result, locref_result, config.stride)
                # scale can be computed here by comparing data_item's im_size and input size
                original_im_size = sample_batched['data_item']['im_size'][0][1:3]
                input_size = input.shape[2:4]
                scale = max(input_size / original_im_size.numpy())
                predictions = accuracy.accuracy.convert_pose_to_prediction(pose, scale)
                joints = sample_batched['data_item']['joints'][0][0].numpy()
                head_rect = sample_batched['data_item']['head_rect']
                acc_map_for__single_input = accuracy.accuracy.compare_predictions_with_joints(predictions, joints,
                                                                                              head_rect)
                running_accuracy[i_batch, :] = acc_map_for__single_input

                if phase == 'train':
                    scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = accuracy.accuracy.compute_accuracy_percentage_from_running_accuracy(running_accuracy)
            avg_epoch_acc = epoch_acc[config.num_joints]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, avg_epoch_acc))
            # deep copy the model
            if phase == 'val' and avg_epoch_acc > best_acc:
                best_acc = avg_epoch_acc
    		torch.save(nn_model, config.save_location + model_name + ".pth")
		# best_model_wts = copy.deepcopy(nn_model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    nn_model.load_state_dict(best_model_wts)
    return nn_model

def lr_determiner(iteration):
    """
    - [0.005, 10000]
    - [0.02, 430000]
    - [0.002, 730000]
    - [0.001, 1030000]
    """
    if iteration > 1030000:
        return 0.001
    elif iteration > 730000:
        return 0.002
    elif iteration > 430000:
        return 0.02
    else:
        return 0.005

def begin_training():
    val_dataloader = data_loader.create_dataloader(shuffle=False,
                                    activity_mode=ActivityMode.validation)
    dataloader = data_loader.create_dataloader()
    # sample_batched = next(iter(dataloader))

    nn_model = DeeperCut(config.num_joints)
    # model = torch.load( config.save_location + "model.pth")

    if torch.cuda.is_available():
        nn_model.to('cuda')

    # this will be moved to inside for dynamic weights
    criterion = nn.BCEWithLogitsLoss()
    loc_ref_criterion = nn.SmoothL1Loss()
    
    optimizer = optim.SGD(nn_model.parameters(), lr=0.005, momentum=0.9)
    
    lr_lambda = lambda iteration: lr_determiner(iteration)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model = train_model(nn_model, dataloader,
                        val_dataloader,
                        criterion,
                        loc_ref_criterion,
                        optimizer,
                        scheduler,
                        num_epochs=5, "mock_model" )

    # model = torch.load(PATH)
    # model.eval()
    
if __name__ == '__main__':
    begin_training()

