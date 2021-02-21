import data_loader
import dataset.mpii
import config
import torch
import time
import model.deepercut
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import accuracy.accuracy
import numpy as np
import copy
from dataset.pose_dataset import ActivityMode


def train_model(nn_model, dataloader, validation_dataloader, criterion, optimizer, scheduler, num_epochs=1):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
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
                    loss = criterion(output, scmap)
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

                # TODO: this can only take first 14 channels of output
                # TODO: compute offset
                pose = accuracy.accuracy.argmax_pose_predict(output, None, config.stride)
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
                best_model_wts = copy.deepcopy(nn_model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    nn_model.load_state_dict(best_model_wts)
    return nn_model


if __name__ == '__main__':
    val_dataloader = data_loader.create_dataloader(shuffle=False,
                                                   activity_mode=ActivityMode.validation)
    dataloader = data_loader.create_dataloader()
    # sample_batched = next(iter(dataloader))

    model = model.deepercut.DeeperCut(config.num_joints)
    # model = torch.load( config.save_location + "model.pth")

    if torch.cuda.is_available():
        model.to('cuda')

    # this will be moved to inside for dynamic weights
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model = train_model(model, dataloader,
                        val_dataloader,
                        criterion,
                        optimizer,
                        scheduler,
                        num_epochs=20)

    torch.save(model, config.save_location + "model.pth")
    # model = torch.load(PATH)
    # model.eval()
