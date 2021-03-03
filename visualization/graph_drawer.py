from utils import loss_acc_recorder
import numpy as np
import matplotlib.pyplot as plt


def draw_loss_acc_graph_from_recorder(recorder_name, title_loss, title_acc):
    recorder = loss_acc_recorder.load_recorder(recorder_name)
    loss_train = recorder.running_training_losses
    loss_val = recorder.validation_losses
    acc_train = recorder.running_training_accuracies
    acc_val = recorder.validation_accuracies
    epochs = range(0, recorder.epoch)

    f1 = plt.figure(1)
    plt.plot(epochs, loss_train, 'g', label='Training Loss (Running)')
    plt.plot(epochs, loss_val, 'b', label='Validation Loss')
    plt.title(title_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    f1.show()

    f2 = plt.figure(2)
    plt.plot(epochs, acc_train, 'b', label='Training Acc (Running)')
    plt.plot(epochs, acc_val, 'r', label='Validation Acc')
    plt.title(title_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy PCKh@0.5')
    plt.legend()
    f2.show()


if __name__ == '__main__':
    draw_loss_acc_graph_from_recorder("resnet152_interm",
                                      title_loss="(ResNet-152) Training and Validation Loss",
                                      title_acc="(ResNet-152) Training and Validation Accuracy")
