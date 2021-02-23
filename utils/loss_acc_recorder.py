import pickle
import config

class LossAccRecorder:
    def __init__(self, model_name):
        self.model_name = model_name
        self.epoch = 0
        self.running_training_losses = []
        self.running_training_accuracies = []
        self.validation_accuracies = []
        self.validation_losses = []

    def increment_epoch(self):
        self.epoch += 1

    def add_training_info(self, loss, acc):
        self.running_training_accuracies.append(acc)
        self.running_training_losses.append(loss)

    def add_validation_info(self, loss, acc):
        self.validation_accuracies.append(acc)
        self.validation_losses.append(loss)

    def save_recorder(self):
        file_handler = open(config.save_location + self.model_name + "_lossacc_recorder.obj", 'wb')
        pickle.dump(self, file_handler)
        file_handler.close()


def load_recorder(recorder_name):
    filename = config.save_location + recorder_name + "_lossacc_recorder.obj"
    file_handler = open(filename, 'rb')
    recorder = pickle.Unpickler(file_handler).load()
    file_handler.close()
    return recorder