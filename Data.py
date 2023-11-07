import glob, os


def get_image(dir_path):
    return glob.glob(dir_path + "*.png")


def get_truth_data(dir_path):
    return glob.glob(dir_path + "*.txt")


class Data:
    train_data = []
    test_data = []

    truth_train_data = []
    truth_test_data = []

    def __init__(self, train_percentage, data_dir):
        all_data = get_image(data_dir)
        self.train_data = all_data[:int(len(all_data) * train_percentage)]
        self.test_data = all_data[int(len(all_data) * train_percentage):]

        all_truth_data = get_truth_data(data_dir)
        self.truth_train_data = all_truth_data[:int(len(all_truth_data) * train_percentage)]
        self.truth_test_data = all_truth_data[int(len(all_truth_data) * train_percentage):]
