import glob, os
from skimage import feature


def get_image(dir_path):
    return glob.glob(dir_path + "*.png")


def get_classes(dir_path):
    return glob.glob(dir_path + "*.png")


def get_truth_data(dir_path):
    return glob.glob(dir_path + "*.txt")


class Data:
    train_data = []
    test_data = []

    truth_train_data = []
    truth_test_data = []

    train_data_cropped = []
    test_data_cropped = []

    class_by_img = {}

    def __init__(self, train_percentage, data_dir, data_class_txt):
        all_data = get_image(data_dir)
        #self.train_data = all_data[:int(len(all_data) * train_percentage)]
        self.train_data = all_data[:3]
        #self.test_data = all_data[int(len(all_data) * train_percentage):]
        self.test_data = all_data[:3]

        self.train_data_cropped = []
        self.test_data_cropped = []

        all_truth_data = get_truth_data(data_dir)
        self.truth_train_data = all_truth_data[:int(len(all_truth_data) * train_percentage)]
        self.truth_test_data = all_truth_data[int(len(all_truth_data) * train_percentage):]

        with open(data_class_txt, 'r', encoding="utf8") as file:
            for line in file:
                split=line.strip().split(" ")
                img_name = int((split[0].split("."))[0])
                img_class = int(split[1])
                self.class_by_img[img_name] = img_class

    def imageToTrain(self, img):
        self.train_data_cropped.append(img)

    def imageToTest(self, img):
        self.test_data_cropped.append(img)
