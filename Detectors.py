import math

import cv2, sys, glob
import numpy as np
from numpy import dot
from numpy.linalg import norm
from skimage import feature

left_ear_cascade = cv2.CascadeClassifier("./Data/haarcascade_mcs_leftear.xml")
right_ear_cascade = cv2.CascadeClassifier("./Data/haarcascade_mcs_rightear.xml")


def vizualization(img, detectionList, image_num):
    y = int(detectionList[1])
    x = int(detectionList[0])
    w = int(detectionList[2])
    h = int(detectionList[3])

    cropped_image = img[y:y + h, x:x + w]
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("./VJ_Images/" + str(image_num) + '_detected.jpg', gray)
    return gray


def calc_IoU_score(rect_stat, ground_truth):
    x1, y1, w1, h1 = rect_stat
    x2, y2, w2, h2 = ground_truth

    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)

    intersection_area = w_intersection * h_intersection

    area1 = w1 * h1
    area2 = w2 * h2

    iou = intersection_area / float(area1 + area2 - intersection_area)

    return iou


def read_ground_truth(file_path):
    with open(file_path, 'r', encoding="utf8") as file:
        unclean_list = file.readline().split(" ")
        return [float(x) for x in unclean_list[1:]]


class VJ:
    Data = None
    detectionList = []
    UvIScore = 0.0
    meanScoreUvI = 0.0
    FPScore = 0.0
    FNscore = 0.0
    scaleFactor = 1.05
    minNeighbors = 5

    def __init__(self, Data, scaleFactor, minNeighbors):
        self.Data = Data
        self.UvIScore = 0.0
        self.meanScoreUvI = 0.0
        self.detected_objects = 0
        self.num_true_pos = 0
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors

    def detect_right_ear(self, img):
        detectionList = right_ear_cascade.detectMultiScale(img, self.scaleFactor, self.minNeighbors)
        return detectionList

    def detect_left_ear(self, img):
        detectionList = left_ear_cascade.detectMultiScale(img, self.scaleFactor, self.minNeighbors)
        return detectionList

    def train(self):
        i = 0
        for img_path in self.Data.train_data:
            print(f"Train: {img_path}")

            img = cv2.imread(img_path)

            left_detected = self.detect_left_ear(img)

            ground_truth = read_ground_truth(self.Data.truth_train_data[i])

            height = img.shape[0]
            width = img.shape[1]

            ground_truth[2] *= width

            ground_truth[3] *= height

            ground_truth[0] *= width

            ground_truth[0] -= (ground_truth[2] / 2)

            ground_truth[1] *= height

            ground_truth[1] -= (ground_truth[3] / 2)

            img_num = int((img_path.split("\\")[1]).split(".")[0])

            for rect_stat in left_detected:
                score = calc_IoU_score(rect_stat, ground_truth)
                print(f"IoU score is {score}")
                vizualization(img, rect_stat, img_num)
                self.meanScoreUvI += score
                if score <= 0.4:
                    self.FPScore += 1
                else:
                    self.Data.train_data_cropped.append((img_num, vizualization(img, rect_stat, img_num)))

            right_detected = self.detect_right_ear(img)

            for rect_stat in right_detected:
                score = calc_IoU_score(rect_stat, ground_truth)
                print(f"IoU score is {score}")

                self.meanScoreUvI += score
                if score <= 0.5:
                    self.FPScore += 1
                else:
                    self.Data.train_data_cropped.append((img_num, vizualization(img, rect_stat, img_num)))

            if len(right_detected) == 0 and len(left_detected) == 0:
                self.FNscore += 1

            self.detected_objects += len(left_detected)
            self.detected_objects += len(right_detected)
            i += 1

    def test(self):
        i = 0
        for img_path in self.Data.test_data:
            print(f"Test: {img_path}")

            img = cv2.imread(img_path)

            left_detected = self.detect_left_ear(img)

            ground_truth = read_ground_truth(self.Data.truth_test_data[i])

            height = img.shape[0]
            width = img.shape[1]

            ground_truth[2] *= width

            ground_truth[3] *= height

            ground_truth[0] *= width

            ground_truth[0] -= (ground_truth[2] / 2)

            ground_truth[1] *= height

            ground_truth[1] -= (ground_truth[3] / 2)

            img_num = int((img_path.split("\\")[1]).split(".")[0])

            for rect_stat in left_detected:
                score = calc_IoU_score(rect_stat, ground_truth)
                print(f"IoU score is {score}")
                vizualization(img, rect_stat, img_num)
                self.meanScoreUvI += score
                if score <= 0.4:
                    self.FPScore += 1
                else:
                    self.Data.test_data_cropped.append((img_num, vizualization(img, rect_stat, img_num)))

            right_detected = self.detect_right_ear(img)

            for rect_stat in right_detected:
                score = calc_IoU_score(rect_stat, ground_truth)
                print(f"IoU score is {score}")

                self.meanScoreUvI += score
                if score <= 0.45:
                    self.FPScore += 1
                else:
                    self.Data.test_data_cropped.append((img_num, vizualization(img, rect_stat, img_num)))

            if len(right_detected) == 0 and len(left_detected) == 0:
                self.FNscore += 1

            self.detected_objects += len(left_detected)
            self.detected_objects += len(right_detected)
            i += 1

    def detect(self):
        self.train()
        print(
            f"TRAIN: IoU score:{self.meanScoreUvI / self.detected_objects}, False positives:{self.FPScore}, False "
            f"negative:{self.FNscore}")

        self.meanScoreUvI = 0.0
        self.num_true_pos = 0.0
        self.FPScore = 0.0
        self.FNscore = 0.0
        self.detected_objects = 0

        self.test()
        print(
            f"TEST: IoU score:{self.meanScoreUvI / self.detected_objects}, False positive rate:{self.FPScore}, False "
            f"negative:{self.FNscore}")


def cos_dist(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


class LocalBinaryPattren:
    Data = None
    feature_vectors = {}
    R = 1
    P = 4

    def __init__(self, Data, R, P):
        self.Data = Data
        self.R = R
        self.P = P
        self.feature_vectors = {}

    def calcPixleValue(self, x, y):
        pass

    def train(self):
        max_len = 0
        for img_num, image in self.Data.train_data_cropped:
            feature_vector = []
            feature_matrix = image.copy()
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    feature_matrix[y][x] = self.calc_pix_value(image, x, y)

            self.feature_vectors[img_num] = feature_matrix.flatten()
            max_len = max(max_len, len(feature_matrix.flatten()))
            # feature_vector = self.make_histogram(np.ravel(feature_matrix))

        self.padd_vectors(max_len)
        self.classify()

    def classify(self):
        predicted_class = {}
        for i in self.feature_vectors.keys():
            feature1 = self.feature_vectors[i]
            distance_scores = []
            for j in self.feature_vectors.keys():
                feature2 = self.feature_vectors[j]
                sim_score_cos = cos_dist(feature1, feature2)
                if 0.5 < sim_score_cos < 0.98:
                    print(f"{i} and {j} are same class (cos score: {sim_score_cos})")
                    distance_scores.append((sim_score_cos, j))

            scores_sorted = sorted(distance_scores)
            predicted_class[i] = self.Data.class_by_img[(scores_sorted[0])[1]]

        accuracy = 0.0
        TP = 0
        total_len = 0
        for i in predicted_class:
            total_len += 1
            if predicted_class[i] == self.Data.class_by_img[i]:
                TP += 1

        print(f"Accuracy score is {(TP / total_len)}")

    def make_histogram(self, vector):
        num_of_bins = self.P * (self.P - 1) + 2
        hist = np.zeros(num_of_bins, dtype=np.int)
        for i in vector:
            hist[i] += 1

    def calc_pix_value(self, img, x_c_cord, y_c_cord):

        bit_array = []
        for p in range(self.P):
            x_cord = round(-self.R * math.sin((2 * math.pi * p) / self.P) + x_c_cord)
            y_cord = round(self.R * math.cos((2 * math.pi * p) / self.P) + y_c_cord)

            c_value = img[y_c_cord][x_c_cord]

            x = x_cord
            y = y_cord
            if (0 <= x < img.shape[1]) and (0 <= y < img.shape[0]):
                bit = 1 if img[y][x] > c_value else 0
                bit_array.append(bit)
            else:
                bit_array.append(0)

        while (bit_array[0] != 0) and (0 in bit_array):
            a = bit_array[0]
            bit_array.pop(0)
            bit_array.extend([a])

        pix_value = 0
        for i in range(0, len(bit_array)):
            pix_value += bit_array[i] * (2 ** i)

        return pix_value

    def padd_vectors(self, max_len):
        for i in self.feature_vectors.keys():
            result = np.zeros(max_len)
            result[:self.feature_vectors[i].shape[0]] = self.feature_vectors[i]
            self.feature_vectors[i] = result


class LocalBinaryPattren_Skimage:
    Data = None
    feature_vectors = {}
    R = 1
    P = 4
    skimage_mode = "default"

    def __init__(self, Data, R, P, skimage_mode):
        self.Data = Data
        self.R = R
        self.P = P
        self.feature_vectors = {}
        self.skimage_mode = skimage_mode

    def calcPixleValue(self, x, y):
        pass

    def train(self):
        max_len = 0
        for img_num, image in self.Data.train_data_cropped:
            feature_vector = []
            feature_matrix = image.copy()
            lbp = feature.local_binary_pattern(image, self.P, self.R, method=self.skimage_mode)
            self.feature_vectors[img_num] = feature_matrix.flatten()
            max_len = max(max_len, len(feature_matrix.flatten()))

        self.padd_vectors(max_len)
        self.classify()

    def classify(self):
        predicted_class = {}
        for i in self.feature_vectors.keys():
            feature1 = self.feature_vectors[i]
            distance_scores = []
            for j in self.feature_vectors.keys():
                feature2 = self.feature_vectors[j]
                sim_score_cos = cos_dist(feature1, feature2)
                if 0.5 < sim_score_cos < 0.98:
                    print(f"{i} and {j} are same class (cos score: {sim_score_cos})")
                    distance_scores.append((sim_score_cos, j))

            scores_sorted = sorted(distance_scores)
            predicted_class[i] = self.Data.class_by_img[(scores_sorted[0])[1]]

        accuracy = 0.0
        TP = 0
        total_len = 0
        for i in predicted_class:
            total_len += 1
            if predicted_class[i] == self.Data.class_by_img[i]:
                TP += 1

        print(f"Accuracy score is {(TP / total_len)}")

    def padd_vectors(self, max_len):
        for i in self.feature_vectors.keys():
            result = np.zeros(max_len)
            result[:self.feature_vectors[i].shape[0]] = self.feature_vectors[i]
            self.feature_vectors[i] = result


class Pixel_wise:
    Data = None
    feature_vectors = {}

    def __init__(self, Data):
        self.Data = Data
        self.feature_vectors = {}

    def train(self):
        max_len = 0
        for img_num, image in self.Data.train_data_cropped:
            feature_vector = []
            feature_matrix = image.copy()
            self.feature_vectors[img_num] = feature_matrix.flatten()
            max_len = max(max_len, len(feature_matrix.flatten()))

        self.padd_vectors(max_len)
        self.classify()

    def classify(self):
        predicted_class = {}
        for i in self.feature_vectors.keys():
            feature1 = self.feature_vectors[i]
            distance_scores = []
            for j in self.feature_vectors.keys():
                feature2 = self.feature_vectors[j]
                feature_diff = feature1 - feature2
                num_of_zeros = np.count_nonzero(feature_diff == 0)
                if 0.5 < num_of_zeros / len(feature1) < 0.98:
                    print(f"{i} and {j} are same class (score: {num_of_zeros / len(feature1)})")
                    distance_scores.append((num_of_zeros / len(feature1), j))

            scores_sorted = sorted(distance_scores)
            if len(distance_scores) == 0:
                predicted_class[i] = -1
            else:
                predicted_class[i] = self.Data.class_by_img[(scores_sorted[0])[1]]

        accuracy = 0.0
        TP = 0
        total_len = 0
        for i in predicted_class:
            total_len += 1
            if predicted_class[i] == self.Data.class_by_img[i]:
                TP += 1

        print(f"Accuracy score is {(TP / total_len)}")

    def padd_vectors(self, max_len):
        for i in self.feature_vectors.keys():
            result = np.zeros(max_len)
            result[:self.feature_vectors[i].shape[0]] = self.feature_vectors[i]
            self.feature_vectors[i] = result
