import cv2, sys, glob

left_ear_cascade = cv2.CascadeClassifier("./Data/haarcascade_mcs_leftear.xml")
right_ear_cascade = cv2.CascadeClassifier("./Data/haarcascade_mcs_rightear.xml")


def vizualization(img, detectionList,ground_truth, i):
    y = int(ground_truth[0]*1000)
    x = int(ground_truth[1]*1000)
    w = int(ground_truth[2]*1000)
    h = int(ground_truth[3]*1000)

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 165, 0), 4)

    for x, y, w, h in detectionList:
        cv2.rectangle(img, (x, y), (x + w, y + h), (128, 255, 0), 4)

    cv2.imwrite("./VJ_Images/" + str(i) + '_detected.jpg', img)


def calc_IoU_score(rect_stat, ground_truth):
    x1, y1, w1, h1 = rect_stat
    x2, y2, w2, h2 = ground_truth

    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)

    intersection_area = w_intersection * h_intersection

    area_box1 = w1 * h1
    area_box2 = w2 * h2

    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area / union_area

    return iou

def read_ground_truth(file_path):

    with open(file_path, 'r', encoding="utf8") as file:
        unclean_list = file.readline().split(" ")
        return [float(x) for x in unclean_list[1:]]

class VJ:
    Data = None
    detectionList = []
    UvIScore = 0.0
    scaleFactor = 1.05
    minNeighbors = 5

    def __init__(self, Data, scaleFactor, minNeighbors):
        self.Data = Data
        self.UvIScore = 0.0
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

            rect_stat = self.detect_left_ear(img)

            ground_truth = read_ground_truth(self.Data.truth_train_data[i])

            if len(rect_stat) != 0:
                decimal_stat = [x * 0.001 for x in rect_stat]
                self.detectionList.append(decimal_stat)
                score = calc_IoU_score(decimal_stat[0], ground_truth)
                vizualization(img,rect_stat,ground_truth,i)
                print(f"IoU score is {score}")

            rect_stat = self.detect_right_ear(img)

            if len(rect_stat) != 0:
                decimal_stat = [x * 0.0001 for x in rect_stat]
                self.detectionList.append([x * 0.001 for x in rect_stat])
                vizualization(img, rect_stat, ground_truth, i)
                score = calc_IoU_score(decimal_stat[0], ground_truth)
                print(f"IoU score is {score}")

            i+=1

    def test(self):
        i = 0
        for img_path in self.Data.test_data:
            print(f"Test: {img_path}")
            img = cv2.imread(img_path)

            self.detect_left_ear(img)
            self.detect_right_ear(img)

    def evaluate(self):
        self.train()
        self.test()

# detectionList = detectFace(img)
# vizualization(img, detectionList)
