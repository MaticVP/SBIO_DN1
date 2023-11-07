from Detectors import VJ
from Data import Data

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image_data = Data(0.8,"./Data/ears/")
    EarDetector = VJ(image_data,1.05,5)
    EarDetector.evaluate()
