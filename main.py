from Detectors import VJ, LocalBinaryPattren, LocalBinaryPattren_Skimage, Pixel_wise
from Data import Data

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image_data = Data(0.8, "./Data/ears/", "./Data/identites.txt")
    EarDetector = VJ(image_data, 1.08, 5)
    EarDetector.detect()

    LocalBinaryPattren = LocalBinaryPattren(EarDetector.Data, 1, 8)
    LocalBinaryPattrenSkimage = LocalBinaryPattren_Skimage(EarDetector.Data, 1, 8, "default")
    PixelWise = Pixel_wise(EarDetector.Data)

    print()
    print("Pixel Wise")
    PixelWise.train()
    print("my LBP")
    LocalBinaryPattren.train()
    print("LBP Skimage")
    LocalBinaryPattrenSkimage.train()

