from PIL import Image
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import pandas as pd

def main():
    im = Image.open(r'C:\users\galun\Desktop\Ran\Dor_Assignments_C\Project_Dor\fishpool-another-ex1.bmp', 'r')
    img = np.asarray(im)
    imgBool = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i, j, 0] == 155) and (img[i, j, 1] == 190) and (img[i, j, 2] == 245):
                imgBool[i, j] = 1
            else:
                imgBool[i, j] = 0
    pd.DataFrame(imgBool).to_csv("./image.csv", sep=',', float_format='%d')
    labeled, nr_objects = ndimage.label(imgBool)
    # labeled = np.flip(labeled, axis=1)
    print(nr_objects)
    for i in range(labeled.shape[0]):
        for j in range(labeled.shape[1]):
            if labeled[i, j] != 0:
                print(i, j, labeled[i, j])
    width, height = im.size
    print(width, height)
    plt.imshow(labeled)
    plt.show()


if __name__ == "__main__":
    main()
