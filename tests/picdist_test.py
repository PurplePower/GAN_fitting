from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from utils.picdist import image2dots

if __name__ == '__main__':
    im = Image.open('../pics/test.png')
    im = im.resize((64, 64))

    x = image2dots(im, 1024)

    plt.scatter(x[:, 0], x[:, 1])
    plt.show()

    pass
