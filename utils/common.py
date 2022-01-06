import os
import glob


def clean_images(path):
    files = glob.glob(path + '/*.png')
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))