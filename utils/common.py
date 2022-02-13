import os
import glob
import pathlib
from pathlib import Path
import shutil
import tensorflow as tf


def clean_images(path):
    files = glob.glob(path + '/*.png')
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))


def random_batch_getter(dataset: tf.Tensor, batch_size):
    """
    help get infinite batches by looping the dataset
    """
    ds = tf.random.shuffle(dataset)
    n_batch = dataset.shape[0] // batch_size
    while True:
        for batch_no in range(n_batch):
            yield ds[(t := batch_no * batch_size):t + batch_size]
        ds = tf.random.shuffle(ds)
    pass


def empty_directory(path):
    if not isinstance(path, Path):
        path = Path(path)

    for f in path.glob('*'):
        if f.is_file():
            f.unlink()
        else:
            shutil.rmtree(f)
    return
