from glob import glob

import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime, os
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from IPython.display import clear_output


AUTOTUNE = tf.data.experimental.AUTOTUNE
print(f"Tensorflow ver. {tf.__version__}")

SEED = 42

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True) # dynamic GPU memory growth
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Image size that we are going to use
IMG_SIZE = 1024
# Number of channels (we have 7-1 (-1 of the dv channel))
N_CHANNELS = 6
# Scene Parsing
#N_CLASSES = 2

BUFFER_SIZE = 1000
BATCH_SIZE = 4

TRAINSET_SIZE = len(glob(main_path + "data/train/*.npy"))
print(f"The Training Dataset contains {TRAINSET_SIZE} images.")

TESTSET_SIZE = len(glob(main_path + "data/test/*.npy"))
print(f"The Testing Dataset contains {TESTSET_SIZE} images.")

DEVSET_SIZE = len(glob(main_path + "data/dev/*.npy"))
print(f"The Dev Dataset contains {DEVSET_SIZE} images.")

VALIDATION_STEPS = DEVSET_SIZE // BATCH_SIZE