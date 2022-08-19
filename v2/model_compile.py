import os
import math
from keras import backend as K

main_path = "F:/gis/sidv2/"

checkpoint_dir = main_path+"/weights"

# Sourcing scripts
exec(open(os.path.join(main_path + "/tf_setup.py")).read())
exec(open(os.path.join(main_path + "/data_loader.py")).read())

# Load in the data
train_dataset = construct_dataset(os.path.join(main_path, "data", "train", "*"))
test_dataset = construct_dataset(os.path.join(main_path, "data", "test", "*"))
dev_dataset = construct_dataset(os.path.join(main_path, "data", "dev", "*"))

BUFFER_SIZE = 1000
BATCH_SIZE = 32

dataset = {"train": train_dataset, "test": test_dataset, "dev":dev_dataset}

dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
dataset['train'] = dataset['train'].repeat()
dataset['train'] = dataset['train'].batch(BATCH_SIZE)
dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

#-- Test Dataset --#
#dataset['test'] = dataset['test'].repeat()
dataset['test'] = dataset['test'].batch(BATCH_SIZE)
dataset['test'] = dataset['test'].prefetch(buffer_size=AUTOTUNE)

#-- Dev Dataset --#
#dataset['test'] = dataset['test'].repeat()
dataset['dev'] = dataset['dev'].batch(BATCH_SIZE)
dataset['dev'] = dataset['dev'].prefetch(buffer_size=AUTOTUNE)

# Model Compilation Setup

dropout_rate = 0.5
input_size = (IMG_SIZE, IMG_SIZE, N_CHANNELS)
initializer = 'he_normal'

# Model Definition

# -- Encoder -- #
# Block encoder 1
inputs = Input(shape=input_size)
conv_enc_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
conv_enc_1 = Conv2D(64, 3, activation = 'relu', padding='same', kernel_initializer=initializer)(conv_enc_1)

# Block encoder 2
max_pool_enc_2 = MaxPooling2D(pool_size=(2, 2))(conv_enc_1)
conv_enc_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_2)
conv_enc_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_2)

# Block  encoder 3
max_pool_enc_3 = MaxPooling2D(pool_size=(2, 2))(conv_enc_2)
conv_enc_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_3)
conv_enc_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_3)

# Block  encoder 4
max_pool_enc_4 = MaxPooling2D(pool_size=(2, 2))(conv_enc_3)
conv_enc_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_4)
conv_enc_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_4)
# -- Encoder -- #

# ----------- #
maxpool = MaxPooling2D(pool_size=(2, 2))(conv_enc_4)
conv = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(maxpool)
conv = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv)
# ----------- #

# -- Decoder -- #
# Block decoder 1
up_dec_1 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv))
merge_dec_1 = concatenate([conv_enc_4, up_dec_1], axis = 3)
conv_dec_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_1)
conv_dec_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_1)

# Block decoder 2
up_dec_2 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_1))
merge_dec_2 = concatenate([conv_enc_3, up_dec_2], axis = 3)
conv_dec_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_2)
conv_dec_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_2)

# Block decoder 3
up_dec_3 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_2))
merge_dec_3 = concatenate([conv_enc_2, up_dec_3], axis = 3)
conv_dec_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_3)
conv_dec_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_3)

# Block decoder 4
up_dec_4 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_3))
merge_dec_4 = concatenate([conv_enc_1, up_dec_4], axis = 3)
conv_dec_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_4)
conv_dec_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
conv_dec_4 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
# -- Dencoder -- #

# output = Conv2D(N_CLASSES, 1, activation = 'softmax')(conv_dec_4) - only valid for 2-class segmentation problem
output = Conv2D(1,1, activation="sigmoid")(conv_dec_4)

# Weight Decay "Schedule" - can be replaced with the learning_rate parameter.
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)

#MeanIou For use with the Sparse Categorical Cross Entrophy 
# class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
#   def __init__(self,
#                y_true=None,
#                y_pred=None,
#                num_classes=None,
#                name=None,
#                dtype=None):
#     super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

#   def update_state(self, y_true, y_pred, sample_weight=None):
#     y_pred = tf.math.argmax(y_pred, axis=-1)
#     return super().update_state(y_true, y_pred, sample_weight)

# RMSE for continuous evaluation
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

model = tf.keras.Model(inputs = inputs, outputs = output)
model.compile(optimizer=Adam(learning_rate=lr_schedule), loss = root_mean_squared_error,
              metrics=['MeanSquaredError'])

# Mini-batching settings ######################

STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE

# Using #dataset.repeat to randomly sample some 50 epochs and train on them.
EPOCHS = 30

VALIDATION_STEPS = DEVSET_SIZE // BATCH_SIZE

print(model.summary())