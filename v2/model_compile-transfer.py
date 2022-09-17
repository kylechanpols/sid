# A transfer learning application for our problem. We attach a top layer mapping the 6 channels of the features into a 3-channel
# feature map, then pass these feature maps through a pre-trained Efficient Net B0. With a custom decoder layer, I turn
# the Efficient Net B0 output to a regression output of the urban index of the image.

import os
from pickletools import optimize
from tabnanny import check
from keras import backend as K
import tensorflow
from tensorflow.keras.applications.efficientnet import EfficientNetB0
import tensorflow_addons as tfa

main_path =os.getcwd()

checkpoint_dir = os.path.join(main_path,"weights")

# Sourcing scripts
exec(open(os.path.join(main_path, "tf_setup.py")).read())
exec(open(os.path.join(main_path,  "data_loader.py")).read())

# Load in the data
train_dataset = construct_dataset(os.path.join(main_path, "data", "train", "*"), IMG_SIZE=IMG_SIZE, single_tgt=True)
test_dataset = construct_dataset(os.path.join(main_path, "data", "test", "*"), IMG_SIZE=IMG_SIZE, single_tgt=True)
dev_dataset = construct_dataset(os.path.join(main_path, "data", "dev", "*"), IMG_SIZE=IMG_SIZE, single_tgt=True)

# Call the unit tester to see if the returned images are of the correct dims
unit_test_dataloader(train_dataset,IMG_SIZE=IMG_SIZE, N_CHANNELS=N_CHANNELS, single_tgt=True)

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

# RMSE for continuous evaluation
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

# Model Architecture setup
input_size = (IMG_SIZE, IMG_SIZE,N_CHANNELS)
inputs = Input(shape=input_size)
dropout_rate = 0.5
initializer = 'he_normal'

#custom top
decreasing = Conv2D(3, (1,1), activation='relu', padding='same', kernel_initializer=initializer)(inputs) # this will downsize the 6 dims image back to 3 dims as a feature map

# transfer learning to a EfficientNetB0
model = EfficientNetB0(include_top=False,
 input_shape=(IMG_SIZE, IMG_SIZE, 3),
 weights=os.path.join(checkpoint_dir,"efficientnetb0_notop.h5"))
 # weights="imagenet") # using alternative weights from Google

model.trainable = False # freeze all layers of Efficient Net B0

bottom_input = model(decreasing)

# custom bottom to turn efficient net b0 output to regression

bottom_pool = GlobalAveragePooling2D(name="avg_pool")(bottom_input)
bottom_batchnorm = BatchNormalization()(bottom_pool)
bottom_dropout = Dropout(dropout_rate)(bottom_batchnorm)
bottom_output = Dense(1, name="pred")(bottom_dropout)

model = tf.keras.Model(inputs,bottom_output)

# Model Compilation

# A lookahead optimizer with rectified Adam
radam = tfa.optimizers.RectifiedAdam(learning_rate=0.5)
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
optimizer = ranger


model.compile(optimizer=optimizer, loss=root_mean_squared_error)

# Mini-batching settings

STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
print(f"Steps per epoch: {STEPS_PER_EPOCH}")

# Using #dataset.repeat to randomly sample some 50 epochs and train on them.
EPOCHS = 40

VALIDATION_STEPS = DEVSET_SIZE // BATCH_SIZE
print(f"Val steps: {VALIDATION_STEPS}")
print(model.summary())

