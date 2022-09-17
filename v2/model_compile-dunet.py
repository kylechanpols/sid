# An adapation of the D-Unet (Zhou et. al. 2019 , https://arxiv.org/abs/1908.05104) with a custom last layer to 
# generate pixel-by-pixel regression prediction of the urban index of an image.

import os
from pickletools import optimize
from keras import backend as K

main_path =os.getcwd()

checkpoint_dir = os.path.join(main_path,"weights")

# Sourcing scripts
exec(open(os.path.join(main_path, "tf_setup.py")).read())
exec(open(os.path.join(main_path,  "data_loader.py")).read())

# Load in the data
train_dataset = construct_dataset(os.path.join(main_path, "data", "train", "*"), IMG_SIZE=IMG_SIZE)
test_dataset = construct_dataset(os.path.join(main_path, "data", "test", "*"), IMG_SIZE=IMG_SIZE)
dev_dataset = construct_dataset(os.path.join(main_path, "data", "dev", "*"), IMG_SIZE=IMG_SIZE)

# Call the unit tester to see if the returned images are of the correct dims
unit_test_dataloader(train_dataset,IMG_SIZE=IMG_SIZE, N_CHANNELS=N_CHANNELS)

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

exec(open(os.path.join(main_path, "model_candidates", "D-Unet.py")).read())
model = D_Unet()
model.compile(optimizer = SGD(learning_rate=1e-6),
    loss=root_mean_squared_error)

STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
print(f"Steps per epoch: {STEPS_PER_EPOCH}")

# Using #dataset.repeat to randomly sample some 50 epochs and train on them.
EPOCHS = 20

VALIDATION_STEPS = DEVSET_SIZE // BATCH_SIZE
print(f"Val steps: {VALIDATION_STEPS}")
print(model.summary())