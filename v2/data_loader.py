import numpy as np
import os
import tensorflow as tf
dataset_path = "F:/gis/sidv2/data/train"

# path = os.path.join(dataset_path, "London_2017_5.npy")

# data = np.load(path)
# label = data[5, :, :]
# label = np.reshape(label, (1,1024,1024))
# #swap evi with the UI channel, and slice it off
# data[5,:,:] = data[6,:,:]
# features = data[:6,:,:]

# features = tf.reshape(features, [1024,1024,6])
# label = tf.reshape(label, [1024,1024,1])

def parse_data(path , calibrate_param:float = 0.4):
    data = np.load(path)
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    label = data[5, :, :]
    label = np.reshape(label, (1,1024,1024))

    features = np.delete(data, 5, 0) # cut off the UI channel

    #encode NAs to 0s
    features = np.nan_to_num(features)
    label = np.nan_to_num(label)

    features = tf.reshape(features, [1024,1024,6])
    label = tf.reshape(label, [1024,1024,1])

    RGB_calibrated = tf.map_fn(lambda x: tf.math.minimum(x,calibrate_param), features[:,:,0:3])
    features = tf.concat([RGB_calibrated, features[:,:,3:]], axis=2)
    return (features, label) # TODO : consider mapping from tuple to dict? or does tuple work?

training_dataset = tf.data.Dataset.list_files("F:/gis/sidv2/data/train/*", seed=321, shuffle=False)
#training_dataset = training_dataset.map(lambda x: parse_data(x, 0.4))
#training_dataset = training_dataset.map(lambda x: tf.py_function(parse_data, [x,0.4], (tf.float32, tf.float32)))

training_dataset = training_dataset.map(lambda i: tf.numpy_function(func=parse_data, 
                                               inp=[i,0.4], 
                                               Tout=(tf.float32, tf.float32)
                                               ), 
                      num_parallel_calls=tf.data.AUTOTUNE)

for i in training_dataset.take(1):
    print(i[0].numpy().shape)

# features, label = calibrate(features, label, 0.4)

# train_dataset = tf.data.Dataset.from_tensor_slices((features, label))

###############################

# tf.reduce_max(tensor_without_nans)