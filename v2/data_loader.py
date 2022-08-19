import numpy as np
import os
import tensorflow as tf

def parse_data(path , calibrate_param:float = 0.4, rotation:bool = False, augmentation_prop:float = 0.0):
    data = np.load(path)
    data = tf.convert_to_tensor(data, dtype=tf.float32)

    # Preprocessing steps for all layers:

    label = data[5, :, :]
    label = np.reshape(label, (1,1024,1024))

    features = np.delete(data, 5, 0) # cut off the UI channel

    #encode NAs to 0s
    features = np.nan_to_num(features)
    label = np.nan_to_num(label)

    #Random rotation
    if rotation:
        if(np.random.uniform(0,1) > augmentation_prop):
            features = np.apply_along_axis(np.rot90, 0, features)
            label = np.rot90(label)

    # conver to tf (now immutable)

    features = tf.reshape(features, [1024,1024,6])
    label = tf.reshape(label, [1024,1024,1])

    RGB_calibrated = tf.map_fn(lambda x: tf.math.minimum(x,calibrate_param), features[:,:,0:3])
    RGB_calibrated = tf.cast(RGB_calibrated, tf.float32) / calibrate_param # normalize the RGB features to vary between [0,1]
    out = RGB_calibrated # TODO: reconstruct the features tensor (possibility to be optimized?)

    for i in range(3,6):
        norm_param = tf.math.reduce_max(features[:,:,i])
        normalized = features[:,:,i]/norm_param
        normalized = tf.reshape(normalized, [1024,1024,1])
        out = tf.concat([out, normalized], axis=2)

    # out = features.set_shape((1024,1024,6))
    # label = label.set_shape((1024,1024))

    return (out, label)

def set_shapes(img, label):
    img.set_shape([1024,1024,6])
    label.set_shape([1024,1024,1])
    return img, label


def construct_dataset(path:str):
    dataset = tf.data.Dataset.list_files(path, seed=321, shuffle=False)

    dataset = dataset.map(lambda i: tf.numpy_function(func=parse_data, 
                                                inp=[i,0.4], 
                                                Tout=(tf.float32, tf.float32)
                                                ), 
                        num_parallel_calls=tf.data.AUTOTUNE)

    # ensure shape goes here
    dataset = dataset.map(lambda img,label: set_shapes(img,label))

    return dataset


# Testing purposes only:

# for case in training_dataset.take(1):
#     for dim in range(0,6):
#         print(f"Dimension {dim} Mean: {np.mean(case[0].numpy()[:,:,dim])}")
#         print(f"Dimension {dim} 90th percentile: {np.percentile(case[0].numpy()[:,:,dim], 90)}")
#         print(f"Dimension {dim} 10th percentile: {np.percentile(case[0].numpy()[:,:,dim], 10)}")
#         print("---")