import numpy as np
import os
import tensorflow as tf
from skimage.transform import resize

def parse_data(path , rotation:bool = False, augmentation_prop:float = 0.5, should_resize:bool= True, IMG_SIZE:int = 256):
    data = np.load(path)
    data = np.nan_to_num(data)

    # New normalizer

    data_min = data.min(axis=(1,2), keepdims=True) # dont divide by 0
    _recode_0 = np.vectorize(lambda x: max(x,0.0001))
    data_min = _recode_0(data_min)
    data_max = data.max(axis=(1,2),keepdims=True)
    data = (data - data_min)/(data_max - data_min)

    # EXPR - resize using skimage.
    if should_resize:
        small = np.zeros((7,IMG_SIZE,IMG_SIZE))
        for dim in range(0,7):
            small[dim,:,:] = resize(data[dim,:,:], (IMG_SIZE,IMG_SIZE))
        data = tf.convert_to_tensor(small, dtype=tf.float32)
        del small # save memory

    # Preprocessing steps for all layers:

    #Random rotation
    if rotation:
        if(np.random.uniform(0,1) > augmentation_prop):
            data = np.rot90(data, axes=(1,2))

    label = data[5, :, :]
    label = np.reshape(label, (1,IMG_SIZE,IMG_SIZE))

    features = np.delete(data, 5, 0) # cut off the UI channel


    # conver to tf (now immutable)

    features = tf.reshape(features, [IMG_SIZE,IMG_SIZE ,6])
    #features = tf.cast(features, tf.float32)
    label = tf.reshape(label, [IMG_SIZE,IMG_SIZE,1])
    #label = tf.cast(label, tf.float32)

    #RGB_calibrated = tf.map_fn(lambda x: tf.math.minimum(x,calibrate_param), features[:,:,0:3])
    #RGB_calibrated = tf.cast(RGB_calibrated, tf.float32) / calibrate_param # normalize the RGB features to vary between [0,1]
    # out = tf.experimental.numpy.empty([IMG_SIZE,IMG_SIZE,1]) # TODO: reconstruct the features tensor (possibility to be optimized?)
    # out = tf.cast(out, tf.float32)
    # for i in range(0,6):
    #     norm_param = tf.math.reduce_max(features[:,:,i])
    #     normalized = features[:,:,i]/norm_param
    #     normalized = tf.reshape(normalized, [IMG_SIZE,IMG_SIZE,1])
    #     normalized = tf.cast(normalized, tf.float32)
    #     out = tf.concat([out, normalized], axis=2)

    features = np.nan_to_num(features)
    label = np.nan_to_num(label)

    features = tf.cast(features, tf.float32)
    label = tf.cast(label, tf.float32)

    return (features, label)
            
def set_shapes(img, label, IMG_SIZE:int = 256):
    img.set_shape([IMG_SIZE,IMG_SIZE,6])
    label.set_shape([IMG_SIZE,IMG_SIZE,1])
    return img, label


def construct_dataset(path:str, IMG_SIZE:int = 256):
    dataset = tf.data.Dataset.list_files(path, seed=321, shuffle=False)

    dataset = dataset.map(lambda i: tf.numpy_function(func=parse_data, 
                                                inp=[i,True, 0.5, True,IMG_SIZE], # parameters for parse_data()
                                                # parse_data(path , rotation:bool = False, augmentation_prop:float = 0.5, resize:bool= True, IMG_SIZE:int = 256):
                                                Tout=(tf.float32, tf.float32)
                                                ), 
                        num_parallel_calls=tf.data.AUTOTUNE)

    # ensure shape goes here
    dataset = dataset.map(lambda img,label: set_shapes(img,label))

    return dataset

def unit_test_dataloader(dataset, IMG_SIZE:int = 256, N_CHANNELS:int = 6):
    for i in dataset.take(1):
        assert i[0].shape == (IMG_SIZE, IMG_SIZE, N_CHANNELS), f"Features have incorrect shape: {i[0].shape}; Expected: {(IMG_SIZE,IMG_SIZE, N_CHANNELS)}"
        assert i[1].shape == (IMG_SIZE,IMG_SIZE,1), f"Label has incorrect shape: {i[1].shape}; Expected: {(IMG_SIZE,IMG_SIZE,1)}"
    print("SUCCESS: Passed Unit Test for Data Loader")


# Testing purposes only:

# for case in training_dataset.take(1):
#     for dim in range(0,6):
#         print(f"Dimension {dim} Mean: {np.mean(case[0].numpy()[:,:,dim])}")
#         print(f"Dimension {dim} 90th percentile: {np.percentile(case[0].numpy()[:,:,dim], 90)}")
#         print(f"Dimension {dim} 10th percentile: {np.percentile(case[0].numpy()[:,:,dim], 10)}")
#         print("---")