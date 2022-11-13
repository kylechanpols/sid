import numpy as np
import os
import tensorflow as tf
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def parse_data(path ,
 rotation:bool = False,
  augmentation_prop:float = 0.5,
   should_resize:bool= True,
    IMG_SIZE:int = 256,
    dev:bool = True,
    single_tgt:bool = False,
    calibrate_rgb:float= 0.4,
    calibrate_rs:float= 0.1,
    rgb:bool= False):
    '''
    Generic data parser for Google Earth Engine Satellite Images (7-channels) to extract and transform relevant features and label from source image
    args:
    path (str) - a path to the source image, must be .npy files
    rotation (bool) - Shall we perform random rotation?
    augmentation_prop (float) - Probability at which the image augmentation technique is applied
    should_resize (bool) - Shall we resize the source image? Note that it's always recommended to downsize the source image to simplify the complexity of the learning task.
    IMG_SIZE (int) - Size of the image to be resized to. Currently the parser only support square images, so the output image will be turned to an array of IMG-SIZE * IMG_SIZE.
    dev (bool) - Flag to run the parser in development mode. In development mode, image augmentation tasks are applied to the image. In prediction mode (dev= False), no augmentation steps are applied to the image.
    single_tgt (bool) - Flag to generate single target ouput. If set to True, an extra reduce sum along the image width and image height axes will be applied to the image for use in the Efficient Net B0 model. If set to False,
                        the parser will return the output image as is.
    rgb(bool) - Should we return only RGB? Will return only the RGB layers if set to True.
    output:
        A tuple (feature, label). Feature will be of dimension (N_EXAMPLE, IMG_SIZE, IMG_SIZE, N_CHANNELS).
        The label dimension is (N_EXAMPLE, IMG_SIZE, IMG_SIZE, 1) if single_tgt is set to False, and (N_EXAMPLE, 1, 1) if single_tgt is set to True.
    '''
    data = np.load(path)

    # NA Encoding
    # rgb, na code to 0
    data[:3, : ,:] = np.nan_to_num(data[:3, : ,:], nan=0) #na is out of bounds pixel, and thus recoded as 0

    # rs, na code to their respective layer minimum
    # but do note that all nan slice could exist, and those require encoding:

    for dim in range(3, data.shape[0]):
        if np.isnan(data[dim, : ,:]).sum() == data.shape[1]*data.shape[2]:
            slice_min = -10
        else:
            slice_min = np.nanmin(data[dim, : ,:])
        data[dim, : ,:] = np.nan_to_num(data[dim, : ,:], nan=slice_min) #na is out of bounds pixel, and thus recoded as 0

    # data = np.nan_to_num(data, nan=0) #na is out of bounds pixel, and thus recoded as 0

    # Calibration
    # We normalize the channels as follows: RGB, divide by rgb calibration factor
    data[:3,:,:] = data[:3,:,:]/calibrate_rgb
    data[3:,:,:] = data[3:,:,:]/calibrate_rs

    #then clip these pixels
    data[:3,:,:] = np.clip(data[:3,:,:], 0, 255)
    #data[3:,:,:] = np.clip(datadata[3:,:,:],0,1)

    # Normalization for each of the RS layers
    mms = MinMaxScaler()
    for dim in range(3, data.shape[0]):
        data[dim, :, :] = mms.fit_transform(data[dim,:,:])

    # resize using skimage.
    if should_resize:
        small = np.zeros((7,IMG_SIZE,IMG_SIZE))
        for dim in range(0,7):
            small[dim,:,:] = resize(data[dim,:,:], (IMG_SIZE,IMG_SIZE))
        data = tf.convert_to_tensor(small, dtype=tf.float32)
        del small # save memory

    #Random rotation
    if dev and rotation:
        if(np.random.uniform(0,1) > augmentation_prop):
            data = np.rot90(data, axes=(1,2))
    
    _recode_0 = np.vectorize(lambda x: max(x,0))
    
    if dev:
        label = data[5, :, :]
        # Enforce to 0 and 1, then clip
        label = _recode_0(label)
        label = np.reshape(label, (1,IMG_SIZE,IMG_SIZE))

    features = np.delete(data, 5, 0) # cut off the UI channel


    # conver to tf
    tmp = np.zeros((IMG_SIZE,IMG_SIZE,6))
    for c in range(0, 6):
        tmp[:,:,c] = features[c,:,:]
    features = tmp
    del tmp

    if dev:
        label = tf.reshape(label, [IMG_SIZE,IMG_SIZE,1])

    features = np.nan_to_num(features)
    features = _recode_0(features)
    features = tf.cast(features, tf.float32)
    if dev:
        label = np.nan_to_num(label)
        label = tf.cast(label, tf.float32)

    if single_tgt:
        label = tf.math.reduce_sum(label)

    if rgb:
        features = features[:,:,:3]

    if dev:
        return (features, label)
    else:
        return features

            
def set_shapes(img, label, IMG_SIZE:int = 256, single_tgt:bool = False, rgb:bool= False):
    '''
    Explicitly state the shape of the image arrays for tf.Data.Dataset to work properly.
    args:
        img - feature array (must be np array)
        label - label array (must be np array)
        IMG_SIZE - Size of the image to set to
        single_tgt -  Flag to generate single target ouput. If set to True, the shape of the label is set to ([]), which means a scalar. If set to False,
            It will be set to (IMG_SIZE, IMG-SIZE, 1)
    
    output:
        img, label with shaps explicitly defined for tf.Data.Dataset.
    '''
    if rgb:
        n_c = 3
    else:
        n_c = 6

    img.set_shape([IMG_SIZE,IMG_SIZE,n_c])
    if single_tgt:
        label.set_shape([]) #scalar
    else:
        label.set_shape([IMG_SIZE,IMG_SIZE,1])
    return img, label

def set_pred_shapes(img, IMG_SIZE:int = 256, rgb:bool= False):
    '''
    See set_shapes(). Prediction routine variant that only sets the shape for the prediction image array.
    '''

    if rgb:
        n_c = 3
    else:
        n_c = 6

    img.set_shape([IMG_SIZE,IMG_SIZE,n_c])
    return img

def construct_dataset(path:str, IMG_SIZE:int = 256, single_tgt:bool = False, rgb:bool= False, calibrate_rgb:float = 0.4, calibrate_rs:float = 0.1):
    '''
    Main body function to create a tf.Data.Dataset.
    
    input args:
        path (str)- the path to the image, must be a .npy array.
        IMG_SIZE (int) - the desired size of the image to resize to. Currently the parser only support square images, so the output image will be turned to an array of IMG-SIZE * IMG_SIZE.
        single_tgt (bool) - Flag to generate single target ouput. If set to True, an extra reduce sum along the image width and image height axes will be applied to the image for use in the Efficient Net B0 model. If set to False,
                        the parser will return the output image as is.
    ouput:
        a tf.data.Dataset class with a tuple (feature, label) inside.
    '''
    dataset = tf.data.Dataset.list_files(path, seed=321, shuffle=False)

    dataset = dataset.map(lambda i: tf.numpy_function(func=parse_data, 
                                                inp=[i,True, 0.5, True,IMG_SIZE, True, single_tgt, calibrate_rgb, calibrate_rs, rgb], # parameters for parse_data()
                                                # parse_data(path , rotation:bool = False, augmentation_prop:float = 0.5, resize:bool= True, IMG_SIZE:int = 256, dev:bool= True, single-tgt:bool = False):
                                                Tout=(tf.float32, tf.float32)
                                                ), 
                        num_parallel_calls=tf.data.AUTOTUNE)

    # dataset = dataset.map(lambda i: tf.numpy_function(func=parse_data, 
    #                                         inp=[i,True, 0.5, True,IMG_SIZE, True, True], # parameters for parse_data()
    #                                         # parse_data(path , rotation:bool = False, augmentation_prop:float = 0.5, resize:bool= True, IMG_SIZE:int = 256, dev:bool= True, single-tgt:bool = False):
    #                                         Tout=(tf.float32, tf.float32)
    #                                         ), 
    #                 num_parallel_calls=tf.data.AUTOTUNE)

    # ensure shape goes here
    # dataset = dataset.map(lambda img,label: set_shapes(img,label,IMG_SIZE))
    dataset = dataset.map(lambda img,label: set_shapes(img,label,IMG_SIZE, single_tgt=single_tgt, rgb=rgb))
    return dataset

def construct_predset(path:str, IMG_SIZE:int = 256, single_tgt:bool = False, rgb:bool= False, calibrate_rgb:float = 0.4, calibrate_rs:float = 0.1):
    '''
    See construct_dataset(). The prediction data variant that only transforms the prediction data, with no label attached in the 
    outcome tf.data.Dataset.
    output:
        a class tf.data.Dataset with the feature array defined
    '''
    dataset = tf.data.Dataset.list_files(path, seed=321, shuffle=False)

    dataset = dataset.map(lambda i: tf.numpy_function(func=parse_data, 
                                                inp=[i,True, 0.5, True,IMG_SIZE, False,  single_tgt, calibrate_rgb, calibrate_rs, rgb], # parameters for parse_data()
                                                # parse_data(path , rotation:bool = False, augmentation_prop:float = 0.5, resize:bool= True, IMG_SIZE:int = 256):
                                                Tout=tf.float32
                                                ), 
                        num_parallel_calls=tf.data.AUTOTUNE)

    # ensure shape goes here
    dataset = dataset.map(lambda img: set_pred_shapes(img,IMG_SIZE, rgb))

    return dataset

def unit_test_dataloader(dataset, IMG_SIZE:int = 256, N_CHANNELS:int = 6, single_tgt:bool = False):
    '''
    A simple tester for the dataloader routine. It will check the shape of the constructed dataset to see if the shape is as expected.
    input args:
        dataset (tf.Data.Dataset) - a dataset built by calling `construct_dataset()`.
        IMG_SIZE (int) - IMG_SIZE defined when calling `construct_dataset()`.
        N_CHANNELS (int) - Number of channels in the feature array
        single-tgt (bool) - Flag for single target output. If set to True, the unit tester expects the shape of the label to be [] (scalar). If set to False,
            the unit tester expects the shape of the label to be (IMG_SIZE, IMG_SIZE, 1).
    '''
    for i in dataset.take(1):
        assert i[0].shape == (IMG_SIZE, IMG_SIZE, N_CHANNELS), f"Features have incorrect shape: {i[0].shape}; Expected: {(IMG_SIZE,IMG_SIZE, N_CHANNELS)}"
        if single_tgt:
            assert i[1].shape == [], f"Label has incorrect shape: {i[1].shape}; Expected: [] (Scalar)"
        else:
            assert i[1].shape == (IMG_SIZE,IMG_SIZE,1), f"Label has incorrect shape: {i[1].shape}; Expected: {(IMG_SIZE,IMG_SIZE,1)}"
    print("SUCCESS: Passed Unit Test for Data Loader")