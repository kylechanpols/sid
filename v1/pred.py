import tensorflow as tf
import datetime
import os
import pandas as pd #required for manipulation of the data as a dataset.

################################ Example Routine for making predictions

def predict(script_path: str, pred_dataset: str, show: bool)-> tf.Tensor:
    '''
    Description:
    General routine for predicting on out-of-sample images.

    Inputs:
    script_path (str)
        The root of the dataset and program scripts.
    pred_dataset (str)
        The location of the out-of-sample images
    show (bool)
        Should the model visualize the predictions? If yes, the model shows the source image, the ground truth and the model prediction.

    Output:
    A (n,IMG_SIZE,IMG_SIZE,2) tf.Tensor.
    '''

    #script_path = ""
    #pred_dataset = "/predictions/images/testing/*"

    exec(open(os.path.join(script_path + "model_compile.py")).read())

    # Local model weights
    model.load_weights(checkpoint_dir+"/cp-day2-wd.ckpt")

    ##### Get Predictions, Compute a luminosity index (count the number of 1s due to the use of Sparse Categorical Crossentrophy)

    #### Load the data

    data_for_pred = tf.data.Dataset.list_files(dataset_path + pred_dataset, seed=SEED, shuffle=False)
    #dataset = {"pred":data_for_pred}
    data_for_pred = data_for_pred.map(parse_image_pred)

    #dataset['pred'] = dataset['pred'].map(load_image_pred)
    data_for_pred = data_for_pred.batch(BATCH_SIZE)
    data_for_pred = data_for_pred.prefetch(buffer_size=AUTOTUNE)

    #### Visualize preds
    if(show == True):
        draw = int(np.random.randint(low=0, high=BATCH_SIZE-1, size=1)) #randomly draw a number between 0-32
        show_predictions(dataset=data_for_pred,num=1, num_in_batch = draw)

    return model.predict(data_for_pred)

def gini(x: pd.Series) -> float:
    '''
    Description:
    A **bruteforce** function to compute the gini coefficient.
    It computes the MAD of a distribution, and then reweights it with the relative MAD.

    Input: 
    x (pd.Series) 
        - a pandas series representing a predicted distribution.
    
    Output: 
    g (float)
        A float point value between -1 and 1 representing the degree of income inequality.
        -1 means absolutely no inequality and 1 means full inequality.

    Time Complexity: O(n**2)
    Space Complexity: O(n**2)

    Do NOT use for very large samples, e.g. n>10000.
    '''
    x = x.to_numpy(dtype=np.float32)

    x = x[np.logical_not(np.isnan(x))]
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

def makeDataset(val: pd.Series) -> pd.DataFrame:
    '''
    Description: Make a dataset from a list of named lists.

    Input: 
    val (pd.Series)
        pd.Series of predictions for a city, with the name of the series set to the name of that city.

    Output: 
        A pd.DataFrame containing all predictions for a city as columns.
    '''
    df = {}
    for city,value in zip(ref,val):
        df['city'] = val
    return df

def acquireValues(path: str, normalized: bool) -> pd.Series:
    '''
    Description: a Function to translate satellite images from a folder to a measure of development.

    Inputs:
    - path (str)
        The path to the folder containing .jpg satellite images.
    - normalized (bool)
        Should the program return normalized predictions or unnormalized image predictions?
        Normalization is defined as the coercion of pixels from scale 0 - 255 (RGB) to 0-1.
        Note that if normalized, then the images cannot be displayed properly by keras.utils.array_to_img().

    Outputs:
    A pd.Series containing the sum of bright pixels in an image.
    
        pd.Series.name - Contains the name of the folder (assumed to be name of city)
    '''
    ### the following code is for windows where using os.path.join would result in a unique identifier \\
    # for one path
    #city = re.search(r'\\([a-zA-Z]+$)',path)
    #city = city.group(1) #Acquire Name of city via Regex

    #for linux/mac, use this:
    city = os.path.basename(os.path.normpath(path)) #normpath gives us the "normalized" path by removing anything in between folders. #basename gives the current folder name.

    print("|","Processing:",city, sep=" ")

    TESTSET_SIZE = len(glob(path+ "/*.jpg"))
    print("|","City:",city,"contains",TESTSET_SIZE,"images", sep=" ")

    #correcting path
    path = tf.strings.regex_replace(path, "\\\\", "/")

    data_for_pred = tf.data.Dataset.list_files(path+"/*.jpg", seed=SEED, shuffle=False)
    data_for_pred = data_for_pred.map(parse_image_pred)
    data_for_pred = data_for_pred.batch(BATCH_SIZE)
    data_for_pred = data_for_pred.prefetch(buffer_size=AUTOTUNE)

    print("|--->","City",city,"has been processed by tf.data.Dataset.", sep =" ")

    preds = model.predict(data_for_pred)

    print("|--->","Predictions for",city,"acquired", sep=" ")

    if normalized == True:
        preds = tf.reshape(preds[:,:,:,1], (TESTSET_SIZE,IMG_SIZE, IMG_SIZE))
    else:
        preds = tf.reshape(preds[:,:,:,1]*255.0, (TESTSET_SIZE,IMG_SIZE, IMG_SIZE))

    preds = tf.math.reduce_sum(preds, axis=[1,2])

    out = pd.Series(list(preds), dtype=np.float32, name=city)

    print("|--->","Predictions for",city,"restructured to pd.Series.", sep=" ")

    return out