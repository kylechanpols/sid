import os
from pickletools import optimize
from keras import backend as K
import glob
import pandas as pd
import numpy as np
import tensorflow_addons as tfa

# Define paths
main_path =os.getcwd()

checkpoint_dir = os.path.join(main_path,"weights")

# Desired image size
IMG_SIZE=128

# Sourcing scripts
exec(open(os.path.join(main_path, "tf_setup.py")).read())
exec(open(os.path.join(main_path,  "data_loader.py")).read())

# Optimizer setup
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
radam = tfa.optimizers.RectifiedAdam(learning_rate=0.5)
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
optimizer = ranger

# Load the pretrained model
model = tf.keras.models.load_model(os.path.join(checkpoint_dir, "20220912_efficientnetb0"),
    custom_objects = {"root_mean_squared_error":root_mean_squared_error,
        'radam':radam,
        'ranger':ranger})

# Helper functions to make predictions based on city and year.

def pred(model, requirements, IMG_SIZE:int = 128, shard:bool= True, shard_size:int = 100, filename_prefix:str = "model",array_tgt:bool = True):
    '''
    Main function to make predictions and save the predictions as .csv.
    input ars:
        model (tf.keras.models.Model) - A pretrained Keras Model to make predictions from
        requirements (list) - A list of tuples (CITY, YEAR) to make predictions from.
        IMG_SIZE (img) - The desired image width and height to resize to. Note that it's always recommended to downsize the source image to simplify the complexity of the learning task.
        shard (bool) - Should we shard the output .csv into different parts? If set to true, a shard will be made for every `shard_size` rows in the .csv.
        shard_size (int) - How many rows of predictions to store in each shard, if shard is set to True.
        filename_prefix (str) - Prefix of the predictions .csv files.
        array_tgt (bool) : Flag for a model that accepts an np.ndarray target. If set to True, no transformation would be applied when constructing the .csv, and the .csv will show
            the IMG_SIZE*IMG_SIZE array of predicted urban index values. If set to False, it will apply a reduce sum to flatten the predictions to a [] scalar of the sum of the urban index.

    output:
        .csvs of model predictions for the desired city and year listed in `requirements`.
    '''
    def make_pred(model, city, year, IMG_SIZE:int = 128, array_tgt:bool = True):
        # Prediction Helper Function
        dataset = construct_predset(os.path.join(main_path, "data", "pred", f"{city}_{year}_*"), IMG_SIZE=IMG_SIZE)
        dataset = {"pred": dataset}
        dataset['pred'] = dataset['pred'].batch(BATCH_SIZE)
        dataset['pred'] = dataset['pred'].prefetch(buffer_size=AUTOTUNE)
        preds = model.predict(dataset['pred'])
        if array_tgt:
            preds = preds.sum(axis=(1,2,3))
        return preds
    
    # Output handler and sharding routine

    final_out = pd.DataFrame() # init a dataset to be saved to disk
    regular_exec = True #flag to see if the files were found, and thus execute

    tot = len(requirements)
    if shard: # sharding helps cache progress to disk
        shard_step = 0
        # calc how many shards to make:
        required_shards = tot//shard_size
        while(shard_step <= required_shards):
            iter = 1
            final_out = pd.DataFrame() #init empty df at each shard
            while(iter <= shard_size):
                regular_exec = True # always reset regular exectuon back to True

                city, year = requirements[(shard_size*shard_step) + iter][0], requirements[(shard_size*shard_step) + iter][1]
                print(f"Attempt to make preds for {city} in {year}")
                uri = os.path.join(main_path, "data", "pred", f"{city}_{year}_*")
                # search for the desired CITY, YEAR combination in the pred set.
                try:
                    assert len(glob.glob(uri))>0, f"failed to find files for {city} in {year}."
                except AssertionError:
                    # if not found, we skip this CITY, YEAR combination. We do this by setting regular_exec to false such that the prediction routine would not run.
                    print(f"Failed to make preds for {city} in {year}. Skipping..")
                    regular_exec = False
                    iter += 1
                if regular_exec:
                    tmp_out = make_pred(model, city, year, array_tgt=array_tgt)
                    tmp_size = len(glob.glob(uri))
                    if array_tgt == False:
                        tmp_out = np.reshape(tmp_out, (tmp_size,)) #flatten back to 1-d array.
                    tmp = pd.DataFrame(
                    {"City": np.repeat(city, tmp_size),
                    "Year":  np.repeat(year, tmp_size),
                    "grid_id": np.arange(0,tmp_size),
                    "pred":tmp_out}
                    ) #tmp dataframe to be merged with final_out

                    print(f"Preds for {city} in {year} created")
                    final_out = pd.concat([final_out, tmp], ignore_index=True) # merge with the `final_out` dataframe
                    iter +=1

                    if ((shard_size*shard_step) + iter) == tot: #last shard
                        final_out.to_csv(os.path.join(main_path, "preds",f"{filename_prefix}_{shard_step}.csv"))
                        print(f"Saved shard {shard_step}")
                        print("Done")
                        break

            # upon reaching desired shard size, wrap up the last shard and generate the ouput.
            final_out.to_csv(os.path.join(main_path, "preds",f"{filename_prefix}_{shard_step}.csv"))
            shard_step += 1
            print(f"Saved shard {shard_step}")
    else:
        for i in range(0, tot):
            regular_exec = True # always reset regular exectuon back to True

            city, year = requirements[i][0], requirements[i][1]
            print(f"Attempt to make preds for {city} in {year}")
            uri = os.path.join(main_path, "data", "pred", f"{city}_{year}_*")
            try:
                assert len(glob.glob(uri))>0, f"failed to find files for {city} in {year}."
            except AssertionError:
                print(f"Failed to make preds for {city} in {year}. Skipping..")
                regular_exec = False
                continue
            if regular_exec:
                tmp_out = make_pred(model, city, year, array_tgt=array_tgt)
                tmp_size = len(glob.glob(uri))
                if array_tgt == False:
                    tmp_out = np.reshape(tmp_out, (tmp_size,)) #flatten back to 1-d array.
                tmp = pd.DataFrame(
                {"City": np.repeat(city, tmp_size),
                "Year":  np.repeat(year, tmp_size),
                "grid_id": np.arange(0,tmp_size),
                "pred":tmp_out}
                ) #tmp dataframe to be merged with final_out

                print(f"Preds for {city} in {year} created")
                final_out = pd.concat([final_out, tmp], ignore_index=True)
        final_out.to_csv(os.path.join(main_path, "preds",f"{filename_prefix}.csv"))
        print(f"Wrote final output to {filename_prefix}.csv")

# Constructing the `requirements` list:
# It's actuallty a cartesian product of the required city vector and required year vector.
import glob
cities = glob.glob(os.path.join(main_path, "data", "pred","*_2015_0.npy"))
cities = [city.split("\\")[-1].split("_")[0] for city in cities] # split by \\, find last element, then split by _, find first element, that's the city name.

years = glob.glob(os.path.join(main_path, "data", "pred",f"{cities[1]}_*_0.npy"))
years = [year.split("\\")[-1].split("_")[1] for year in years] # same as above but second element is the year.

from itertools import product

# populate the requirements list
requirements = []
for i in product(cities, years):
    requirements.append(i)

pred(model, requirements, IMG_SIZE= 128, shard= True, shard_size= 100, filename_prefix= "trans_efnet_preds", array_tgt=False)

# Can also rewrite this as a main function for running directly:
# if '__name__' == '__main__':
#     pred(model, requirements, IMG_SIZE= 128, shard= True, shard_size= 100, filename_prefix= "trans_efnet_preds", array_tgt=False)