# -*- coding: utf-8 -*-
import numpy as np
import os
from shutil import move
from re import sub
from glob import glob

np.random.seed(0) #seed for choice

def drawNMove(drt: str, target_dir:str ,n:int):
    '''
    Description:
    Draws N images (and its segmentation masks) from a target directory, and move them elsewhere.
    Use this function to rebalance the train-test split.

    Inputs:
    -   drt (str)
        Source directory containing the images and segmentation masks to be moved from.
    -   target_dir (str)
        Target directory where the images and segmentation masks to be moved to.
    -   n (int)
        The number of images to be sampled randomly and moved.

    Output:
        None
    '''

#drt = "/dataset/testset/images/testing"
#target_dir="/dataset/trainset/images/training"

    files = glob(drt+"/*.npy")
    filenames = [f.split("\\")[-1] for f in files]
    idx = range(0,len(files))
    ind = np.random.choice(idx,n, replace=False)

    for j in idx:
            if(j in ind):
                #test.append(j)
                move(files[j],target_dir+"/"+filenames[j])
                print("Moved Image:",filenames[j],"to",target_dir+"/"+filenames[j],sep=' ')

def TrainTestSplit(drt: str, target_dir:str ,probs:list):
    '''
    Description:
    An experimental version of automatic Train-test-split utilizing the drawNMove function.

    Input:
    drt (str)
        -A path where the unsplit dataset is located.

    target_dir (str)
        -A path where the splitted dataset should be created in.

    probs (list)
        -A list of length 3 representing the proportion of data to be splitted into the training, dev and test sets respectively.
        -If a dev set does not exist, then specify 0.
    
    Output:
    None
    '''
    assert np.sum(probs) ==1 # probs must sum to one

    N = len(glob(drt + "/*.npy"))
    N_train = int(np.floor(N*probs[0]))
    N_dev = int(np.floor(N*probs[1]))
    N_test = N-N_train-N_dev
    
    drawNMove(drt, target_dir+"/train" ,N_train)
    if(probs[1]>0):
        drawNMove(drt, target_dir+"/dev" ,N_dev)
    drawNMove(drt, target_dir+"/test" ,N_test)

TrainTestSplit("F:/gis/github/v2/data", "F:/gis/github/v2/data", probs=[0.8, 0.1, 0.1])
print("done")