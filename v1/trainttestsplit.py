# -*- coding: utf-8 -*-
import numpy as np
import os
from shutil import move
from re import sub
from glob import glob

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


    mask_drt = sub("images", "annotations", drt)
    mask_tgt = sub("images", "annotations",target_dir)

    path = os.walk(drt)

    for root, directories, files in path:
        idx = range(0,len(files))
        ind = np.random.choice(idx,n, replace=False)
        
        for j in idx:
            if(j in ind):
                #test.append(j)
                move(drt+"/"+files[j],target_dir+"/"+files[j])
                print("Moved Image:",files[j],"to",target_dir+"/"+files[j],sep=' ')
                move(mask_drt+"/"+files[j],mask_tgt+"/"+files[j])
                print("Moved Mask:",files[j],"to",mask_tgt+"/"+files[j],sep=' ')

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

    N = len(glob(drt + "/images/training/*.jpg"))
    
    drawNMove(drt, target_dir+"/training" ,N*probs[0])
    if(probs[1]>0):
        drawNMove(drt, target_dir+"/dev" ,N*probs[1])
    drawNMove(drt, target_dir+"/testing" ,N*probs[2])