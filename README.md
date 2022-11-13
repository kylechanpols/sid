# Satellite Index of Development (SID)

Welcome to the GitHub repository for the Satellite Index of Development (SID) project. The GitHub repository serves as a repository for the latest build of code and Google Colab notebooks.

## Stable Release
`v1` folder contains version 1.0 of the model. The v1 model makes use of the RGB layers of a satellite image to make predictions of the segmentation mask.

## Project Description

This is a project that aims to construct an index of local economic development from Google Satellite Images. Because there is no existing method to measure economic development in different localities, this project aims to fill that gap by leveraging the power of Convolutional Neural Networks (ConvNets). The method used here can be used to construct an index using images measuring different levels of localities: counties, cities, regions and countries.

The model takes a distribution of satellite images, parses them through a U-Net architecture pre-trained to predict on a binary segmentation mask where class 0 points to non-building pixels and class 1 points to building pixels. The model expects a pair of source image and its segmentation mask for training and testing:

<img src="https://drive.google.com/uc?id=12NYztvcf0-WqEKkDSuhys19CF4BYJo7d" alt="Urban" width="256"/>

<img src="https://drive.google.com/uc?id=1VMBxH81uwqydFm4BS1yOdftRTwy00E4R" alt="Rural" width="256"/>

But it can take a distribution of unseen images of any dimension to make predictions.

# Version 2 - 7-channel model (WIP)

## File Description

### `data_loader.py`
- This contains the `tf.data.Dataset` data pipeline and the preprocessing pipeline in one file.

## A note on model weights

These are available on Google Drive as GitHub does not allow excessive large files. See the [entire shared folder here](https://drive.google.com/drive/folders/1m38V-wL2gPonnoeZtQYD3jBfLf0dbBth?usp=sharing).

## File Description - Jupyter Notebooks

### training_demo.ipynb

The main demo for the project code and scripts, with an example of training and testing on the project's dataset of 4300 images.

### training_demo_technical.ipynb

The main demo notebook but with a detailed walkthrough of all helper functions written for this project.

## File Description - Scripts

### tf_setup.py

Setup script for TensorFlow.

### data_loader.py

Contains all helper functions for the data pipeline (for preprocessing images and parsing images for tf).

### model_compile.py

Model Definition and Model Compilation.

### pred.py

An example routine for making predictions on unseen examples with the pre-trained model weights.

### traintestsplit.py

Functions to perform directory-level train-test split required for the model.

### move1lvup.py

A helper function to move all files in nested sub-directories to one single folder. This is required if you download Google Satellite Images with the Google Satellite Images Downloader.





