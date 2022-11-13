# Satellite Index of Development (SID)

Welcome to the GitHub repository for the Satellite Index of Development (SID) project. The GitHub repository serves as a repository for the latest build of code and Google Colab notebooks.

## Release
`v1` folder contains version 1.0 of the model. The v1 model makes use of the RGB layers of a satellite image to make predictions of the segmentation mask.

## Project Description

This is a project that aims to construct an index of local economic development from Google Satellite Images. Because there is no existing method to measure economic development in different localities, this project aims to fill that gap by leveraging the power of Convolutional Neural Networks (ConvNets). The method used here can be used to construct an index using images measuring different levels of localities: counties, cities, regions and countries.

The model takes a distribution of satellite images, parses them through a U-Net architecture pre-trained to predict on a binary segmentation mask where class 0 points to non-building pixels and class 1 points to building pixels. The model expects a pair of source image and its segmentation mask for training and testing:

<img src="https://drive.google.com/uc?id=12NYztvcf0-WqEKkDSuhys19CF4BYJo7d" alt="Urban" width="256"/>

<img src="https://drive.google.com/uc?id=1VMBxH81uwqydFm4BS1yOdftRTwy00E4R" alt="Rural" width="256"/>

But it can take a distribution of unseen images of any dimension to make predictions.

# Version 2 - 6-channel model

The V2 models use a different training dataset (Landsat 8 time series). It has 6 channels, and the problem has been reformulated as a image regression problem. Instead of classifying pixels as built-up or not, I use the U-Net architecture to create an image regression model that guesses the value of the Normalized Difference Built-up Index (NDBI), or Urban Index for each pixel.

<img src="https://drive.google.com/uc?id=1buH_AyQ51jpy1qDKPtxHClv4alN0tD99" alt="NDBI"/>
                                                                                                 
The image below shows an example of model predictions: The upper image shows the source image, the lower left image shows the source NDBI band, and the lower right shows the model's predictions for the NDBI band.

<img src="https://drive.google.com/uc?id=1Qgoq4oKRoguSRmqZuAICT-x8Cn5jAVH5" alt="NDBI"/>

A brief summary of the performance against an EfficientNet B0 Transfer Learning implementation is available below:

| Model | Training RMSE | Testing RMSE | Correlation with Brookings 2014 Data^ |
| ----- | ------------- | ------------ | ------------------------------------ |
| U-Net | .071 | .098 | .486 |
| EfficientNetB0 | .115 | .093 | .467

Note: 
*- EfficientNetB0 RMSE was reported at the image-level not pixel-level. These are estimates acquired by averaging the image-level RMSE by the size of the image (dividing by 128x128=16384)

^- With the model predictions I computed the weighted coefficient of variation (Williamson, 1965) and tested the economic development predictions against the Brookings Institute 2014 city-level GDP data. This column reports the correlation between the model predictions against an existing dataset on city-level economic development.

## File Description

## File Description - Core Scripts

### `tf_setup.py`

Setup script for TensorFlow.

### `data_loader.py`

Contains all helper functions for the data pipeline (for preprocessing images and parsing images for tf).

### `model_compile.py`

Model Definition and Model Compilation.

### `prediction.py`

An example routine for making predictions on unseen examples with the pre-trained model weights. Supports batch caching to prevent data loss upon prediction error.

### `build_index_batch.py`

Routine to convert image tile-level predictions to SID predictions at the city-year level.

## File Description - Utilities

### `get_coords_osm_api.py`

Program to extract city limits from the OpenStreetMaps API

### `tif_read_batch.py`

Program to read a .TIF image downloaded from the Google Earth API and split them into tiles in .npy

### `google_earth_engine_extract_in_batch.js`

Program to scrap Landsat 8 images using the Google Earth API.

### `google_earth_engine_vis_ui.js`

Program to visualize the NDBI band of Landsat 8 images on the Google Earth Online GUI.



