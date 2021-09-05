# Satellite Index of Development (SID)

Welcome to the GitHub repository for the Satellite Index of Development (SID) project. The GitHub repository serves as a repository for the latest build of code and Google Colab notebooks.

## Stable Release
Please consult the [Project webpage](https://kylechanhy.netlify.app/project/sidv1/) for the stable version of the code and scripts.

## Project Description

This is a project that aims to construct an index of local economic development from Google Satellite Images. Because there is no existing method to measure economic development in different localities, this project aims to fill that gap by leveraging the power of Convolutional Neural Networks (ConvNets). The method used here can be used to construct an index using images measuring different levels of localities: counties, cities, regions and countries.

The model takes a distribution of satellite images, parses them through a U-Net architecture pre-trained to predict on a binary segmentation mask where class 0 points to non-building pixels and class 1 points to building pixels. The model expects a pair of source image and its segmentation mask for training and testing:

<img src="https://drive.google.com/uc?id=12NYztvcf0-WqEKkDSuhys19CF4BYJo7d" alt="Urban" width="256"/>

<img src="https://drive.google.com/uc?id=1VMBxH81uwqydFm4BS1yOdftRTwy00E4R" alt="Rural" width="256"/>

But it can take a distribution of unseen images of any dimension to make predictions.

## File Description

### training_demo.ipynb

The main demo for the project code and scripts, with an example of training and testing on the project's dataset of 4300 images.

### training_demo_technical.ipynb

The main demo notebook but with a detailed walkthrough of all helper functions written for this project.




