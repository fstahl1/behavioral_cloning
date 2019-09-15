# Self-Driving Car Engineer Nanodegree

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
---

## Deep Learning Project: Behavioral Cloning

Design, train and validate a model for predicting steering angles from image data

---

### Overview

The steps of the project are the following:

* Data acquisition
* Data exploration
* Data augmentation
* Adaption of the CNN architecture from the NVIDIA paper [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
* Model training
* Test of the trained model with the [Udacity simulator](https://github.com/udacity/self-driving-car-sim)
* Video recording of the vehicle driving one full lap autonomously

---


### Model architecture

Since the NVIDIA model from the [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) paper was designed exactly for this purpose, I adapted the model for this project. 

The model is basically a convolutional neural network (CNN) with five convolutional layers and three fully connected layers and is shown below. 

* Normalization and Cropping: After normalizing the input images in the first layer, the top of each image is cropped in the second layer since the upper part mostly shows sky and trees and does not contain any useful features for steering the vehicle.
* Gaussian noise: The next layer adds gaussian noise to the images to avoid using identical flipped images for training.
* Convolutional layers: Next, five convolutional layers are added. Like the NVIDIA model, the first three layers use a 5x5 Kernel with 2x2 stride followed by two layers with 3x3 Kernel and 1x1 stride.
* Flatten and Dense layers: After fla

Since it is a regression network which is supposed to output a continuous steering angle, the last layer consists of a single node without an activation function. For minimizing the error between the predicted steering angles and the ground truth steering angles, mean squared error (mse) is used as loss function.


### Model training

To avoid loading all images into memory at the same time, a generator was used. Training was done using ADAM optimizer which works with an adaptive learning rate. 

The dataset basically consists of one round driving in the middle of the road as precisely as possible and one recovery round, where the recording was started at the side of the road and steered back to the center. Additionally, the two curves with high curvature are driven again.
The images of all three cameras are used with an offset angle of 0.2 for the left image and -0.2 for the right image. Furthermore, a flipped version of each image was included to the dataset.