import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
import sklearn
import random
from sklearn.model_selection import train_test_split
import cv2
from scipy import ndimage # for importing image as RGB
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers import Cropping2D, GaussianNoise, MaxPool2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers



######################
### Initialization ###
######################

path = '../data/'
example_img = plt.imread(path+'IMG/center_2019_09_14_10_46_34_907.jpg')
img_shape = example_img.shape

save_path = './model.h5'



#############################
### Parameter definitions ###
#############################

# number of bins for histogram
n_bins = 15

# offset steering value for left and right camera images
angle_offset = .25
# test size ratio (train-test-split)
test_size = .2
# training parameters
batch_size = 32
num_epochs = 5

# L2-regularization rate
l2_penal = 0#.0001
# Dropout rates for fully connected layers
dr1 = 0.3
dr2 = 0.2
dr3 = 0.1



############################
### Function definitions ###
############################


def read_csv():
	
	"""Read the .csv file and return pandas dataframe."""

	columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
	df = pd.read_csv(path+'driving_log.csv', usecols=[0,1,2,3], names=columns)

	return df


def augment_data(df):

	"""Augment data: duplicate images with rare larger angles."""

	# create histogram and calculate maximum number of elements per bin
	(counts, bins) = np.histogram(df.steering, bins=n_bins)
	max_count = max(counts)

	# initialize dataframe for augmented data
	df_aug = pd.DataFrame(columns=list(df.columns))

	for curr_bin in range(n_bins):
	    
	    ind = (df.steering > bins[curr_bin]) & (df.steering <= bins[curr_bin+1])
	    elem_per_bin = sum(ind)
	    
	    if elem_per_bin>0:
	        aug_fact = int(max_count / elem_per_bin)
	        df_aug = df_aug.append([df[ind]]*max(int(aug_fact/3),1), ignore_index=True)

	# add noise for not using the identical angles for augmented images
	df_aug.steering += np.random.normal(-0.005,0.005,len(df_aug.steering))
	    
	return df_aug


def def_model():

	"""Define CNN model based on NVIDIA paper \"End to End Learning for Self-Driving Cars\"."""

	model = Sequential()
	# normalize images 
	model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(img_shape[0], img_shape[1], img_shape[2]), 
		output_shape=(img_shape[0], img_shape[1], img_shape[2])))
	 # downscale image
	model.add(MaxPool2D((2,2)))
	# add noise layer
	model.add(GaussianNoise(.25))
	# add convolutional layers
	model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation='relu'))#, subsample=(2, 2)))
	model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation='relu'))#, subsample=(2, 2)))
	model.add(Conv2D(48, kernel_size=(5, 5), strides=(2,2), activation='relu'))#, subsample=(2, 2)))
	model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation='relu'))
	model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation='relu'))
	# add flatten layer
	model.add(Flatten())
	# add fully connected layers plus dropout
	model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(l2_penal)))
	model.add(Dropout(rate=dr1))
	model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(l2_penal)))
	model.add(Dropout(rate=dr2))
	model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(l2_penal)))
	model.add(Dropout(rate=dr3))
	model.add(Dense(1))

	# compile method configures learning process
	model.compile(optimizer='adam', loss='mse')

	return model


def generator(df_split, batch_size=32):

	"""Load and output training data in batches to avoid running out of memory."""

	num_samples = len(df_split)
	while 1:
		df_split = sklearn.utils.shuffle(df_split)
		for offset in range(0, num_samples, batch_size):
			batch_samples = df_split.iloc[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples.iterrows():
				
				# center camera image
				center_img_name = path+batch_sample[1].center
				center_img = plt.imread(center_img_name)
				center_angle = float(batch_sample[1].steering)
				images.append(center_img)
				angles.append(center_angle)
				# include flipped image
				images.append(np.fliplr(center_img))
				angles.append(-center_angle)
				
				# left camera images
				left_img_name = path+batch_sample[1].left
				left_img = plt.imread(left_img_name)
				left_angle = center_angle + angle_offset
				images.append(left_img)
				angles.append(left_angle)
				# include flipped image
				images.append(np.fliplr(left_img))
				angles.append(-left_angle)
				
				# right camera images
				right_img_name = path+batch_sample[1].right
				right_img = plt.imread(right_img_name)
				right_angle = center_angle - angle_offset
				images.append(right_img)
				angles.append(right_angle)
				# include flipped image
				images.append(np.fliplr(right_img))
				angles.append(-right_angle)

			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)


def prepare_data(df):

	"""prepare data by splitting into training and validation samples and creating generators."""

	train_df, valid_df = train_test_split(df, test_size=0.2)

	train_generator = generator(train_df, batch_size=batch_size)
	valid_generator = generator(valid_df, batch_size=batch_size)

	return len(train_df), len(valid_df), train_generator, valid_generator


def check_gpu_status():

	"""Check if GPU is available for training."""

	print('\n\nIs GPU available: ', tf.test.is_gpu_available())
	if not tf.test.is_gpu_available:
		raise Exception('GPU not available')
	print('GPU device name: ', tf.test.gpu_device_name())
	print('Is built with cuda: ', tf.test.is_built_with_cuda())
	print('\n')


def train_model(model, n_train, n_valid, train_generator, valid_generator):

	"""Train and validate the model."""

	# create callbacks
	checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)
	stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=3)

	# fit model
	history = model.fit_generator(train_generator, 
						steps_per_epoch=np.ceil(n_train*6/batch_size), # the factor 6 is there because the generator outputs all 3 camera images plus flipped version
						validation_data=valid_generator, 
						validation_steps=np.ceil(n_valid*6/batch_size), 
						callbacks=[checkpoint, stopper],
						epochs=num_epochs,
						verbose=1)

	print(history.history.keys())
	### plot the training and validation loss for each epoch
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()



#####################
### main function ###
#####################

def main():
	
	df = read_csv()
	df_aug = augment_data(df)
	model = def_model()
	n_train, n_valid, train_generator, valid_generator = prepare_data(df_aug)
	check_gpu_status()
	train_model(model, n_train, n_valid, train_generator, valid_generator)


if __name__ == "__main__":
	main()