import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
import sklearn
from sklearn.model_selection import train_test_split
import cv2
from scipy import ndimage # for importing image as RGB
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers import Cropping2D, GaussianNoise
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

# set crop values
top_crop = 60
bottom_crop = 0
left_crop = 0
right_crop = 0

test_size = .2
angle_offset = .25
batch_size = 32
num_epochs = 5

# L2-Regularization
l2_penal = .0001
# Dropout rates
dr1 = 0.3
dr2 = 0.2
dr3 = 0.1


############################
### Function definitions ###
############################

def read_csv():

	columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

	data = pd.read_csv(path+'driving_log.csv', usecols=[0,1,2,3], names=columns)

	return data


# # create list with all image names and angles
# def read_csv():
# 	samples = []
# 	_angles_orig = []
# 	_angles_aug = []
# 	with open(path+'driving_log.csv') as csvfile:
# 	    reader = csv.reader(csvfile)
# 	    next(reader) # skip headers in first row
# 	    for line in reader:
# 	        # samples.append(line)
# 	        samples.append(line)
# 	        # append sample again for all angles != 0 (balancing) # Noise layer in the model avoids using the identical images.
# 	        _angle = float(line[3])
# 	        _angles_orig.append(_angle)
# 	        _angles_aug.append(_angle)
# 	        # if _angle!=0:
#         	num_copies = abs(int(np.round(_angle*10)))
#         	# print(num_copies)
#         	for i in range(num_copies):
# 	        	samples.append(line)
# 	        	_angles_aug.append(_angle)
# 	print('number of images (augmented data set): ',len(samples))
# 	return samples, _angles_orig, _angles_aug

# Model is based on NVIDIAS paper "End to End Learning for Self-Driving Cars"
def def_model():

	model = Sequential()
	# normalize data 
	model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(img_shape[0], img_shape[1], img_shape[2]), 
		output_shape=(img_shape[0], img_shape[1], img_shape[2])))
	# crop images
	model.add(Cropping2D(cropping=((top_crop, bottom_crop),(left_crop, right_crop))))
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

	# show model summary
	model.summary()

	# compile method configures learning process
	model.compile(optimizer='adam', loss='mse')

	return model


# Generator is defined to avoid loading all data into memory at once
def generator(split_data, batch_size=32):
    num_samples = len(split_data)
    while 1:
        split_data = sklearn.utils.shuffle(split_data)
        for offset in range(0, num_samples, batch_size):
            batch_samples = split_data.iloc[offset:offset+batch_size]

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


# prepare data by splitting into training and validation samples and creating generators
def prepare_data(data):

	train_samples, validation_samples = train_test_split(data, test_size=0.2)

	train_generator = generator(train_samples, batch_size=batch_size)
	validation_generator = generator(validation_samples, batch_size=batch_size)

	return len(train_samples), len(validation_samples), train_generator, validation_generator


# check if gpu is available for training
def check_gpu_status():

	print('\n\nIs GPU available: ', tf.test.is_gpu_available())
	if not tf.test.is_gpu_available:
		raise Exception('GPU not available')
	print('GPU device name: ', tf.test.gpu_device_name())
	print('Is built with cuda: ', tf.test.is_built_with_cuda())
	print('\n')


# train and validate the model
def train_model(model, num_train_samples, num_validation_samples, train_generator, validation_generator):

	# create callbacks
	checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)
	stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=3)

	# fit model
	history = model.fit_generator(train_generator, 
	                    steps_per_epoch=np.ceil(num_train_samples*6/batch_size), # the factor 6 is there because the generator outputs all 3 camera images plus flipped version
	                    validation_data=validation_generator, 
	                    validation_steps=np.ceil(num_validation_samples*6/batch_size), 
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
	
	data = read_csv()
	model = def_model()
	num_train_samples, num_validation_samples, train_generator, validation_generator = prepare_data(data)
	check_gpu_status()
	train_model(model, num_train_samples, num_validation_samples, train_generator, validation_generator)


if __name__ == "__main__":
    main()