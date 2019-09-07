import matplotlib.pyplot as plt
import numpy as np
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

path = '../data/data/'
example_img = plt.imread(path+'IMG/center_2016_12_01_13_30_48_287.jpg')
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
num_epochs = 20

# L2-Regularization
l2_penal = .0
# Dropout rates
dr1 = 0.3
dr2 = 0.2
dr3 = 0.1


############################
### Function definitions ###
############################

# create list with all image names and angles
def read_csv():
	samples = []
	with open(path+'driving_log.csv') as csvfile:
	    reader = csv.reader(csvfile)
	    next(reader) # skip headers in first row
	    for line in reader:
	    	# Append twice for data augmentation. Noise layer in the model avoids using the identical images.
	        samples.append(line)
	        samples.append(line)
	print('number of images (augmented data set): ',len(samples))
	return samples

# Model is based on NVIDIAS paper "End to End Learning for Self-Driving Cars"
def def_model():

	model = Sequential()
	# normalize data 
	model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(img_shape[0], img_shape[1], img_shape[2]), 
		output_shape=(img_shape[0], img_shape[1], img_shape[2])))
	# crop images
	model.add(Cropping2D(cropping=((top_crop, bottom_crop),(left_crop, right_crop))))
	# add noise layer
	model.add(GaussianNoise(.1))
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
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                # center camera image
                center_img_name = path+'IMG/'+batch_sample[0].split('/')[-1]
                center_img = plt.imread(center_img_name)
                center_angle = float(batch_sample[3])
                images.append(center_img)
                angles.append(center_angle)
                # include flipped image
                images.append(np.fliplr(center_img))
                angles.append(-center_angle)
                
                # left camera images
                left_img_name = path+'IMG/'+batch_sample[1].split('/')[-1]
                left_img = plt.imread(left_img_name)
                left_angle = center_angle + angle_offset
                images.append(left_img)
                angles.append(left_angle)
                # include flipped image
                images.append(np.fliplr(left_img))
                angles.append(-left_angle)
                
                # right camera images
                right_img_name = path+'IMG/'+batch_sample[2].split('/')[-1]
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
def prepare_data(samples):

	train_samples, validation_samples = train_test_split(samples, test_size=0.2)

	train_generator = generator(train_samples, batch_size=batch_size)
	validation_generator = generator(validation_samples, batch_size=batch_size)

	return len(train_samples), len(validation_samples), train_generator, validation_generator


# check if gpu is available for training
def check_gpu_status():

	print(tf.test.is_gpu_available())
	print(tf.test.gpu_device_name())
	print(tf.test.is_built_with_cuda())


# train and validate the model
def train_model(model, num_train_samples, num_validation_samples, train_generator, validation_generator):

	# create callbacks
	checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)
	stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=3)

	# fit model
	history = model.fit_generator(train_generator, 
	                    steps_per_epoch=np.ceil(num_train_samples*6/batch_size), 
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
	
	samples = read_csv()
	model = def_model()
	num_train_samples, num_validation_samples, train_generator, validation_generator = prepare_data(samples)
	check_gpu_status()
	train_model(model, num_train_samples, num_validation_samples, train_generator, validation_generator)


if __name__ == "__main__":
    main()