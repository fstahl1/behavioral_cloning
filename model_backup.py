import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import seaborn as sns
import cv2
import numpy as np
import sklearn
from scipy import ndimage # for importing image as RGB
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping


# initialization

path = '../data/data/'
example_img = plt.imread(path+'IMG/center_2016_12_01_13_30_48_287.jpg')

img_shape = example_img.shape

def explore_data():
	print('img_shape =',img_shape)
	plt.imshow(example_img)
	x_val = np.arange(0,example_img.shape[1])
	plt.plot(x_val, np.ones(len(x_val))*60, 'r')
	plt.show()


top_crop = 60
bottom_crop = 0
left_crop = 0
right_crop = 0

test_size = .2
batch_size = 32
num_epochs = 20



def read_csv():

	samples = []
	angles = []
	with open(path+'driving_log.csv') as csvfile:
	    reader = csv.reader(csvfile)
	    next(reader) # skip headers in first row
	    for line in reader:
	        samples.append(line)
	        angles.append(float(line[3]))

	train_samples, validation_samples = train_test_split(samples, test_size=test_size)

	print('number of images: ',len(samples))
	print('number of training samples: ', len(train_samples))
	print('number of validation samples: ', len(validation_samples))

	np.random.seed(42)

	# display example images and angles:
	for i in range(3):
	    rand_num = np.random.randint(len(samples))
	    rand_sample = samples[rand_num]
	    center_img_name = path+'IMG/'+rand_sample[0].split('/')[-1]
	    center_img = plt.imread(center_img_name)
	    center_angle = float(rand_sample[3])
	    print(center_angle)
	    plt.imshow(center_img)
	    #plt.show()

	fig = plt.figure(figsize=(10,6))
	plt.hist(angles, 30)
	plt.title('Steering angle distribution (absolute)')
	plt.xlabel('Steering angle')
	#plt.show()

	fig = plt.figure(figsize=(10,6))
	plt.plot(angles)
	plt.title('Angle over time')
	plt.xlabel('Sample')
	plt.ylabel('Angle')
	#plt.show()

	return train_samples, validation_samples


def check_gpu_status():

	print(tf.test.is_gpu_available())
	print(tf.test.gpu_device_name())
	print(tf.test.is_built_with_cuda())


def def_model():

	model = Sequential()
	# Preprocess incoming data, centered around zero with small standard deviation 
	model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(img_shape[0], img_shape[1], img_shape[2]), 
		output_shape=(img_shape[0], img_shape[1], img_shape[2])))
	model.add(Cropping2D(cropping=((top_crop, bottom_crop),(left_crop, right_crop))))
	model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation='relu'))#, subsample=(2, 2)))
	model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation='relu'))#, subsample=(2, 2)))
	model.add(Conv2D(48, kernel_size=(5, 5), strides=(2,2), activation='relu'))#, subsample=(2, 2)))
	model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation='relu'))
	model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation='relu'))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(.5))
	model.add(Dense(50, activation='relu'))
	model.add(Dropout(.5))
	model.add(Dense(10, activation='relu'))
	model.add(Dropout(.5))
	model.add(Dense(1))

	model.summary()

	model.compile(loss='mse', optimizer='adam')
	# model.compile(optimizer=Adam(learning_rate), loss='mse')

	return model


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_img_name = path+'IMG/'+batch_sample[0].split('/')[-1]
                center_img = plt.imread(center_img_name)
#                 left_img_name = path+'IMG/'+batch_sample[1].split('/')[-1]
#                 left_img = plt.imread(left_img_name)
#                 right_img_name = path+'IMG/'+batch_sample[2].split('/')[-1]
#                 right_img = plt.imread(right_img_name)
                center_angle = float(batch_sample[3])
#                 left_angle = center_angle + angle_offset
#                 right_angle = center_angle - angle_offset
                images.append(center_img)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def prepare_data(train_samples, validation_samples):

	train_generator = generator(train_samples, batch_size=batch_size)
	validation_generator = generator(validation_samples, batch_size=batch_size)

	return train_generator, validation_generator


def train_model(model, train_samples, validation_samples, train_generator, validation_generator):

	save_path = './model.h5'

	checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)
	stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=5)

	history = model.fit_generator(train_generator, 
	                    steps_per_epoch=np.ceil(len(train_samples)/batch_size), 
	                    validation_data=validation_generator, 
	                    validation_steps=np.ceil(len(validation_samples)/batch_size), 
	                    callbacks=[checkpoint, stopper],
	                    epochs=num_epochs,
	                    verbose=1)



	print(history.history.keys())
	### plot the training and validation loss for each epoch
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	#plt.show()

	# # working:
	# model.fit_generator(train_generator, 
	#                     steps_per_epoch=np.ceil(len(train_samples)/batch_size), 
	#                     validation_data=validation_generator, 
	#                     validation_steps=np.ceil(len(validation_samples)/batch_size), 
	#                     epochs=5, 
	#                     verbose=1)

	# model.save('model.h5')


	# # Note: we aren't using callbacks here since we only are using 5 epochs to conserve GPU time
	# model.fit_generator(datagen.flow(X_train, y_one_hot_train, batch_size=batch_size), 
	#                     steps_per_epoch=len(X_train)/batch_size, epochs=epochs, verbose=1, 
	#                     validation_data=val_datagen.flow(X_val, y_one_hot_val, batch_size=batch_size),
	#                     validation_steps=len(X_val)/batch_size)



def main():
	
	train_samples, validation_samples = read_csv()
	check_gpu_status()
	model = def_model()
	train_generator, validation_generator = prepare_data(train_samples, validation_samples)
	train_model(model, train_samples, validation_samples, train_generator, validation_generator)



if __name__ == "__main__":
    main()