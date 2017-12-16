import os
import csv
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers import Convolution2D, Cropping2D, MaxPooling2D

'''
read into the data
'''
samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # create adjusted steering measurements for the side camera images
        steering_center = float(line[3])     
        correction = 0.2
        
        #saperately append center, left and right image path/angle
        samples.append([line[0],steering_center])
        samples.append([line[1],steering_center+ correction])
        samples.append([line[2],steering_center- correction])

'''
extract 20% data as the validation data, the rest are training data
'''
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


'''
flip every image to augment the data
'''       
def augmentation(image, angle):
    image_flipped = np.fliplr(image)
    angle_flipped = -angle  
    return image_flipped, angle_flipped

'''
use generator to avoid occupy the large memory all in onece
read the images into train/valid samples.
Note: because of the use of flip in every interate, the actual batch_size is two times of the batch_size transferred into the generator.
and the total sample numbers should aslo mutiplied by 2 in model.fit_generator()
eg. set batch_size to 32, then the actual batch_size should be 64. 

'''
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)     
        for offset in range(0, num_samples, batch_size):
            print(offset)
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                image = cv2.imread('./IMG/'+batch_sample[0].split('\\')[-1])
                angle = float(batch_sample[1])
                image_flipped,angle_flipped = augmentation(image, angle)
                images.append(image)
                angles.append(angle)
                images.append(image_flipped)
                angles.append(angle_flipped)                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=len(validation_samples))

'''
use NVIDA model in the program.
'''

#resize the image to [66,200] to fit the orginal NVIDA model
def resize(img):
    from keras.backend import tf
    return tf.image.resize_images(img, [66, 200])

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Lambda(resize))

# 3 convolutional layers with kernal= 5x5, stride= 2x2
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))  
# 2 convolutional layers with kernal = 3x3, stride=1x1
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu')) 
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu')) 
# Fully connected layers
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
# Output layer
model.add(Dense(1))

model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0001))
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*2, validation_data=validation_generator, nb_val_samples=len(validation_samples)*2,nb_epoch=5)
model.save('model_last.h5')

'''
print the keys contained in the history object
plot the training and validation loss for each epoch
'''
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
