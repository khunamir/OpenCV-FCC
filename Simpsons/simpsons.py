# !pip install --upgrade caer canaro

import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
import matplotlib.pyplot as plt
import sklearn.model_selection as skm 

IMAGE_SIZE = 80,80
channels = 1
char_path = r'../input/the-simpsons-characters-dataset/simpsons_dataset'

# Creating a character dictionary, sorting it in descending order
char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path,char)))

# Sort in descending order
char_dict = caer.sort_dict(char_dict, descending=True)
char_dict

#  Getting the first 10 categories with the most number of images
characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break
characters

train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMAGE_SIZE, isShuffle=True)

len(train)

plt.figure(figsize=(30,30))
plt.imshow(train[0][0], cmap='gray')
plt.show()

featureSet, labels = caer.sep_train(train, IMG_SIZE=IMAGE_SIZE)

from tensorflow.keras.utils import to_categorical

featureSet = caer.normalize(featureSet)
labels = to_categorical(labels, len(characters))

split_data = skm.train_test_split(featureSet, labels, test_size=0.2)
x_train, x_val, y_train, y_val = (np.array(item) for item in split_data)

del train
del featureSet
del labels
gc.collect()

BATCH_SIZE = 32
EPOCHS = 10

datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

model = canaro.models.createSimpsonsModel(IMG_SIZE=IMAGE_SIZE, channels=channels,
                                          output_dim=len(characters), loss='binary_crossentropy', 
                                          decay=1e-7, learning_rate=0.01, momentum=0.9,
                                          nesterov=True)

model.summary()

from tensorflow.keras.callbacks import LearningRateScheduler

# Training the model
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]

training = model.fit(train_gen,
                     steps_per_epoch = len(x_train)//BATCH_SIZE,
                     epochs = EPOCHS,
                     validation_data = (x_val, y_val),
                     validation_steps = len(y_val)//BATCH_SIZE,
                     callbacks = callbacks_list)

test_path = r'../input/the-simpsons-characters-dataset/kaggle_simpson_testset/kaggle_simpson_testset/bart_simpson_20.jpg'

img = cv.imread(test_path)

plt.imshow(img, cmap='gray')
plt.show()

def reshape(x, IMG_SIZE, channels):
    width, height = IMG_SIZE[:2]
    return np.array(x).reshape(-1, width, height, channels)

def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, IMAGE_SIZE)
    img = reshape(img, IMAGE_SIZE, 1)
    return img

predictions = model.predict(prepare(img))

print(characters[np.argmax(predictions[0])])