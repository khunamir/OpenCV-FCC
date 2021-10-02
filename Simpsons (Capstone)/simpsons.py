import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
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

# Error on kaggle
train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMAGE_SIZE, isShuffle=True)