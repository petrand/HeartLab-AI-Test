#!/usr/bin/env python3

# Imports!
import os
import random
import sys
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import model

from tensorflow.keras.utils import to_categorical 

# importing testing files 
sys.path.insert(1,'tests')
print(sys.path)

import preprocessing_tests
import model_tests

# Config and Inits
data_dir = "chest_xray"
size = (256, 256)
# Ensuring reproducible results
np.random.seed(42)

# Fucntions!
def img_2_arr(
    img_path: str,
    resize: bool = False,
    grayscale: bool = True,
    size: tuple = (256, 256),
) -> np.ndarray:

    """
    This function is responsible for opening an image, Preprocessing
    it by color or size and returning a numpy array.

    Input:
        - img_path: str, a path to the location of a image file on disk
        - resize: bool, True/False if the image is to be resized
        - grayscale: bool, True/False if image is meant to be B&W or color
        - size: tuple, a 2d tuple containing the x/y size of the image.

    Output:
        - a np.ndarray which is assosiated to the image that was input.
    """

    if grayscale:
        img_arr = cv2.imread(img_path, 0)
    else:
        img_arr = cv2.imread(img_path)

    if resize:
        img_arr = cv2.resize(img_arr, size)

    return img_arr


def create_datasets(data_dir: str) -> np.ndarray:
    """
    This function is responsible for creating a dataset which
    contains all images and their associated class.

    Inputs:
        - data_dir: str, which is the location where the chest x-rays are
            located.

    Outputs:
        - a np.ndarray which contains the processed image, and the class
            int, associated with that class.

    """
    # Image Loading and Preprocessing
    all_normal_img_paths = []
    all_viral_img_paths = []
    all_bact_img_paths = []
    # mac creates .DS_Store files which break original code
    directories = ["NORMAL", "PNEUMONIA"]
    for cls in directories: # NORMAL or PNEUMONIA
        for img in os.listdir(os.path.join(data_dir, cls)): # all images
            if cls == "NORMAL":
                all_normal_img_paths.append(os.path.join(data_dir, cls, img))
            elif "virus" in img:
                all_viral_img_paths.append(os.path.join(data_dir, cls, img))
            else:
                all_bact_img_paths.append(os.path.join(data_dir, cls, img))

    # 0 for normal, 1 for bacterial and 2 for viral
    dataset = (
        [
            [img_2_arr(path, grayscale=True, resize=True, size=size), 0]
            for path in all_normal_img_paths
        ]
        + [
            [img_2_arr(path, grayscale=True, resize=True, size=size), 1]
            for path in all_bact_img_paths
        ]
        + [
            [img_2_arr(path, grayscale=True, resize=True, size=size), 2]
            for path in all_viral_img_paths
        ]
    )

    return np.array(dataset, dtype="object")

def get_counts(dataset):
    """
    This function counts the distribution of classes in the dataset
    Input:
        - dataset = dataset returned by the create_dataset function
    Output:
        - list of length 3 with index 0 being natural counts,
          index 1 being bacterical counts, index 2 being viral counts
    """
    imageCount = [0,0,0]
    for i in dataset:
        imageCount[i[1]] += 1
    return imageCount

def train_val_test_split(dataset, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    """
    This function shuffles data, splits into train, validation and test sets with respect to provided proportions
    Input: 
        - dataset = dataset returned by the create_datasets function
        - train prop = what proportion of data to dedicate for training set
        - validation prop = what proportion to dedicate for validation set
        - test prop = what proportion of data to dedicate for test set
    Output:
        - training, validation and test sets subsetted from the dataset
    """
    # testing that input makes sense 
    assert train_prop+val_prop+test_prop == 1, "Train, validation and test set proportions do not add up to 1"
    # reshuffle the array to make sure we are sampling only one class
    np.random.shuffle(dataset)
    n = len(dataset)
    # slicing to get the sets
    train_set = dataset[0:int(n*train_prop)]
    val_set = dataset[int(n*train_prop+1):int(n*(train_prop+val_prop))]
    test_set = dataset[int(n*(train_prop+val_prop)+1):n]
    
    return(train_set, val_set, test_set)

def distribution_plot(train_set, val_set, test_set, distribution_plot = "evaluation_results/distribution_plot.png"):
    n = 3
    bar1 = (get_counts(train_set)[0], get_counts(val_set)[0], get_counts(test_set)[0])
    bar2 = (get_counts(train_set)[1], get_counts(val_set)[1], get_counts(test_set)[1])
    bar3 = (get_counts(train_set)[2], get_counts(val_set)[2], get_counts(test_set)[2])

    ind = np.arange(n)
    plt.figure(figsize=(10,5))
    width = 0.3 
    plt.bar(ind, bar1 , width, label='Normal')
    plt.bar(ind + width, bar2, width, label='Bacterial')
    plt.bar(ind + width*2, bar3, width, label='Viral')

    plt.xlabel('Set types')
    plt.ylabel('Class frequency count')
    plt.title('Class frequency count in each set')

    plt.xticks(ind + width / 2, ('Train set', 'Validation set', 'Test set'))

    plt.legend(loc='best')
    plt.savefig(distribution_plot)


def X_y_split(data):
    """
    This function splits the features from the labels, normalises and reshapes features 
    in right format for CNN. Labels get one-hot encoding for 3 classes.
    Inputs: 
        - data = dataset returned by train_val_test_split function
    Outputs:
        - X = normalised features of the input dataset
        - y_cat = one hot encoded labels of input dataset
    """
    # Normalising the data
    X = np.array([i[0] for i in data])/255
    # Reshaping to add one "color" dimention for CNN
    X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
    # retrieving labels
    y = np.array([i[1] for i in data])
    # transforming labels to one-hot-encoding
    y_cat = to_categorical(y, 3)

    return X, y_cat




def main():

    # get dataset
    dataset = create_datasets(data_dir)
    # get train, val, test split
    train_set, val_set, test_set = train_val_test_split(dataset)
    
    # completing distribution test
    print(preprocessing_tests.distribution(dataset, train_set, val_set, test_set))

    # distribution plot to check if class distribution is skewed
    distribution_plot(train_set, val_set, test_set)
    
    # split data and labels, preprocess it
    X_train, y_cat_train = X_y_split(train_set)
    X_val, y_cat_val = X_y_split(val_set)
    X_test, y_cat_test = X_y_split(test_set)
    
    # completing normalisation test
    for i in list(map(preprocessing_tests.normalisation, [X_train, X_test, X_val])):
        print(i)
    # completing dimension test
    for i in list(map(preprocessing_tests.dimension, [(X_train, y_cat_train), (X_test, y_cat_test), (X_val, y_cat_val)])):
        print(i)

    # calling a model
    cnn_model = model.cnn_model()
    print(cnn_model.summary())

    # training the model
    trained_model = model.train_model(cnn_model, X_train, y_cat_train, X_val, y_cat_val)  
    
    # saving the model 
    trained_model.save('saved_model/cnn_model')
    
    # retrieving stored model
    #trained_model = tf.keras.models.load_model('saved_model/cnn_model')

    # evaluating the model
    model.evaluate_model(trained_model,X_test,y_cat_test,test_set,loaded_model=True)

    # completing test to check whether model can handle noise
    print(model_tests.noise_rotation_test(X_test, y_cat_test, test_set, trained_model))


if __name__ == "__main__":

    main()
