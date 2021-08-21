#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix


def cnn_model(loss='categorical_crossentropy', opt='adam'):
    """
    Fitting a three layered CNN model with Convolutional, MaxPool and Dropout layers
    at each of the three layers. The convolutional layers are followed by a dense 256 layer.
    Model returns 3 classes 0=normal, 1=bacterial, 2=viral.
    """
    model = Sequential()

    ## Layer 1

    # Convolutional layer
    model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(256, 256, 1), activation='relu',))
    # Pooling layer
    model.add(MaxPool2D(pool_size=(2, 2)))
    # Dropout layer 
    model.add(Dropout(0.25))

    ## Layer 2

    model.add(Conv2D(filters=64, kernel_size=(4,4), activation='relu',))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(4,4), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Flatten images from 58 by 58 to 430592 list before final layer
    model.add(Flatten())

    # 256 Neurons in dense Layer
    model.add(Dense(256, activation='relu'))
    # Last Dropout
    model.add(Dropout(0.4))
    # Since there are 3 classes, we add 3 neurons
    model.add(Dense(3, activation='softmax'))

    model.compile(loss=loss,
                optimizer=opt,
                metrics=['accuracy'])

    return model



def train_model(model, X_train, y_cat_train, X_val, y_cat_val, patience=3, epochs=5):
    early_stop = EarlyStopping(monitor='val_loss',patience=patience)
    model.fit(X_train,y_cat_train,epochs=epochs,validation_data=(X_val,y_cat_val),callbacks=[early_stop])
    return model

def evaluate_model(model, X_test, y_cat_test, test_set, accuracy_plot = "evaluation_results/accuracy_plot.png", loss_plot="evaluation_results/loss_plot.png", class_report = "evaluation_results/classification_report.txt", conf_matrix ="evaluation_results/confusion_matrix.png"):

    losses = pd.DataFrame(model.history.history)
    losses[['accuracy','val_accuracy']].plot()
    plt.savefig(accuracy_plot)
    losses[['loss','val_loss']].plot()
    plt.savefig(loss_plot)

    y_test = np.array([i[1] for i in test_set])
    
    preds = model.predict(X_test)
    predictions = np.argmax(preds,axis=1)

    with open(class_report, 'w') as f:
        print(classification_report(y_test,predictions))
        f.write(classification_report(y_test,predictions))

    

    plt.figure(figsize=(10,6))
    sns.heatmap(confusion_matrix(y_test,predictions),annot=True)
    plt.savefig(conf_matrix)