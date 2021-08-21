import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report,confusion_matrix

def noise_rotation_test(X_test, y_cat_test, test_set, model, class_report = "evaluation_results/noisy_classification_report.txt"):

    image_gen = ImageDataGenerator(rotation_range=10, # rotate the image 20 degrees
                                    width_shift_range=0.05, # Shift the pic width by a max of 2.5%
                                    height_shift_range=0.05, # Shift the pic height by a max of 2.5%
                                    rescale=1, # Rescale the image by normalzing it.
                                    zoom_range=0.05, # Zoom in by 5% max
                                    horizontal_flip=False, # Allow horizontal flipping
                                    fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )
    # image before
    plt.imshow(np.squeeze(X_test[0]))

    # apply noise to all images
    for i in range(0,len(X_test)):
        X_test[i] = image_gen.random_transform(X_test[i])
    
    # image after 
    plt.imshow(np.squeeze(X_test[0]))

    preds = model.predict(X_test)
    predictions = np.argmax(preds,axis=1)
    y_test = np.array([i[1] for i in test_set])

    with open(class_report, 'w') as f:
        print(classification_report(y_test,predictions))
        f.write(classification_report(y_test,predictions))

    metrics = model.evaluate(X_test,y_cat_test,verbose=0)
    assert (metrics[1] > 0.7), "FAILED TEST: Model cannot handle noise, accuracy drops below 0.7"

    return "PASSED TEST: Model can handle rotational and shift changes"