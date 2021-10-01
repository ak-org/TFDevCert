"""
CIFAR 10

Create and train a classifier for the CIFAR 10 dataset 
The Neural Network should output 10 classifications and the input shape should be (32x32).

Info: https://keras.io/api/datasets/cifar10/

Note: 
Do not user Lambda layer. Using it will cause Google test to fail.
Be careful here, Google test might fail if the shape of Input layer is incorrect 

"""


import numpy as np 
import tensorflow as tf 

def solution_model():
    cifar10 = tf.keras.datasets.cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = np.array(X_train, np.float32)
    X_test = np.array(X_test, np.float32)

    ### YOUR CODE HERE
    
    return model 

if __name__ == "__main__":
    model = solution_model()
    model.save('cifar10.h5')