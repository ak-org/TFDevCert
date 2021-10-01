"""
Warm-up Question

You are given two arrays (features and targets) as a dataset. If you look closer at values, 
you'll notice that those replicate the classical square function.

Create a neural network that will learn this function.

To test your model, you can call the predict method on it. For example model.predict(10.0)

"""


import numpy as np 
import tensorflow as tf 

def solution_model():
    features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    target = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    model = tf.keras.Sequential([
        ### YOUR CODE HERE
    ])
    
    return model 

if __name__ == "__main__":
    model = solution_model()
    model.save('model.h5')
    print(model.predict(10.0))

