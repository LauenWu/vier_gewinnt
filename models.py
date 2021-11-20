import tensorflow as tf
import numpy as np

def to_categorical(a):
    b = np.zeros((n,m,3))
    for i in range(n):
        for j in range(m):
            b[i,j,int(a[i,j])+1] = 1
    return b

class PolicyNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.l1 = tf.keras.layers.Conv2D(
            filters=3, kernel_size=(4,4), input_shape=(6,7,3), activation='relu')
        self.l2 = tf.keras.layers.Flatten()
        self.l3 = tf.keras.layers.Dense(7, activation='softmax')

    def call(self, input:np.array):
        x = tf.convert_to_tensor(input)
        x = self.l1(x)
        x = self.l2(x)
        return self.l3(x)

    def loss(self, prediction, actual, reward):
        cce = tf.keras.losses.CategoricalCrossentropy()
        return -cce(actual, prediction)*reward

    def train(self, q):
        for state, action, reward in q.iteritems():
        pass

class ValueNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.l1 = tf.keras.layers.Conv2D(
            filters=3, kernel_size=(4,4), input_shape=(6,7,3), activation='relu')
        self.l2 = tf.keras.layers.Flatten()
        self.l3 = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, input:np.array):
        x = tf.convert_to_tensor(input)
        x = self.l1(x)
        x = self.l2(x)
        return self.l3(x)