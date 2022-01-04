import tensorflow as tf
import numpy as np
from .constants import *
import pandas as pd
import tensorflow_probability as tfp
from tensorflow import convert_to_tensor, float32, expand_dims
import os

# class PolicyNet(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         input_shape = (N,M,3)
#         self.l1 = tf.keras.layers.Conv2D(
#             filters=64, kernel_size=(X,X), input_shape=input_shape, activation='relu')
#         self.l2 = tf.keras.layers.Flatten()
#         self.l3 = tf.keras.layers.Dense(100, activation='sigmoid')
#         self.l4 = tf.keras.layers.Dense(M, activation='softmax')

#     def call(self, state) -> float: 
#         x = self.l1(state)
#         x = self.l2(x)
#         x = self.l3(x)
#         return self.l4(x)


# class PolicyNet(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         #input_shape = (N,M,3)
#         self.l1 = tf.keras.layers.Dense(N*M*3, activation='sigmoid')
#         self.l2 = tf.keras.layers.Dense(128, activation='sigmoid')
#         self.l3 = tf.keras.layers.Dense(128, activation='sigmoid')
#         self.l4 = tf.keras.layers.Dense(M, activation='softmax')

#     def call(self, state:np.array) -> float: 
#         x = expand_dims(convert_to_tensor(state, dtype=float32),0)
#         x = self.l1(x)
#         x = self.l2(x)
#         x = self.l3(x)
#         return self.l4(x)

class PolicyNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        input_shape = (N,M,3)
        self.l1 = tf.keras.layers.LocallyConnected2D(
            filters=512, kernel_size=(N,M), input_shape=input_shape, activation='relu')
        self.l2 = tf.keras.layers.Flatten()
        self.l3 = tf.keras.layers.Dense(256, activation='sigmoid')
        self.l4 = tf.keras.layers.Dense(M, activation='softmax')

    def call(self, state) -> float: 
        x = expand_dims(convert_to_tensor(state, dtype=float32),0)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return self.l4(x)

class ValueNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        input_shape = (N,M,2)
        self.l1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(X,X), input_shape=input_shape, activation='tanh')
        self.l2 = tf.keras.layers.Flatten()
        self.l3 = tf.keras.layers.Dense(128, activation='tanh')
        self.l3 = tf.keras.layers.Dense(128, activation='tanh')
        self.l4 = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, state) -> float: 
        x = self.l1(state)
        x = self.l2(x)
        x = self.l3(x)
        return self.l4(x)

# class ValueNet(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         #input_shape = (N,M,3)
#         self.l1 = tf.keras.layers.Dense(N*M*2, activation='tanh')
#         self.l2 = tf.keras.layers.Dense(128, activation='tanh')
#         self.l3 = tf.keras.layers.Dense(128, activation='tanh')
#         self.l4 = tf.keras.layers.Dense(1, activation='tanh')

#     def call(self, state:np.array) -> float: 
#         x = self.l1(state)
#         x = self.l2(x)
#         x = self.l3(x)
#         return self.l4(x)