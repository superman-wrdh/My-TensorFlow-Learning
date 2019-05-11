import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-7, 7, 100)


def sigmoid(input):
    # f(x) = 1/(1 + e**(-x))
    y = [1 / float(1 + np.exp(-x)) for x in input]
    return y
