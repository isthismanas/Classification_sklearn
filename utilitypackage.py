import numpy as np

def sigmoid(x)->float:
    return 1.0 /(1.0 +np.exp(-x))