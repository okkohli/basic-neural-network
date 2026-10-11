import numpy as np

class Activation:

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def relu(X):
        return np.maximum(0, X) # its an element-wise operation, so we can use np.maximum to apply it to the entire array at once   

    # @staticmethod
    # def softmax(X):
    #     exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))  # Subtract max for numerical stability
    #     return exp_X / np.sum(exp_X, axis=1, keepdims=True)
        #

    @staticmethod
    def softmax(Z):
        Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        return A
    

    @staticmethod
    def tanh(X):
        return np.tanh(X)
