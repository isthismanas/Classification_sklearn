import numpy as np

class AdalineGD:

    def __init__(self, eta = 0.01 , n_iters=50 , RandomState = 1 ):
        self.eta = eta
        self.n_iters = n_iters
        self.RandomState = RandomState

    def fit(self , x, y):
        rgen = np.random.RandomState(self.RandomState)
        self.w_ = rgen.normal(loc = 0.0 , scale = 0.01, size = x.shape[1])
        self.b_ = np.float(0.0)
        self.losses_ = []

        for _ in range(self.n_iters):
            net_input = self.net_input(x)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * x.T.dot(errors) / x.shape[1]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, x):
        return np.dot(x , self.w_ + self.b_)
    

    def activation(self , x):
        return x
    
    def predict(self, x):
        return np.where(self.activation(self.net_input(x) > 0.5 , 1, 0))



