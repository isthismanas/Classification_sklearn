import numpy as np


class Perceptron:

    def __init__(self , eta = 0.01 , n_iters = 50 , random_state = 1):

        self.n_iters =  n_iters
        self.eta = eta
        self.random_state = random_state


    def fit(self , x ,y):

        rgen = np.random.default_rng(self.random_state)
        self.w_ = rgen.standard_normal(loc= 0.0, scale = 0.01 , size= x.shape[1])
        self.b_ = np.float_(0.)
        self.errors = []

        for i in range(self.n_iters):
            errors = 0
            for xi , target in zip(x,y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update!=0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, x):
        return np.dot(x , self.w_) + self.b_
    
    def predict(self , x):
        return np.where(self.net_input(x)>= 0 , 1 ,0)