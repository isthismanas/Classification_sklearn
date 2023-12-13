import numpy as np

class AdalineSGD:
    def __init__(self, eta = 0.01 , n_iters = 20 , shuffle = True , random_state = None):
        self.eta = eta
        self.n_iters = n_iters
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, x, y):
        self._initialize_weights(x.shape[1])
        self.losses_ = []
        for i in range(self.n_iters):
            if self.shuffle:
                x , y = self._shuffle(x,y)

            losses = []
            for xi , target in zip(x, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self

    def partial_fit(self, x, y):
        if not self.w_initialized:
            self._initialize_weights(x.shape[1])
        if y.ravel().shape[0] >1:
            for xi, target in zip(x,y):
                self._update_weights(xi , target)
            else :
                self._update_weights(x,y)
        
        return self

    def shuffle(self, x , y):
        r = self.rgen.permutation(len(y))
        return x[r], y[r]

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc = 0.0 , scale = 0.01, size = m)
        self.b_ = np.float(0.)
        self.w_initialized = True                               