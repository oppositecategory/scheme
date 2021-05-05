import numpy as np


class VanillaOptimizer:
    """ Implements basic Gradient Descent method.
    """
    def __init__(self,reg=1e-3, step_size=1e-1):
        self.reg = reg
        self.step_size = step_size
    
    def optimize(self,dWs,dbs,weights, biases):
        for i,dw in enumerate(dWs[::-1]):
                dw = dw + self.reg * dw
                weights[i] = weights[i] - self.step_size * dw
                biases[i]  = biases[i]  - self.step_size * dbs[::-1][i]
        return weights, biases

    def get_learning_params(self):
        # Returns important learning params; regulaizer parameter and learning rate.
        return self.__dict__


class MomentumOptimizer(VanillaOptimizer):
    """ Implements Momentum method.
        Momentum accelerates SGD in the direction with most curvature and dampens oscilations. 
    """
    def __init__(self,mu=0.5,reg=1e-3,step_size=1e-1):
        super().__init__(reg,step_size)
        self.mu = mu

    def optimize(self, dWs, dbs, weights, biases):
        vs = [np.zeros_like(dw) for dw in dWs]
        vs1 = [np.zeros_like(db) for db in dbs]
        for i,dw in enumerate(dWs[::-1]):
                dw = dw + self.reg * dw
                vs[i]  = self.mu * vs[i] - self.step_size * dw 
                vs1[i] = self.mu * vs1[i] - self.step_size * dbs[::-1][i]
                weights[i] = weights[i] + vs[i] 
                biases[i]  = biases[i]  + vs1[i]
        return weights, biases

class RMSpropOptimizer(VanillaOptimizer):
    def __init__(self, decay_rate=0.99,reg=1e-3,step_size=1e-1):
        super().__init__(reg,step_size)
        self.decay_rate = decay_rate
        self.eps = 1e-4 
    
    def optimize(self,dWs,dbs,weights, biases):
        caches = [np.zeros_like(dw) for dw in dWs]
        caches1 = [np.zeros_like(db) for db in dbs]
        for i,dw in enumerate(dWs[::-1]):
                dw = dw + self.reg * dw
                caches[i] = self.decay_rate * caches[i] + (1 - self.decay_rate) * (dw ** 2)
                caches1[i] = self.decay_rate * caches1[i] + (1-self.decay_rate) * (dbs[::-1][i] ** 2)
                weights[i] = weights[i] - self.step_size * dx / (np.sqrt(caches[i]) + self.eps)
                biases[i]  = biases[i]  - self.step_size * dbs[::-1][i] / (np.sqrt(caches1[i]) + self.eps)
        return weights, biases
