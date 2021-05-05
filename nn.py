import numpy as np 
import matplotlib.pyplot as plt
import optimizers
from math import sqrt

activations = {'ReLu':lambda x: np.maximum(0,x),'sigmoid':lambda x: 1.0/(1.0 + np.exp(-x)) }
np.random.seed(0)


"""
TODO:
- Add dropout.
- Add metrics and visualization.
"""
class NeuralNetwork:
    """
    Implements a Neural Network.
    Args:
        - sizes: A list containing number of neurons for each layer (including input and output    layer).
        - iterations: Number of epochs for the neural to train.
        - reg: L2 Regulaizer term for avoiding overfitting. Defaults to 1e-3.
        - step_size: The learning rate for the neural network.
        - f: Activation function for neurons.
        - optimizer: Optimization algorithm. NOTE: It encapsulates the REGULAIZER term and LEARNING RATE.
        - debug_loss: Boolean flag to show improvements during learning or no.
    """
    def __init__(self, sizes, iterations = 10000, f='ReLu', optimizer=optimizers.VanillaOptimizer(),debug_loss=False):
        self.sizes = sizes
        self.network, self.biases = self._initialize_network()
        self.f, self.f_name = activations[f], f
        self.iterations = iterations
        self.optimizer = optimizer
        self.reg, self.step_size = self.optimizer.get_learning_params()['reg'],self.optimizer.get_learning_params()['step_size']
        self.debug_loss = debug_loss

    def _initialize_network(self):
        """ Initalizes the network weights according to self.sizes with multivariate gaussian distribution."""
        layers = []
        biases = []
        for i,size in enumerate(self.sizes[1:]):
            # To calibrate the variance of each neuron we divide by the square root of number of inputs.
            layer = 0.01 * np.random.randn(self.sizes[i],size) / sqrt(self.sizes[i])
            layers.append(layer)
            biases.append(np.zeros((1,size)))
        return layers, biases

    def _feed_forward(self, inputs):
        """ Compute the neural network on the input.
           The input variable is assumed to be an ndarray variable."""
        assert inputs.shape[1] == self.sizes[0], "Invalid input size for network."
        out = inputs
        temps = []
        for i in range(len(self.network)-1):
            out = self.f(np.dot(out,self.network[i]) + self.biases[i])
            temps.append(out)
        out = np.dot(out,self.network[-1]) + self.biases[-1]
        return out, temps

    def _gradient(self,output,X,y):
        """ Compute gradient of cross-entropy loss with respect to scores."""
        m = X.shape[0]
        grad = output
        grad[range(m),y] -= 1
        grad /= m
        return grad

    def _backpropagate(self, X,layers,grad):
        """ Function implements backpropagation method: given the gradient with respect to scores it propogates the gradient backwards
            through the network by chain rule (basically multiplying) while applying ReLu derivative.
        """
        dWs = [] # gradients for weights
        dbs = [] # gradients for biases 
        prev = grad
        for i in range(len(layers)):
            dWs.append(np.dot(layers[::-1][i].T, prev))
            dbs.append(np.sum(prev,axis=0,keepdims=True))
            dlayer = np.dot(prev, self.network[::-1][i].T)
            if self.f_name == 'ReLu':
                dlayer[layers[::-1][i] <= 0] = 0 # The derivative of ReLu
            else:
                dlayer = self.f(dlayer) * (1- self.f(dlayer)) # Derivative of sigmoid. NOTE: It can cause vanished gradients.
            prev = dlayer 
        dWs.append(np.dot(X.T,prev))
        dbs.append(np.sum(prev,axis=0,keepdims=True))
        return dWs, dbs
        

    def fit(self,X,y, plot=False):
        ls = []
        m = X.shape[0]
        for i in range(self.iterations):
            scores, layers = self._feed_forward(X)

            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            correct_logprobs = -np.log(probs[range(m),y])
            data_loss = np.sum(correct_logprobs)/m
            reg_loss = sum([0.5*self.reg*np.sum(W*W) for W in self.network])
            loss = data_loss + reg_loss
       
            if i % 1000 == 0 and self.debug_loss:
                print ("Iteration {}: loss {}".format(i,loss))

            if i % (self.iterations/100) == 0:
                ls.append(loss)
            # Gradient of cross-entropy loss.
            grad = self._gradient(probs,X,y)

            dWs, dbs = self._backpropagate(X,layers,grad)

            for i,dw in enumerate(dWs[::-1]):
                dw = dw + self.reg * dw
                self.network[i] = self.network[i] - self.step_size * dw
                self.biases[i] = self.biases[i] - self.step_size * dbs[::-1][i]
        
        if plot:
            epochs = np.linspace(0,self.iterations,100)
            plt.plot(epochs,ls)
            plt.title('Accuracy vs epochs')
            plt.show()





    
