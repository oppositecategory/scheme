import numpy as np 
from math import sqrt
import matplotlib.pyplot as plt

activations = {'ReLu':lambda x: np.maximum(0,x),'sigmoid':lambda x: 1.0/(1.0 + np.exp(-x)) }
np.random.seed(0)



"""
TODO:
- Add other optimizing method: at least Adam and RMSprop.
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
        - optimizer: Method to optimize the gradient.
        - debug_loss: Boolean flag to show improvements during learning or no.
    """
    def __init__(self, sizes, iterations = 10000,reg=1e-3, step_size=1e-0, f='ReLu', optimizer='GD',debug_loss=False):
        self.sizes = sizes
        self.network, self.biases = self._initialize_network()
        self.f= activations[f]
        self.iterations = iterations
        self.reg = reg
        self.step_size = step_size
        self.optimizer = optimizer
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
        """ Compute gradient of cross-entropy loss """
        m = X.shape[0]
        grad = output
        grad[range(m),y] -= 1
        grad /= m
        return grad

    def _backpropagate(self, X,layers,grad):
        """ Function implements backpropagation method: given the gradient variable it propogates the gradient backwards
            through the network by chain rule (basically multiplying) while applying ReLu derivative.
        """
        dWs = [] # gradients for weights
        dbs = [] # gradients for biases 
        prev = grad
        for i in range(len(layers)):
            dWs.append(np.dot(layers[::-1][i].T, prev))
            dbs.append(np.sum(prev,axis=0,keepdims=True))
            dlayer = np.dot(prev, self.network[::-1][i].T)
            dlayer[layers[::-1][i] <= 0] = 0 # The derivative of ReLu
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

            if i % 100 == 0:
                ls.append(loss)
            # Gradient of cross-entropy loss.
            grad = self._gradient(probs,X,y)

            dWs, dbs = self._backpropagate(X,layers,grad)

            for i,dw in enumerate(dWs[::-1]):
                dw = dw + self.reg * dw
                self.network[i] = self.network[i] - self.step_size * dw
                self.biases[i] = self.biases[i] - self.step_size * dbs[::-1][i]
        
        if plot:
            epochs = np.linspace(0,9000,100)
            plt.plot(epochs,ls)
            plt.title('Accuracy vs epochs')
            plt.show()


N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j


step_sizes = [1e-0,1e-1,1e-2,1e-3,1e-4,1e-5]

"""
accs = []
for step in step_sizes:
    print('Training...')
    nn = NeuralNetwork(sizes=[2,100,3],step_size=step)
    nn.fit(X,y)
    scores, temps = nn._feed_forward(X)
    predicted_class = np.argmax(scores, axis=1)
    accs.append(np.mean(predicted_class==y))
    print('Finished training a neural network.')

plt.plot(step_sizes, accs)
plt.title('Accuracy vs step_size')
plt.show()
"""

nn = NeuralNetwork([2,100,50,3],step_size=1e-1,debug_loss=False)
nn.fit(X,y)
    


    
