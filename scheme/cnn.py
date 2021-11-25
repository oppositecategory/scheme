import numpy as np 

class ConvLayer:
  def __init__(self, 
               params, 
               vect_flag = True):
    """
    Implements simple ReLu Convolutional layer.

    Args:
      params: ndarray of the filters parameters. Each entry in params is a 3-tuple (F,S,P).
              F - The dimension of the array. For example F = 5 means a filter of size 
                     5x5xD .
              S - Stride. How to "slide" the kernel across the image. 
              P - The length of the padding applied to the image.
      vect_flag: A boolean flag indicating for a vectorized ConvLayer meaning we have a stack of
                 N filters to be learned with same dimension and stride to be transformed into
                 matrix operation. vect_flag defaults to True to have a more compact and efficient 
                 implementation.
                 NOTE: If vect_flag == True we expect params to be of the form [(F,S),K].
    """
    self.vect_flag = vect_flag 

    if self.vect_flag == True:
      self.params = params[0]
      self.K = params[1]
    else:
      self.params = params
      self.K = len(params)

    self.D = None # We must have the depth dimension for the filters.
    self.filters = None # K filters each of size FxFxD, which act as the weights of the CNN
    self.cache_gradient = None
  
  def set_depth(self, D):
    self.D = D 
    self.filters = self._generate_filters()
  
  def _generate_filters(self):
    filters = []
    if not self.vect_flag:
      for param in self.params:
        filters.append(np.random.normal(size=(param[0],param[0],self.D)))
    else:
      (F,_) = self.params
      for i in range(self.K):
        filters.append(np.random.normal(size=(F,F,self.D)))
    return np.array(filters)

  def expose_weights(self):
    return np.array([w.flatten() for w in self.filters])

  def convolve(self, X, kernel, stride,padding):
    """ Naive multi-dimensional convolutiion implementation. 

        Args:
          X : ndarray containing the input we convolve over.
          kernel : ndarray of the kernel we convolve across X.
          stride : the stride to slide the kernel across X. 
          padding : the length of the image padding.
    """
    F = kernel.shape[0]
    W, H, _ = X.shape 

    W1 = (W - F + 2*padding)/stride + 1 
    H1 = (H - F + 2*padding)/stride + 1

    # Arithmetic returns a floating point number. We validate it's a whole number
    # before converting to int.
    assert W1.is_integer(), "Incompatible parameters."
    assert H1.is_integer(), "Incompatible parameters."

    W1, H1 = int(W), int(H)
    activation_map = np.zeros((W,H,self.K))
    for depth in range(self.D):
      for i in range(W1):
        for j in range(H1):
          activation_map[i,j,depth] = np.sum(self.X[(i*stride):(F+i*stride), :F, depth] * kernel)
    return activation_map 

  def naive_forward_pass(self, X):
    # Naive implementation of the forward pass of a convolutional layer. 
    assert self.D, "Depth dimension was not defined."
    W, H, D = X.shape 

    output = []
    for i,(F,S,P) in enumerate(self.params):
      activation_map = self.convolve(X=self.X,
                                          kernel=filters[i],
                                          stride=S,
                                          padding=P)
      output.append(activation_map)
    
    output = np.array(output)
    output[output < 0] = 0
    return  output

  def im2col(self, X):
    """ Vectorize the activation map. We stretch each local region of X (defined by the 
        dimensions of kernel) into a column vector. We calculate the number of such local
        regions by the simple formuals for the width and height of the activation map and store
        each local region as a column in a matrix so we get a matrix with dimnesions:
        (F*F*D)x(W*H).
    """
    assert self.D, "Depth dimension was not defined"
    assert self.vect_flag, "ConvLayer instance isn't built for vectorized operation."
    (F,S) = self.params
    W, H, _ = X.shape 
    W1 = (W - F)/S + 1 
    H1 = (H - F)/S + 1

    assert W1.is_integer(), "Incompatible parameters."
    assert H1.is_integer(), "Incompatible parameters."

    W1, H1 = int(W1), int(H1)

    M = [] 
    for i in range(W1):
      for j in range(H1):
        M.append(X[(i*S):(F+i*S), :F, :].flatten())
    return np.array(M).T, W1,H1
  
  def feed_forward(self, X):
    """ Efficient, vectorized, implementation of feed forward in the ConvLayer.
        NOTE: The method assumes we have N filters of the SAME dimension and stride. Then we can vectorize both the activation
        map using the im2col method and then vectorize the weights to be a matrix of size Kx(F*F*D),
        i.e number of filters as rows and size of the filter as columns giving us a matrix with each row 
        as a filter to be dot product with the vectorized input.

        Args:
          reshape = A boolean flag to return the output of the layer as FxFxK (i.e a stack of filters)
                    or leave it at intermediate matrix form. Defaults to True.
    """ 
    assert self.vect_flag, "ConvLayer initialized as not compatible for vectorized feed forward."
    vect_input,W,H = self.im2col(X)
    vect_weights = np.array([w.flatten() for w in self.filters])
    activation_maps = np.dot(vect_weights,vect_input).reshape(W,H,self.K)
    self.cache_gradient = np.where(activation_maps <0)
    activation_maps[activation_maps < 0] = 0
    return activation_maps

class MaxPoolingLayer:
  def __init__(self, F=2, S=2):
    """ Implements a MaxPooling layer to reduce amount of parameters and control model capacity
        (and by doing so prevent overfitting).

        Args:
            F: the dimension of the MaxPooling filter.
            S: the stride.
    """
    self.F = F
    self.S = S
    self.cache_gradient = None 
  
  def feed_forward(self,X):
    W, H, D = X.shape 


    W1 = (W - self.F)/self.S + 1 
    H1 = (H - self.F)/self.S + 1

    # Arithmetic returns a floating point number. As a stupid check to ensure the dimensions
    # are correct we check W and H are indeed integers by abusing Python ==.
    assert W1.is_integer(), "Incompatible parameters."
    assert H1.is_integer(), "Incompatible parameters."

    W1, H1 = int(W1), int(H1)
    downsample = np.zeros((W1,H1,D))
    # We will create a matrix mask so we will keep track of only indices which are maximal 
    # in their local regions.
    self.gradient_cache = np.zeros((W1,H1,D),dtype=np.int8)

    for depth in range(D):
      for i in range(W1):
        for j in range(H1):
          index = np.argmax(X[(i*self.S):(self.F+i*self.S), :self.F,depth])
          i1, j1 = index // self.F , j1 % self.F 
          # Raise a flag in the indices of the max value
          self.gradient_cache[(i*self.S) + i1,j1,depth] = 1 
          downsample[i,j,depth] = np.max(X[(i*self.S):(self.F+i*self.S), :self.F,depth])
    return downsample

  def propagate_backwards(self, output):
    output[self.gradient_cache == 0] = 0
    return output 

  
class FullyConnected:
  def __init__(self, input_shape, num_neurons=4096):
    """ Implements a fully connected layer. 
    """
    self.num_neurons = num_neurons 
    self.input_shape = input_shape 

    (W, H, D) = self.input_shape
    self.weights = np.random.normal(size=(self.num_neurons, W*H*D)) / np.sqrt(2.0/(W*H*D))
    self.gradient_cache = None
  
  def expose_weights(self):
    return self.weights

  def feed_forward(self, X):
    X = X.flatten()
    # Each row in weights corresponds to all the weights of a specific neuron and thus 
    # each entry in the matrix multiplication weights @ X stands for a dot product
    # (w_i * x_i) where x_i is a neuron in the input and eventually ontop of the dot-product 
    # we compose as a non-linearity the ReLu.
    activations = np.dot(self.weights, X)
    self.gradient_cache = np.where(activations < 0)
    activations[activations < 0] = 0
    return activations

  def propagate_backwards(self, output):
    dlayer = np.dot(prev, self.weights.T) 
    dactivation = dlayer
    dactivation[sel.gradient_cache] = 0
    return dactivation, dlayer 

def cross_entropy_gradient(output, y, m):
  grad = output 
  grad[range(m), y] -= 1 
  grad /= m
  return grad
  
class CNN:
  def __init__(self, 
               layers,
               iterations=1000, 
               reg = 0.1):
    """
    Implements a Convolutional Neural Network.

    Args:
      X: ndarray input for the CNN.
      layers: list of MaxPoolingLayers/ConvLayers.
    """
    self.layers = layers 
    self.N = len(self.layers) - 1
    self.gradient_cache = []

    self.iterations = iterations
    self.reg = reg 

  def feed_forward(self,X):
    output = X
    for layer in self.layers:
      if isinstance(layer, ConvLayer):
        # Set the ConvLayer to work with the appropiate depth dimension for the filters.
        # NOTE: The depth dimension can be changed throughout the network 
        # due to a ConvLayer's number of filters hence it shall be changed 
        # accordingly throughout the network.
        layer.set_depth(output.shape[2])
      output = layer.feed_forward(output)
    return output
    
  def backpropagate(self, X, output):
    """
    Perform backpropagation on the whole network. 

    Args:
      X : ndarray object which is input to the network
      output: the derivative of the cross-entropy loss with respect to the scores.
    
    returns:
      dWs = array of the gradients of the network with respect to each layer and it's weights.
    """
    dWs = []
    prev = output 
    for layer in self.layers[::-1]:
      prev, grad_w = layer.propagate_backwards(prev)
      dWs.append(grad_w)
    dWs.append(np.dot(X.T,prev))
    return dWs 

  def aggregate_weights(self):
    network = []
    for layer in self.layers:
      if isinstance(layer, ConvLayer) or isinstance(layer, FullyConnected):
        network.append(layer.expose_weights())
    return np.array(network)

  def fit(self,X, y):
    m = len(X)
    for i in range(self.iterations):
      scores = self.feed_forward(X)
      network = self.aggregate_weights()

      # We compute probabilities by applying SoftMax function on the output layer
      exp_scores = np.exp(scores)
      probs = exp_scores / np.sum(exp_scores,axis=1, keepdims=True)

      # As a loss function we use cross entropy to make small probabilities assigned
      # that was assigned to correct labels contribute more to the loss function.
      correct_logprobs = - np.log(probs[range(m), y])
      data_loss = np.sum(correct_logprobs)/m
      reg_loss = sum([0.5*self.reg*np.sum(W*W) for W in network])
      loss = data_loss + reg_loss 

      # Compute the gradient of cross-entropy with respect to scores 
      grad = cross_entropy_gradient(probs, y, m)

      dWs = self.backpropagate(X, network, grad)