import numpy as np
import nn
import optimizers


step_sizes = [1e-0,1e-1,1e-2,1e-3,1e-4,1e-5]

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



sizes = [2,100,3]
network = nn.NeuralNetwork(sizes, optimizer = optimizers.RMSpropOptimizer(decay_rate=0.9,step_size=1e-3),debug_loss=True)
network.fit(X,y,True)


"""
accs = []
for step in step_sizes:
    print('Training...')
    network= nn.NeuralNetwork(sizes=[2,100,3],step_size=step)
    network.fit(X,y)
    scores, temps = nn._feed_forward(X)
    predicted_class = np.argmax(scores, axis=1)
    accs.append(np.mean(predicted_class==y))
    print('Finished training a neural network.')

plt.plot(step_sizes, accs)
plt.title('Accuracy vs step_size')
plt.show()
"""