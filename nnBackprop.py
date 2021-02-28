import numpy as np
import matplotlib.pyplot as plt
from mnistunpacker import load_data_wrapper
import random
# i have hardcoded an assumption that the size of the batch divides the length of the training image set
#use a step to fix this

def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))

def sigmoidprime(z):
	return sigmoid(z)*(1 - sigmoid(z))

class Network(object):
	"""docstring for Network"""
	def __init__(self, sizes, learning_rate = 3.0):
		self.sizes = sizes
		self.learning_rate = learning_rate
		self.num_layers = len(sizes)
		#initialise weights and biases with randomised arrays
		self.weights = [np.random.randn(y, x) 
		                for x, y in zip(sizes[:-1], sizes[1:])]
		self.biases = [np.random.randn(x, 1)
					   for x in sizes[1:]]

	def feedforward(self, activation):
		#here our input is x
		#has no use until we have already a full trained network
		for weight, bias in zip(self.weights, self.biases):
			activation = sigmoid(np.dot(weight,activation) + bias)
		return activation

	def gradient_descent(self,training_data, epochs = 40, batch_size = 20, test_data = None, period = 10):
		'''training_data is a list of tuples, the input and expected output
		'''
		#first we shuffle the training data
		n = len(training_data)
		#these are our mini batches
		#next run backprop on each
		print("our initialreading is {0} / {1}".format(self.evaluate(test_data), len(test_data)))
		for epoch in range(epochs):
			random.shuffle(training_data)
			mini_batches = []
			for k in range(n // batch_size):
				mini_batches.append(training_data[batch_size*k : batch_size*(k + 1)])
			for mini_batch in mini_batches:
				self.mini_batch_updater(mini_batch)
			if test_data:
				try:
					print("Epoch {0}: {1} / {2}".format(epoch + 1, self.evaluate(test_data), len(test_data)))
				except:
					print('error in the testing procedure')

	def mini_batch_updater(self, mini_batch):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		#print(type(mini_batch))
		for x,y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			#add together nabla_b and delta_nabla_b
			nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw + dnw for nw, dnw in zip(nabla_w,delta_nabla_w)]
		self.weights = [w - (self.learning_rate/len(mini_batch))*nw
						for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b - (self.learning_rate/len(mini_batch))*nb 
						for b, nb in zip(self.biases,nabla_b)]


	def backprop(self, activation, expected_result):
		#we want to output nabla_b and nabla_w
		#we should make a nice batch of training examples
		#then caluclate the backprop on each of them 
		#and add all these backprops together at the end of the mini batch
		#and then we adjust the weights
		activations = [activation]
		z_activations = []

		for weight, bias in zip(self.weights, self.biases):
			z_activation = np.dot(weight, activation) + bias 
			z_activations.append(z_activation)
			activation = sigmoid(z_activation)
			activations.append(activation)
		delta_L = self.deriv_of_cost(activations[-1], expected_result) * sigmoidprime(z_activations[-1])
		#now we need to push the error back through the network
		#return (activations, z_activations)
		nabla_b = [delta_L]
		nabla_w = [np.dot(delta_L, activations[-2].transpose())]
		for index in range(2, self.num_layers):
			#print(self.weights[1-index].transpose().shape, error_vectors[1-index].shape, z_activations[1-index].shape)
			dummy_delta = np.dot(self.weights[1 - index].transpose(), nabla_b[1 - index]) \
									* sigmoidprime(z_activations[-index])
			nabla_b[:0] = [dummy_delta]
			nabla_w[:0] = [np.dot(dummy_delta, activations[-1-index].transpose())]
		#print([element.shape for element in activations])
		#print([element.shape for element in z_activations])
		return (nabla_b, nabla_w)
		#at this stage all the shapes match 

	def deriv_of_cost(self, output_activations, expected_activations):
		return (output_activations - expected_activations)

	def evaluate(self,test_data):
		test_results = [(np.argmax(self.feedforward(x)), y)
						for (x, y) in test_data]
		return sum(int(x == y) for (x,y) in test_results)


c = Network([784,30, 10])
'''
a = np.random.randn(784,1)
b = np.random.randn(10,1)
x = c.backprop(a,b)
#print([element.shape for element in c.weights])
'''
total_data = load_data_wrapper()
training_data = [(c,d) for c,d in total_data[0]]
validation_data = [(c,d) for c,d in total_data[1]]
test_data = [(c,d) for c,d in total_data[2]] 

#c.gradient_descent(training_data, 40, 20, test_data,10)
'''
x = training_data[0][0].reshape(28,28)
plt.imshow(x, cmap = 'gray')
plt.show()
'''