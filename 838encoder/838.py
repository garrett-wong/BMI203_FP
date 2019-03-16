import sys
import numpy as np

def setup_nn(layerSizes):
	'''
	set up neural net. takes as input a list of nodes per layer.
	returns a dictionary of (W,b) tuples per layer.
	'''
	params = {} # a list of (w, b) tuples for each layer
	for i, l in enumerate(layerSizes):
		k = layerSizes[i-1]
		if i == 0:
			# There's no parameters to keep track of for the input
			# layer. We keep a tuple of empties to keep our indexing in
			# params sane (so params[i] are the parameters for layer i)
			W = np.empty(0)
			b = np.empty(0)
		else:
			# each node has k weights for each of the k activations;
			# there are l such nodes. These weights are arranged in l rows
			# of k weights in W so the matrix multiplication works out.
			# This is initialization for the example at https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/.
			#if i == 1:
			#	W = np.array([[0.15, 0.20], [0.25, 0.30]])
			#	b = np.array([0.35, 0.35])
			#elif i == 2:
			#	W = np.array([[0.40, 0.45], [0.50, 0.55]])
			#	b = np.array([0.60, 0.60])
			W = np.random.randn(l, k)*INITIAL_STDEV
			b = np.random.randn(l)*INITIAL_STDEV
		assert "W_" + str(i) not in params and "b_" + str(i) not in params
		params["W_" + str(i)] = W
		params["b_" + str(i)] = b
	return params

def sigmoid(Z):
	'''sigmoid activation function.
	'''
    return 1/(1+np.exp(-Z))

F = sigmoid
def feed_forward(x, params):
	'''
	runs forward propogation.

	input a input vector x and params;
	a dictionary of (W,b) tuples per layer. output a dictionary of As and Zs per layer.
	'''
	forward_params = {} # contains A and Zs per layer.
	for i, l in enumerate(layerSizes):
		if i == 0: 
			a = x # our inputs are the activations for the first layer.
			z = np.empty(0) # there are no corresponding weighted sums.
		else:
			W = params["W_" + str(i)]
			b = params["b_" + str(i)]
			a_last = forward_params["a_" + str(i-1)]
			z = np.dot(W, a_last) + b
			a = F(z)
		assert "z_" + str(i) not in params and "a_" + str(i) not in params
		forward_params["z_" + str(i)] = z
		forward_params["a_" + str(i)] = a
	return forward_params

def getError(target, output):
	'''ordinary error function. super differentiable.
	'''
	return np.sum(np.square(output-target)*1/2)

def sigmoidPrime(x):
	'''
	derivative of sigmoid function.
	'''
	return x * (1-x)

def errorPrime(target, output):
	'''
	derivative of error function.
	'''
	return(-(target - output))

fPrime = sigmoidPrime
def backprop(target, forward_params, params):
	''' runs backpropogation.

	input the dictionaries of parameters from forward propogation and weights
	output the dictionary of deltas and grads from backprop.
	'''
	rev_params = {} # deltas and grads.
	# first, deltas, iterating backward through the layers:
	for i, l in reversed(list(enumerate(layerSizes))):
		a = forward_params["a_" + str(i)]
		W = params["W_" + str(i)]
		if i == 0: continue # done!
		if i == len(layerSizes) -1:
			delta = errorPrime(target, a) * fPrime(a)
		else:
			delta_ahead = rev_params["delta_" + str(i+1)]
			delta = W.T.dot(delta_ahead) * fPrime(a)
		rev_params["delta_" + str(i)] = delta
	# then grads iterating forward (it doesn't really matter):
	for i, l in enumerate(layerSizes):
		if i == 0:
			continue
		# We have to make sure our dimensions are right here....
		a_last = forward_params["a_" + str(i-1)]
		a_last.shape = (l, 1)
		delta = rev_params["delta_" + str(i)]
		delta.shape = (l, 1)
		grad_W = np.dot(delta, a_last.T)
		grad_b = delta # This is wrong...
		rev_params["grad_W_" + str(i)] = grad_W
		rev_params["grad_b_" + str(i)] = grad_b
	return rev_params

learnRate = 0.5
def updateWeights(params, rev_params):
	'''
	just goes through and updates the weights given grads and deltas.

	input the weights and dictionary of grads and deltas, output new weights.
	'''
	newParams = {}
	for i, l in enumerate(layerSizes):
		if i == 0: continue
		W = params["W_" + str(i)]
		b = params["b_" + str(i)]
		grad_W = rev_params["grad_W_" + str(i)]
		grad_b = rev_params["grad_b_" + str(i)]
		new_W = W - learnRate*grad_W
		new_b = b - grad_b*learnRate
		newParams["W_" + str(i)] = new_W
		newParams["b_" + str(i)] = new_b
	newParams["W_0"] = np.empty(0)
	newParams["b_0"] = np.empty(0)
	return newParams

# initialization for the example at https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/.
# it seems to work for one iteration and gives the same result as the example, but then there's some issue with
# my variables. I think I'm failing to do all my housekeeping on variable names.

#layerSizes = [2, 2, 2]
#x = np.array([0.05, 0.10])
#target = np.array([0.01, 0.99])

def train_nn(x, target, layerSizes):
	errors = []
	params = setup_nn(layerSizes)
	for i in range(100):
		print(i)
		forward_params = feed_forward(x, params)
		error = getError(target, forward_params["a_" + str(len(layerSizes)-1)])
		errors.append(error)
		rev_params = backprop(target, forward_params, params)
		newParams = updateWeights(params, rev_params)
		params = newParams


