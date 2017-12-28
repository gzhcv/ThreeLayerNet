import numpy as np

input_size = 4
hidden_size = 20
num_classes = 3
num_inputs = 30

def init_toy_model():
    np.random.seed(0)
    return ThreeLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = (3*np.random.rand(30)).astype(int)
    return X, y

class ThreeLayerNet(object):
	"""
	the structure of the network is:
		input -> fully connected layer -> sigmoid -> fully connected layer -> sigmoid.
	the output of the second fully-connected layer are the scores for each class.
	"""
	
	def __init__(self, input_size, hidden_size, output_size, std=1e-4):
		"""
		Initialize the model. Weight are initialized to small random values and
		bias are initialized to zero. Weights and biases are stored in the 
		variabel self.params, which is a dictionary with the following keys:
		
		W1: weights of connected first and second layer; has shape (D, H) 
		b1: bias of connected first and second layer;    has shape (H,  )
		W2: weights of connected second and third layer; has shape (H, C)
		b2: bias of connected second and third layer;    has shape (C,  )
		
		Inputs:
		- input_size:   The dimension D of the input data.
		- hidden_size:  The dimension H in the hidden layer.
		- output_size:  The numer of classes C.
		"""
		self.params = {}
		self.params['W1'] = std * np.random.randn(input_size, hidden_size)  
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = std * np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)
	
	def loss(self, X, y):
		"""
		compute the loss and gradients of the model with data (X, y).
		
		Inputs:
		- X: Inputs dat of shape (N,D). Each X[i] is a training sample.
		- y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
			 an integer in the range 0 <= y[i] < C.
		
		Returns:
		- loss:  Loss for this batch of training
				 samples.
		- grads: Dictionary mapping parameter names to gradients of those parameters
				 with respect to the loss function; has the same keys as self.params.
		"""
		# Unpack variables from the params dictionary
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		N, D = X.shape
		
		net_j = X.dot(W1) + b1
		y_j = 1 / (1+np.exp(-net_j))
		net_k = np.dot(y_j, W2) + b2
		z_k = 1 / (1+np.exp(-net_k))
		
		# Compute the loss
		y_trueClass = np.zeros_like(z_k)
		y_trueClass[np.arange(N), y] = 1.0
		loss = np.sum( np.power(y_trueClass - z_k,2) ) / N   
		
		# Backward pass: compute gradients
		grads = {}

		dz_dnetk = z_k*(1-z_k)
		dL_dnetk = (z_k-y_trueClass) * dz_dnetk
		grads['W2'] = np.dot(y_j.T, dL_dnetk ) / N 
		grads['b2'] = np.sum(dL_dnetk, axis=0) / N
		dy_dnetj = y_j*(1-y_j)
		dL_dnetj = np.dot(dL_dnetk, W2.T) * dy_dnetj
		grads['W1'] = np.dot(X.T, dL_dnetj) / N 
		grads['b1'] = np.sum(dL_dnetj, axis=0) / N
		
		return loss, grads
	
	def train(self, X, y, X_val, y_val,
			  lr=1e-3, num_iters=100, batch_size=15, verbose=False):
		"""
		Train this network using stochastic gradient descent.
		
		Inputs:
		- X: A numpy array of shape (N,D) giving training data.
		- y: A numpy array of shape (N, ) giving training labels; y[i] = c means that
		     X[i] has label c, where 0<= c < C.
		- X_val: A numpy array of shape (N_val, D) giving validation data.
		- y_val: A numpy array of shape (N_val,) giving validation labels.	
		- lr:    Scalar giving learning rate for optimization.
		- num_iters: Number of steps to take when optimizing.
		- batch_size: Number of training examples to use per step.
		- verbose: boolean; if true print progress during optimization.
		"""
		num_train = X.shape[0]
		iterations_per_epoch = max(num_train / batch_size, 1)
		
		# Use SGD to optimize the parameters in self.model
		loss_history = []
		train_acc_history = []
		val_acc_history = []
		
		for it in range(num_iters):
			indces = np.random.choice(num_train, batch_size, replace=False)
			X_batch = X[indces,:]
			y_batch = y[indces]
			# Compute loss and gradients using the current minibatch
			loss, grads = self.loss(X_batch, y=y_batch)
			loss_history.append(loss)
			
			self.params['W1'] = self.params['W1'] - lr*grads['W1']
			self.params['W2'] = self.params['W2'] - lr*grads['W2']
			self.params['b1'] = self.params['b1'] - lr*grads['b1']
			self.params['b2'] = self.params['b2'] - lr*grads['b2']
			
			if verbose and it % 10 == 0:
				print('iteration %d / %d: loss %f' % (it, num_iters, loss))
			
			if it % iterations_per_epoch == 0:
			# Check accuracy
				train_acc = (self.predict(X_batch) == y_batch).mean()
				train_acc_history.append(train_acc)
		return {
				'loss_history': loss_history,
				'train_acc_history': train_acc_history,
			   }
			   
	def predict(self, X):
		"""
		Use the trained weights of this Three-layer network to predict labels for
		data points. For each data point we predict scores for each of the C
		classes, and assign each data point to the class with the highest score.

		Inputs:
		- X: A numpy array of shape (N, D) giving N D-dimensional data points to
		  classify.

		Returns:
		- y_pred: A numpy array of shape (N,) giving predicted labels for each of
		  the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
		  to have class c, where 0 <= c < C.
		"""
		y_pred = None
		net_j = X.dot(self.params['W1']) + self.params['b1']
		fl = y_j = 1 / (1+np.exp(-net_j))
		net_k = np.dot(fl, self.params['W2']) + self.params['b2']
		scores = 1 / (1+np.exp(-net_k))
		y_pred = np.argmax(scores, axis=1)
	 
		return y_pred

def main():
	net = init_toy_model()
	X, y = init_toy_data()
	stats = net.train(X, y, X, y, lr=1, num_iters=1000, verbose=True)
	print('Final training loss: ', stats['loss_history'][-1])
	# Predict on the training set
	val_acc = (net.predict(X) == y).mean()
	print(net.predict(X))
	print('training accuracy: ', val_acc)

main()
