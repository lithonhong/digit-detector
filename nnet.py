import numpy as np


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(i, 1) for i in sizes[1:]]
        self.weights = [np.random.randn(j, i) for i, j in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, input_vct):
        """
        Return the output of the neural network with `input_vct` as input.
        """
        for b, w in zip(self.biases, self.weights):
            input_vct = sigmoid(np.dot(w, input_vct) + b)

        return input_vct

    def sgd(self, train_data, epoch, mini_batch_size, eta, test_data=None):
        """
        Implements mini-batch stochastic gradient descent.
        `train_data`: A list of tuples representing the train data and its desired output.
        `epoch`: The number of epoches to train the neural network.
        `mini_batch_size`: The size of mini-batches to estimate $\nabla C$, the gradient vector.
        `eta`: The learning rate of the SGD.
        `test_data`: Optional. If provided, the neural network is compared with the test data after every 10% of epoch.
        """

        pcts = [int(epoch / 10 * i) for i in range(1, 11)]

        for e in range(1, epoch+1):
            np.random.shuffle(train_data)
            mini_batches = [train_data[k:k+mini_batch_size] for k in range(0, len(train_data), mini_batch_size)]

            for m in mini_batches:
                self.update_mini_batch(m, eta)

            if test_data:
                if e in pcts:
                    print(f"{(pcts.index(e)+1)*10}% | Epoch {e}/{epoch} | Accuracy {self.evaluate(test_data)}/{len(test_data)}")

            #print(f"Epoch {e} completed.")

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        `mini_batch`: A list of tuples `(x, y)`
        `eta`: The learning rate.

        To review this code after Chapter 2.
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [
            w - (eta/len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta/len(mini_batch)) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """
        Return a tuple `(nabla_b, nabla_w)` representing the
        gradient for the cost function C_x.
        `nabla_b`, `nabla_w`: Layer-by-layer lists of numpy arrays.

        To review this code after Chapter 2.
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.

        To review this code after Chapter 2.
        """

        test_results = [
            (np.argmax(self.feedforward(x)), y)
            for (x, y) in test_data
        ]
        
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives $\partial C_x /
        \partial a$ for the output activations.

        To review this code after Chapter 2.
        """

        return (output_activations-y)