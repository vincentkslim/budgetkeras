import numpy as np


### LOSS FUNCTIONS

def binary_crossentropy():
    """Binary cross entropy (logistic) loss function"""

    def binary_crossentropy_forward(A_L, Y):
        """Computes the cost for the final layer"""

        # m = num training examples
        m = Y.shape[1]

        cost = -1/m * np.sum(Y*np.log(A_L) + (1-Y)*np.log(1-A_L))

        return np.squeeze(cost)

    def binary_crossentropy_backward(A_L, Y):
        """Computes the derivative of the logistic loss function"""

        return -(np.divide(Y, A_L) - np.divide(1 - Y, 1 - A_L))

    return binary_crossentropy_forward, binary_crossentropy_backward


### WEIGHT INITIALIZATION

def initialize_weights(layers_dims, initializers):
    """
    Initializes the weights of each layer according to the initializer for each
    layer.

    Arguments:
    layers_dims         - array containing the dimensions of each layer
    initializers        - array containing the initialization functions of each layer

    Note that the baises are initialized to zero.
    """

    parameters_W, parameters_b = [], []

    for l in range(1, len(layers_dims)):
        initializer = initializers[l]

        layer = layers_dims[l]
        layer_prev = layers_dims[l-1]

        parameters_W.append(initializer(layer, layer_prev))
        parameters_b.append(np.zeros((layer, 1)))

    return parameters_W, parameters_b

def random_normal(layer, layer_prev, mean=0, std=1):
    """
    Returns a random normal weight initialization for a layer

    Arguments:
    layer       - number of units in the current layer (output units)
    layer_prev  - number of units in the previous layer (input units)
    mean        - mean of the normal distribution to sample from
    std         - standard deviation of the normal distribution to sample from
    """
    rng = np.random.default_rng(0)
    return rng.normal(mean, std, (layer, layer_prev))

def random_uniform(layer, layer_prev, low=-1, high=1):
    """
    Returns a random uniform distribution weight initialization for a layer

    Arguments:
    layer       - number of units in the current layer (output units)
    layer_prev  - number of units in the previous layer (input units)
    low         - lower bound of the interval to sample from
    high        - upper bound of the interval to sample from
    """
    rng = np.random.default_rng(0)
    return rng.uniform(low, high, (layer, layer_prev))

def kaiming(layer, layer_prev):
    """
    Returns the Kaiming (He) initialization for a layer

    This initialization is equivalent to the standard normal distribution scaled by
    a factor of sqrt(2/layer_prev)
        
    Arguments:
    layer       - number of units in the current layer (output units)
    layer_prev  - number of units in the previous layer (input units)


    This initialization is good for ReLU activations
    """
    return random_normal(layer, layer_prev, mean=0, std=np.sqrt(2/layer_prev))

def xavier(layer, layer_prev):
    """
    Returns the xavier initialization for a layer

    This initialization is equivalent to the standard normal distribution scaled by
    a factor of sqrt(6/(layer + layer_prev))
        
    Arguments:
    layer       - number of units in the current layer (output units)
    layer_prev  - number of units in the previous layer (input units)


    This initialization is good for tanh and sigmoid activations
    """
    bound = np.sqrt(6/(layer + layer_prev))
    return random_uniform(layer, layer_prev, low=-bound, high=bound)


### FORWARD AND BACKWARD PROPAGATION

def forward_prop(A_prev, W, b, activation):
    """
    Computes the activations for a layer
    
    A_prev      - array of activations from the previous layer
    W           - array of weights for a layer
    b           - array of biases for a layer
    activation  - function that takes Z and returns the computed activations
    """

    Z = np.dot(W, A_prev) + b
    A = activation(Z)

    cache = (A_prev, W, b, Z)

    return A, cache

def backward_prop(dA, cache, activation_backward):
    """
    Backpropagation step for a hidden layer
    
    Arguments:
    dA          - dL/dA for the current layer
    cache       - cache containing (A_prev, W, b, Z)
                - A_prev is activations from previous layer
                - W, b, and Z are all for the current layer

    Returns:
    dA_prev     - dL/dA for the previous layer
    dW          - gradients for the weights for the current layer
    db          - gradients for the baises for the current layer
    """
    
    A_prev, W, b, Z = cache
    m = A_prev.shape[1]

    dZ = dA * activation_backward(Z)
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)

    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


### ACTIVATION FUNCTIONS

def sigmoid():
    """
    Returns sigmoid_forward and sigmoid_backward

    sigmoid_forward is used in the forward propagation step and computes the sigmoid(z)
    sigmoid_backward is the derivative of sigmoid
    """
    def sigmoid_forward(z):
        """Returns the element-wise sigmoid of z"""
        return 1/(1+np.exp(-z))

    def sigmoid_backward(z):
        """Returns the derivative of sigmoid function of every element within numpy array z"""
        return sigmoid_forward(z) * (1-sigmoid_forward(z))

    return sigmoid_forward, sigmoid_backward

def relu():
    """
    Returns relu_forward and relu_backward

    relu_forward is used in the forward propagation step
    relu_backward is the derivative of sigmoid
    """
    def relu_forward(z):
        """Returns the element-wise relu of z"""
        return np.maximum(z, 0)

    def relu_backward(z):
        """Applies the element-wise derivative of relu on z"""

        return (z >= 0).astype(int)

    return relu_forward, relu_backward

def tanh():
    """
    Returns tanh_forward and tanh_backward

    tanh_forward is used in the forward propagation step
    tanh_backward is the derivative of sigmoid
    """
    def tanh_forward(z):
        """Returns the element-wise tanh of z"""
        return np.tanh(z)

    def tanh_backward(z):
        """Applies the element-wise derivative of tanh on z"""
        return 1-np.power(tanh_forward(z), 2)

    return tanh_forward, tanh_backward

def forward(activation):
    return activation[0]

def backward(activation):
    return activation[1]

### LAYERS

def dense(units, input_shape=None, activation=relu, initializer=kaiming):
    return (units, input_shape, activation, initializer)


### MODEL

def sequential():
    layers = []                 # dimensions of each layer, layers[0] is the input layer
    activations = [()]          # activation functions of each layer, 1 indexed
    initializers = [()]         # weight initializers of each layer, 1 indexed

    parameters_W = None         # weights of each layer, 1 indexed
    parameters_b = None         # biases of each layer, 1 indexed

    caches = None               # caches for each layer, 1 indexed
    history = []

    optimizer = None
    loss = None

    def add(layer):
        """
        Adds a layer to the model
        """
        nonlocal layers, activations, parameters_W, parameters_b, caches, optimizer, loss, initializers

        units, input_shape, activation, initializer = layer

        if len(layers) == 0:
            assert input_shape is not None, "First layer must have input shape"
            assert len(input_shape) == 1, "First layer must have shape (n_x, )"

            layers.append(input_shape[0])
        
        layers.append(units)

        activations.append(activation())

        initializers.append(initializer)

    def compile_model(optimizer_func, loss_func, metrics=[]):
        nonlocal layers, activations, parameters_W, parameters_b, caches, optimizer, loss

        parameters_W, parameters_b = initialize_weights(layers, initializers)

        parameters_W.insert(0, [])     # dummy weights to make it 1 indexed
        parameters_b.insert(0, [])

        optimizer = optimizer_func
        loss = loss_func()

        return fit


    def fit(x_train, y_train, batch_size=None, epochs=10, val_data=None):
        nonlocal layers, activations, parameters_W, parameters_b, caches, optimizer, loss

        assert optimizer is not None, "Must call compile first"
        assert loss is not None

        for epoch in range(epochs):
            # print(parameters_W)
            # print(parameters_b)
            A = x_train.T       # This assumes that the train data is set that each row is a training example, so
                                # need to transpose so that each column is a training example.

            caches = [[]]
            grads_W = {}
            grads_b = {}

            # forward prop
            for l in range(1, len(layers)):
                A_prev = A
                # layers[0] = input layer
                A, cache = forward_prop(A_prev, parameters_W[l], parameters_b[l], activations[l][0])
                caches.append(cache)

            # compute cost
            cost = loss[0](A, y_train)
            history.append(cost)

            # initialize backprop
            dA_L = loss[1](A, y_train)

            # backprop
            dA = dA_L
            for l in reversed(range(1, len(layers))):
                dA_prev, dW, db = backward_prop(dA, caches[l], activations[l][1])
                grads_W["dW" + str(l)] = dW
                grads_b["db" + str(l)] = db

                dA = dA_prev

            # update weights
            parameters_W, parameters_b = optimizer(parameters_W, parameters_b, grads_W, grads_b)

        return history, predict

    def predict(X):
        nonlocal layers, activations, parameters_W, parameters_b, caches, optimizer, loss

        # forward prop
        A = X.T
        for l in range(1, len(layers)):
            A_prev = A
            # layers[0] = input layer
            A, cache = forward_prop(A_prev, parameters_W[l], parameters_b[l], activations[l][0])
        return A

    def summary():
        nonlocal layers, activations, parameters_W, parameters_b, caches, optimizer, loss

        print(f'+{"":-^20}+{"":-^30}+')
        print(f'|{"Hidden Units":^20}|{"Activation Function":^30}|')
        print(f'+{"":-^20}+{"":-^30}+')

        for i in range(1, len(layers)):
            print(f'|{layers[i]:^20}|{activations[i][0].__name__:^30}|')

        print(f'+{"":-^20}+{"":-^30}+')

    return add, compile_model, summary


### OPTIMIZERS

def gradient_descent(learning_rate=0.01):
    def update_weights(parameters_W, parameters_b, grads_W, grads_b):
        for p in range(1, len(parameters_W)):
            parameters_W[p] = parameters_W[p] - learning_rate * grads_W["dW" + str(p)]
            parameters_b[p] = parameters_b[p] - learning_rate * grads_b["db" + str(p)]

        return parameters_W, parameters_b
    return update_weights

def gradient_descent_with_momentum(learning_rate=0.01, momentum=0.9):
    vel_dW, vel_db = {}, {}

    def update_weights(parameters_W, parameters_b, grads_W, grads_b):
        for l in range(1, len(parameters_W)):
            i_W = "dW" + str(l)
            i_b = "db" + str(l)
            vel_dW[i_W] = momentum * vel_dW.get(i_W, 0) + (1-momentum) * grads_W[i_W]
            vel_db[i_b] = momentum * vel_db.get(i_b, 0) + (1-momentum) * grads_b[i_b]

            parameters_W[l] = parameters_W[l] - learning_rate * vel_dW[i_W]
            parameters_b[l] = parameters_b[l] - learning_rate * vel_db[i_b]

        return parameters_W, parameters_b
    return update_weights

def RMSprop(learning_rate = 0.001, rho=0.9, epsilon=1e-8):
    S_dW, S_db = {}, {}

    def update_weights(parameters_W, parameters_b, grads_W, grads_b):
        for l in range(1, len(parameters_W)):
            i_W = "dW" + str(l)
            i_b = "db" + str(l)
            S_dW[i_W] = rho * S_dW.get(i_W, 0) + (1-rho) * np.power(grads_W[i_W], 2)
            S_db[i_b] = rho * S_db.get(i_b, 0) + (1-rho) * np.power(grads_b[i_b], 2)

            parameters_W[l] = parameters_W[l] - learning_rate * grads_W[i_W] / np.sqrt(S_dW[i_W] + epsilon)
            parameters_b[l] = parameters_b[l] - learning_rate * grads_b[i_b] / np.sqrt(S_db[i_b] + epsilon)

        return parameters_W, parameters_b
    return update_weights


