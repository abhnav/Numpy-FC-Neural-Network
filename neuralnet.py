################################################################################
# CSE 251B: Programming Assignment 2
# Winter 2022
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import time
import yaml
import numpy as np
import pickle
import pdb
import matplotlib.pyplot as plt


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(inp):
    """
    TODO: Normalize your inputs here to have 0 mean and unit variance.
    """
    # need this contorted way because of difference in test, train data format
    n = len(inp)
    k = inp[0].shape[0]
    out = np.zeros((n,k))

    for i in np.arange(len(inp)):
      tp = inp[i].astype(float)
      for j in np.arange(3): #normalize the channels separately
        b = 1024*j
        e = 1024*(j+1)
        mn = tp[b:e].mean()
        std = tp[b:e].std()
        tp[b:e] = tp[b:e] - mn
        tp[b:e] = tp[b:e] / std
      out[i] = tp

    return out


def one_hot_encoding(labels, num_classes=10):
    """
    TODO: Encode labels using one hot encoding and return them.
    """
    hl = np.zeros((len(labels), num_classes))
    hl[np.arange(len(labels)), np.array(labels)] = 1
    return hl


def load_data(path, mode='train'):
    """
    Load CIFAR-10 data.
    """
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    cifar_path = os.path.join(path, "cifar-10-batches-py")

    if mode == "train":
        images = []
        labels = []
        for i in range(1,6):
            images_dict = unpickle(os.path.join(cifar_path, f"data_batch_{i}"))
            data = images_dict[b'data']
            label = images_dict[b'labels']
            labels.extend(label)
            images.extend(data)
        normalized_images = normalize_data(images)
        one_hot_labels    = one_hot_encoding(labels, num_classes=10) #(n,10)
        return np.array(normalized_images), np.array(one_hot_labels)
    elif mode == "test":
        test_images_dict = unpickle(os.path.join(cifar_path, f"test_batch"))
        test_data = test_images_dict[b'data']
        test_labels = test_images_dict[b'labels']
        normalized_images = normalize_data(test_data)
        one_hot_labels    = one_hot_encoding(test_labels, num_classes=10) #(n,10)
        return np.array(normalized_images), np.array(one_hot_labels)
    else:
        raise NotImplementedError(f"Provide a valid mode for load data (train/test)")


def softmax(x):
    """
    TODO: Implement the softmax function here.
    Remember to take care of the overflow condition.
    x is of form N x K where N is the number of samples, K is number of
    classes in one-hot encoding
    """
    mx = x.max(axis=1)[:,np.newaxis]
    y = np.exp(x-mx)
    sm = y.sum(axis=1)[:,np.newaxis]
    return y / sm


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        TODO: Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "leakyReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

        elif self.activation_type == "leakyReLU":
            return self.leakyReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()
        else:
            grad = self.grad_leakyReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        TODO: Implement the sigmoid activation here.
        """
        self.x = 1 / (1 + np.exp(-x)) # TODO: overflow?
        return self.x

    def tanh(self, x):
        """
        TODO: Implement tanh here.
        """
        self.x = np.tanh(x)
        return self.x

    def ReLU(self, x):
        """
        TODO: Implement ReLU here.
        """
        y = np.copy(x)
        y[x<0] = 0
        self.x = y
        return self.x

    def leakyReLU(self, x):
        """
        TODO: Implement leaky ReLU here.
        """
        y = np.copy(x)
        y[x<0] = y[x<0] * 0.1 # scale down the negative values
        self.x = y
        return self.x

    def grad_sigmoid(self):
        """
        TODO: Compute the gradient for sigmoid here.
        """
        return self.x * (1-self.x)

    def grad_tanh(self):
        """
        TODO: Compute the gradient for tanh here.
        """
        return 1 - self.x * self.x

    def grad_ReLU(self):
        """
        TODO: Compute the gradient for ReLU here.
        """
        grad = np.copy(self.x)
        grad[self.x > 0] = 1
        return grad

    def grad_leakyReLU(self):
        """
        TODO: Compute the gradient for leaky ReLU here.
        """
        grad = np.copy(self.x)
        grad[self.x > 0] = 1
        grad[self.x < 0] = 0.1
        return grad

    def update_weights(self, lr):
      pass

class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        # using rand seems to give better results
        self.w = np.random.randn(in_units, out_units)   # Declare the Weight matrix
        self.b = np.random.randn(1,out_units)    # Create a placeholder for Bias, row vector
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        TODO: Compute the forward pass through the layer here.
        DO NOT apply activation here.
        Return self.a
        Assumes input of format N x k where k is the input dimension
        """
        self.x = x
        self.a = x.dot(self.w) + self.b # broadcast bias for all input samples
        return self.a

    def backward(self, delta):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        self.d_x = delta.dot(self.w.T)
        self.d_b = delta.sum(axis=0)[np.newaxis,:] # bias coefficient is always 1
        self.d_w = np.einsum("ni,nj->ij", self.x, delta) # in_dim x out_dim matrix
        return self.d_x

    def update_weights(self, lr):
      self.b = self.b - lr*self.d_b
      self.w = self.w - lr*self.d_w

class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        TODO: Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        Targets must be one-hot as well
        """
        self.x = x
        self.targets = targets
        ls = None

        y = np.copy(x)
        for la in self.layers:
          y = la(y)
        y = softmax(y)
        if targets is not None:
          ls = self.loss(y, targets)
        self.y = y
        return self.y, ls

    def loss(self, logits, targets):
        '''
        TODO: compute the categorical cross-entropy loss and return it.
        logits is the softmax predictions
        '''
        eps = 1e-8
        ls = -targets*np.log(logits + eps)
        ls = ls.sum(axis=1)
        return ls

    def backward(self):
        '''
        TODO: Implement backpropagation here.
        Call backward methods of individual layers.
        '''
        # Dividing by number of targets reduces the gradient magnitude,
        # making it harder to learn.
        # Momentum is required to get better results. Until then don't divide
        delta = (self.y - self.targets) #/ self.targets.shape[0]
        for la in reversed(self.layers):
          delta = la.backward(delta)

    # TODO: include momentum gamma
    def update_weights(self,lr):
      for la in self.layers:
        la.update_weights(lr)

def dataloader(x_train, y_train, bs):
  n = x_train.shape[0]
  order = np.random.permutation(n)
  tt = 0
  while(tt<n):
    yield x_train[order[tt:tt+bs]], y_train[order[tt:tt+bs]]
    tt = tt + bs

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    epochs = config['epochs']
    batchsize = config["batch_size"]
    lr = config["learning_rate"]
    netloss_t = []
    netloss_v = []
    for ep in np.arange(epochs):
      bb = time.time()
      loss_t = 0
      print(f"starting run {ep+1}")
      for x_t,y_t in dataloader(x_train, y_train, batchsize):
        p_t,loss_batch = model(x_t,y_t)
        loss_t = loss_t + loss_batch.sum()
        model.backward()
        model.update_weights(lr)
      p_v, loss_v = model(x_valid, y_valid)
      loss_v = loss_v.sum()
      # Average the losses over samples and classes
      loss_v = loss_v / (x_valid.shape[0] * y_valid.shape[1])
      loss_t = loss_t / (x_train.shape[0] * y_train.shape[1])
      netloss_t.append(loss_t)
      netloss_v.append(loss_v)
      print(f"train loss:{loss_t}, validation loss:{loss_v}")
      acc = test(model, x_valid, y_valid)
      print(f"val accuracy:{acc}")
      print(f"took time {time.time() - bb}")
    return netloss_t, netloss_v

def test(model, X_test, y_test):
    """
    TODO: Calculate and return the accuracy on the test set.
    """
    p_test,ls = model(X_test)
    p_c = p_test.argmax(axis=1)
    t_c = y_test.argmax(axis=1)
    cr = np.sum(p_c == t_c)
    return (100*cr)/p_c.shape[0]

if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./data/")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./data", mode="train")
    x_test,  y_test  = load_data(path="./data", mode="test")
    
    # TODO: Create splits for validation data here.
    N = x_train.shape[0]
    val_ratio = 0.1
    N_val = int(N*val_ratio)
    order = np.random.permutation(N)

    x_val, y_val = x_train[order[:N_val]], y_train[order[:N_val]]
    x_t, y_t = x_train[order[N_val:]], y_train[order[N_val:]]

    l_t,l_v = train(model, x_t, y_t, x_val, y_val, config)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(l_t)), np.array(l_t), "b")
    ax.plot(np.arange(len(l_v)), np.array(l_v), "r")
    plt.show()

    test_acc = test(model, x_test, y_test)
    print(f"Test accuracy is {test_acc} percent")

    # # TODO: Plots
    # # plt.plot(...)
