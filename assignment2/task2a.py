import numpy as np
import utils
import typing

np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784, \
        f"X.shape[1]: {X.shape[1]}, should be 784"

    # normalization trick
    X = (X - np.mean(X)) / np.std(X)

    # bias trick: add 1 at the end of each image
    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)

    return X


def sigmoid(z):
    """
    The sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Derivative of the sigmoid function
    """
    return sigmoid(z) * (1 - sigmoid(z))


def improved_sigmoid(z):
    """
    The improved sigmoid function, zero-centered function
    """
    return 1.7159 * np.tanh(2 * z / 3)


def improved_sigmoid_prime(z):
    """
    Derivative of the improved sigmoid
    """
    return 1.7159 * 2 / (3 * np.cosh(2 * z / 3) ** 2)


def softmax(z):
    """
    The softmax function
    """
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """

    assert targets.shape == outputs.shape, \
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    C = - (targets * np.log(outputs)).sum(axis=1)

    return C.mean()


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3b hyperparameter
                 use_improved_weight_init: bool  # Task 3a hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define hidden_layer_output
        self.hidden_layer_output = [np.zeros(n) for n in neurons_per_layer[:-1]]
        # Add hidden_layer before activation
        self.hidden_layer_activation = [np.zeros(n) for n in neurons_per_layer[:-1]]

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            if use_improved_weight_init:
                w = np.random.normal(0, 1 / np.sqrt(prev), w_shape)
            else:
                w = np.random.uniform(-1, 1, w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """

        def _sigmoid(z):
            """Internal function for switching between standard sigmoid and the erivative of the improved sigmoid"""
            if self.use_improved_sigmoid:
                return improved_sigmoid(z)
            return sigmoid(z)

        # Forward pass using improved_sigmoid
        # first hidden layer
        self.hidden_layer_activation[0] = X.dot(self.ws[0])
        self.hidden_layer_output[0] = _sigmoid(self.hidden_layer_activation[0])

        # Task 4c - variable number of hidden layer
        for layer_idx in range(1, len(self.hidden_layer_output)):
            self.hidden_layer_activation[layer_idx] = self.hidden_layer_output[layer_idx - 1].dot(self.ws[layer_idx])
            self.hidden_layer_output[layer_idx] = _sigmoid(self.hidden_layer_activation[layer_idx])

        return softmax(self.hidden_layer_output[-1].dot(self.ws[-1]))

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert targets.shape == outputs.shape, \
            f"Output shape: {outputs.shape}, targets: {targets.shape}"

        def _sigmoid_prime(z):
            """Internal function for switching between standard sigmoid and the erivative of the improved sigmoid"""
            if self.use_improved_sigmoid:
                return improved_sigmoid_prime(z)
            return sigmoid_prime(z)

        # compute error for the last layer
        error = outputs - targets

        # reverse iteration over layers
        for layer_idx in range(1, len(self.ws)):
            # update gradient with error
            self.grads[-layer_idx] = np.dot(self.hidden_layer_output[-layer_idx].T, error) / outputs.shape[0]

            # compute error for the previous layer
            error = np.dot(error, self.ws[-layer_idx].T) * _sigmoid_prime(self.hidden_layer_activation[-layer_idx])

        # update gradient of first layer
        self.grads[0] = np.dot(X.T, error) / outputs.shape[0]

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    return np.eye(num_classes)[Y[:, 0]]


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                             model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon ** 2, \
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785, \
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
