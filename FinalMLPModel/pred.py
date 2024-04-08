import pandas as pd
import numpy as np
from data_parsing import retreive_file_data, retreive_data

LABELS = ["Dubai", "Rio de Janeiro", "New York City", "Paris"]

class MultiLayerPerceptron:
    def __init__(self, num_features=138, num_hidden=(300, 300, 300, 300), num_classes=4, activation="logistic"):
        """
        Initialize the weights and biases of this multi-layer perceptron.
        """
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.num_layers = len(num_hidden) + 1
        self.activation = activation

        # Initialize weights for all layers
        self.layer_matrices = self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weight matrices for all layers.
        """
        layer_matrices = []

        # Add weights for input layer
        input_layer_weights = np.zeros((self.num_hidden[0], self.num_features))
        layer_matrices.append(input_layer_weights)

        # Add weights for hidden layers
        for hidden_layer_index in range(len(self.num_hidden) - 1):
            hidden_layer_weights = np.zeros((self.num_hidden[hidden_layer_index + 1], self.num_hidden[hidden_layer_index]))
            layer_matrices.append(hidden_layer_weights)

        # Add weights for output layer
        output_layer_weights = np.zeros((self.num_classes, self.num_hidden[-1]))
        layer_matrices.append(output_layer_weights)

        # Read weights from files and set into matrices
        self._read_weights_from_files(layer_matrices)

        return layer_matrices
    
    def sigmoid_activation(self, z):
        """
        Compute sigmoid activation function for vector z or row-wise for a matrix z.
        """
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """
        Compute forward pass to produce predictions for inputs.
        """
        activation_fn = self.sigmoid_activation if self.activation == "logistic" else None

        # Forward pass through network layers
        value = X
        for weights in self.layer_weights:
            value = activation_fn(weights @ value)

        return value

def run_tests(model):
    # Get training data from data_parsing module
    x_train, y_train, _, _ = retreive_file_data()

    num_correct = 0
    total_samples = len(x_train)
    
    # Loop over training samples
    for i in range(total_samples):
        features = np.array(x_train[i], dtype=float).reshape(-1, 1)
        true_label = LABELS[np.argmax(y_train[i])]

        model_pred = model.forward(features)
        predicted_label = LABELS[np.argmax(model_pred)]

        if true_label == predicted_label:
            num_correct += 1
            print("CORRECT PREDICTION: ", end="")
        else:
            print("INCORRECT PREDICTION: ", end="")
        print(true_label, " - ", predicted_label)  
    print("Accuracy:", num_correct / total_samples)

def load_model():
    # Load hyperparameters from a file
    with open("./Weights/Hyperparameters.txt", "r") as f:
        hidden_units = eval(f.readline().strip().split(": ")[1])
        num_features = eval(f.readline().strip().split(": ")[1])
    
    # Initialize MLPModel instance with loaded hyperparameters
    model = MLPModel(num_features=num_features, hidden_units=hidden_units, num_classes=4)
    return model

def make_prediction(x):
    """
    Make a prediction for a given input x.
    """
    x = np.array(x, dtype=float).reshape(-1, 1)
    model = load_model()
    model_pred = model.forward(x)
    predicted_label = LABELS[np.argmax(model_pred)]

    return predicted_label

def predict_all(filename):
    """
    Make predictions for all data samples in specified file.
    """
    model = load_model()
    features = retreive_data(filename)

    model_preds = []
    for data_point in features:
        features = np.array(data_point, dtype=float).reshape(-1, 1)
        pred = model.forward(features)
        predicted_label = LABELS[np.argmax(pred)]
        model_preds.append(predicted_label)

    return model_preds
'''
# Example usage
print(predict_all("./clean_dataset.csv"))
x_train, y_train, x_test, y_test = retreive_file_data()
print(make_prediction(x_train[0]))

model = load_model()
run_tests(model)
'''