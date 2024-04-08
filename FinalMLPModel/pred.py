import pandas as pd
import numpy as np
from data_parsing import retreive_file_data, retreive_data

# List of city labels
LABELS = ["Dubai", "Rio de Janeiro", "New York City", "Paris"]

class MLPModel:
    def __init__(self, num_features=138, hidden_units=(300, 300, 300, 300), num_classes=4, activation="logistic"):
        """
        Initialize weights and biases of multi-layer perceptron.
        """
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.num_layers = len(hidden_units) + 1
        self.activation = activation

        # Initialize weights for all layers
        self.layer_weights = self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weight matrices for all layers.
        """
        layer_weights = []

        # Add weights for input layer
        input_weights = np.zeros((self.hidden_units[0], self.num_features))
        layer_weights.append(input_weights)

        # Add weights for hidden layers
        for i in range(len(self.hidden_units) - 1):
            hidden_weights = np.zeros((self.hidden_units[i + 1], self.hidden_units[i]))
            layer_weights.append(hidden_weights)

        # Add weights for output layer
        output_weights = np.zeros((self.num_classes, self.hidden_units[-1]))
        layer_weights.append(output_weights)

        # Read weights from files and set into matrices
        self._read_weights_from_files(layer_weights)

        return layer_weights
    
    def sigmoid_activation(self, z):
        """
        Compute sigmoid activation function for vector z / row-wise for matrix z.
        """
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """
        Compute the forward pass to produce prediction for inputs.

        Parameters:
            `X` - A numpy array of shape (N, self.num_features)

        Returns: A numpy array of predictions of shape (N, self.num_classes)
        """

        act = None
        if self.activation == "logistic":
            act = self.sigmoid_activation

        assert len(self.layer_weights) >= 2

        # Check if the number of features in input X matches the expected number of features
        assert X.shape[1] == self.num_features, "Number of features in input does not match model's num_features."

        # First Layer
        value = act(self.layer_weights[0] @ X.T)

        # Deep Layers
        for i in range(self.num_layers - 2):
            value = act(self.layer_weights[i+1] @ value)

        # Last Layer
        value = act(self.layer_weights[-1] @ value)

        return value.T

    def _read_weights_from_files(self, layer_weights):
        '''
        Read weights from files and set into weight matrices.
        '''
        # Loop through each layer's weight matrix
        for i, weights in enumerate(layer_weights):
            # Open file containing weights for current layer
            with open(f"FinalMLPModel/Weights/Layer{i}weights.txt", "r") as f:
                # Initialize variables
                col_index = 0  # Represents current column index
                col_values = []  # Holds weights for current column
                
                # Loop through each line in file
                for line in f.readlines():
                    # Check if line represents node
                    if "Node " in line:
                        # If weights already in there, assign to weight matrix
                        if col_values:
                            # Ensure num of weights matches number of rows in matrix
                            assert len(col_values) == weights.shape[0]

                            # Assign weights to current column in matrix
                            weights[:, col_index] = col_values
                            col_values = []  # Reset column values for next column
                        
                        # Extract node num from line and convert it to an int
                        node_num = int(line[len("Node "):-3])
                        col_index = node_num  # Set column index for current node
                    else:
                        # Strip whitespace and remove leading/trailing '[' and ']' characters
                        line = line.strip().lstrip("[").rstrip("]")

                        # Split line into individual nums and convert to floats
                        nums = [float(n) for n in line.split() if n]
                        col_values.extend(nums)  # Add weights to column values list

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
    # Load hyperparameters from file
    with open("FinalMLPModel/Weights/Hyperparameters.txt", "r") as f:
        hidden_units = eval(f.readline().strip().split(": ")[1])
        num_features = eval(f.readline().strip().split(": ")[1])
    
    # Initialize MLPModel instance with loaded hyperparameters
    model = MLPModel(num_features=num_features, hidden_units=hidden_units, num_classes=4)
    return model

def make_prediction(x):
    """
    Make prediction for given input x.
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

# Example usage
print(predict_all("./clean_dataset.csv"))
x_train, y_train, x_test, y_test = retreive_file_data()
print(make_prediction(x_train[0]))

model = load_model()
run_tests(model)
