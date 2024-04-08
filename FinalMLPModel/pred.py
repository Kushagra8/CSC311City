import numpy as np
from data_parsing import retreive_file_data, retreive_data

LABELS = ["Dubai", "Rio de Janeiro", "New York City", "Paris"]

class MLPModel:
    def __init__(self, num_features=138, hidden_units=(300, 300, 300, 300), num_classes=4, activation="logistic"):
        """
        Initialize the weights and biases of this multi-layer perceptron (MLP) model.
        """
        # Model configuration
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.activation = activation

        # Initialize weight matrices for each layer
        self.layer_weights = []

        # Add weights for the input layer
        self.layer_weights.append(np.zeros((self.hidden_units[0], self.num_features)))

        # Add weights for hidden layers
        for i in range(len(self.hidden_units) - 1):
            self.layer_weights.append(np.zeros((self.hidden_units[i+1], self.hidden_units[i])))

        # Add weights for the output layer
        self.layer_weights.append(np.zeros((self.num_classes, self.hidden_units[-1])))

        # Read weights from files and set them into matrices
        # This part is omitted for clarity

    def sigmoid_activation(self, z):
        """
        Compute the sigmoid activation function for vector z or row-wise for a matrix z.
        """
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """
        Compute the forward pass to produce predictions for inputs.
        """
        activation_fn = self.sigmoid_activation if self.activation == "logistic" else None

        # Forward pass through the network layers
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

        prediction = model.forward(features)
        predicted_label = LABELS[np.argmax(prediction)]

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
    prediction = model.forward(x)
    predicted_label = LABELS[np.argmax(prediction)]

    return predicted_label

def predict_all_samples(filename):
    """
    Make predictions for all data samples in the specified file.
    """
    model = load_model()
    features = retreive_data(filename)

    predictions = []
    for data_point in features:
        features = np.array(data_point, dtype=float).reshape(-1, 1)
        pred = model.forward(features)
        predicted_label = LABELS[np.argmax(pred)]
        predictions.append(predicted_label)

    return predictions

# Example usage
print(predict_all_samples("./clean_dataset.csv"))
x_train, _, _, _ = retreive_file_data()
print(make_prediction(x_train[0]))

model = load_model()
run_tests(model)