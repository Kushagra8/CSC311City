import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('clean_dataset.csv')

# Preprocess the data
data.drop('id', axis=1, inplace=True)

# Define a function to encode categorical variables
def encode_categorical(df):
    encoded_df = df.copy()
    for col in encoded_df.columns:
        if encoded_df[col].dtype == 'object':
            encoded_df[col] = pd.factorize(encoded_df[col])[0]
    return encoded_df

# Encode categorical variables
encoded_data = encode_categorical(data)

# Separate features (X) and target variable (y)
X = encoded_data.drop('Label', axis=1).values
y = encoded_data['Label'].values

# Define the multi-layer perceptron (MLP) class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weights_input_hidden = np.random.randn(input_dim, hidden_dim)
        self.bias_input_hidden = np.zeros(hidden_dim)
        self.weights_hidden_output = np.random.randn(hidden_dim, output_dim)
        self.bias_hidden_output = np.zeros(output_dim)
    
    '''
        def relu(self, x):
        return np.maximum(0, x)
    '''
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_hidden_output
        self.output_probs = self.softmax(self.output_layer_input)
        return self.output_probs
    
    def backward(self, X, y, learning_rate, clip_value=1.0):
        num_examples = len(X)
        doutput = self.output_probs
        doutput[range(num_examples), y] -= 1
        doutput /= num_examples
        dweights_hidden_output = np.dot(self.hidden_layer_output.T, doutput)
        dbias_hidden_output = np.sum(doutput, axis=0, keepdims=True)
        dhidden = np.dot(doutput, self.weights_hidden_output.T) * (self.hidden_layer_output * (1 - self.hidden_layer_output))
        dweights_input_hidden = np.dot(X.T, dhidden)
        dbias_input_hidden = np.sum(dhidden, axis=0, keepdims=True)

        # Update parameters
        self.weights_hidden_output -= learning_rate * dweights_hidden_output
        self.bias_hidden_output -= learning_rate * dbias_hidden_output.squeeze()
        self.weights_input_hidden -= learning_rate * dweights_input_hidden
        self.bias_input_hidden -= learning_rate * dbias_input_hidden.squeeze()
    
    def train(self, X, y, learning_rate, epochs, clip_value=1.0):
        for epoch in range(epochs):
            # Forward pass
            output_probs = self.forward(X)
            
            # Backward pass
            self.backward(X, y, learning_rate, clip_value)
            
            # Print loss (optional)
            if epoch % 100 == 0:
                loss = -np.log(output_probs[range(len(X)), y]).mean()
                print(f'Epoch {epoch}: Loss {loss:.4f}')
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Define the dimensions of the neural network
input_dim = X.shape[1]
hidden_dim = 5  # You can adjust the number of hidden units
output_dim = len(np.unique(y))

# Initialize and train the MLP
mlp = MLP(input_dim, hidden_dim, output_dim)
mlp.train(X, y, learning_rate=0.01, epochs=100)

# Make predictions
predictions = mlp.predict(X)

# Evaluate the model
accuracy = np.mean(predictions == y)
print(f'Accuracy: {accuracy}')
