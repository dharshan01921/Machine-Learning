import numpy as np
# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)
# Input dataset (XOR)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
# Output labels
y = np.array([[0],[1],[1],[0]])
# Initialize weights randomly
np.random.seed(1)
input_hidden_weights = np.random.uniform(size=(2,2))
hidden_output_weights = np.random.uniform(size=(2,1))
# Learning rate
lr = 0.1
# Training the ANN
for epoch in range(10000):
    # -------- Forward Propagation --------
    hidden_input = np.dot(X, input_hidden_weights)
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, hidden_output_weights)
    predicted_output = sigmoid(final_input)
    # -------- Backpropagation --------
    error = y - predicted_output
    d_output = error * sigmoid_derivative(predicted_output)
    error_hidden = d_output.dot(hidden_output_weights.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)
    # -------- Weight Update --------
    hidden_output_weights += hidden_output.T.dot(d_output) * lr
    input_hidden_weights += X.T.dot(d_hidden) * lr
# Testing
print("Input  Output")
for i in range(len(X)):
    print(X[i], " ", round(predicted_output[i][0]))
