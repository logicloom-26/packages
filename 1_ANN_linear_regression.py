import numpy as np                      # Import NumPy library for numerical operations

# Training Dataset
# X = Input Features (X1, X2)
# Y = Target Output

X = np.array([
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7]
], dtype=float)                         # Convert inputs into floating-point array

Y = np.array([
    [8],
    [11],
    [14],
    [17],
    [20]
], dtype=float)                         # Convert outputs into floating-point array

# Initialize Weights and Bias
# These are random starting values.
# During training they will be updated.

W1 = 0.5                                # Weight corresponding to X1
W2 = 0.3                                # Weight corresponding to X2
b = 0.2                                 # Bias

learning_rate = 0.01                    # Step size for updating weights
epochs = 10000                          # Maximum number of training iterations


# Display Initial Parameters
print("Initial Weights")
print("W1 =", W1)
print("W2 =", W2)
print("Bias =", b)
print("-"*50)

# Training Process
# Repeat until the error becomes very small

for epoch in range(epochs):

    total_error = 0                     # Reset total error for each epoch

    # Loop through every training sample
    for i in range(len(X)):

        # Get Inputs and Target Output
        x1 = X[i][0]                    # First input feature
        x2 = X[i][1]                    # Second input feature
        target = Y[i][0]                # Expected output

        # Forward Propagation
        # Net = X1*W1 + X2*W2 + Bias
        net = (x1 * W1) + (x2 * W2) + b

        # Linear Activation Function
        # Output = Net
        output = net

        # Calculate Error
        # Error = Target - Prediction
        error = target - output

        # Add Squared Error for MSE calculation
        total_error += error ** 2

        # Backpropagation
        # Calculate Gradients

        dW1 = -error * x1               # Gradient of W1
        dW2 = -error * x2               # Gradient of W2
        dB = -error                     # Gradient of Bias

        # Gradient Descent
        # Update Weights and Bias
        # New Weight = Old Weight - LR × Gradient

        W1 = W1 - learning_rate * dW1
        W2 = W2 - learning_rate * dW2
        b = b - learning_rate * dB

    # Mean Squared Error (MSE)

    mse = total_error / len(X)

    # Print MSE after every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}  MSE = {mse:.6f}")

    # Stop training if error becomes very small
    if mse < 0.00001:
        print("\nTraining Completed")
        print("Epoch =", epoch)
        break


# Display Final Learned Parameters

print("\nFinal Parameters")
print("W1 =", round(W1, 4))
print("W2 =", round(W2, 4))
print("Bias =", round(b, 4))


# Testing / Prediction

print("\nPredictions")
print("-"*50)

for i in range(len(X)):

    x1 = X[i][0]                        # First input
    x2 = X[i][1]                        # Second input

    # Calculate prediction using trained weights
    predicted = (x1 * W1) + (x2 * W2) + b

    # Display actual output, predicted output and error
    print(
        "Input =", X[i],
        " Actual =", Y[i][0],
        " Predicted =", round(predicted, 3),
        " Error =", round(Y[i][0] - predicted, 5)
    )

