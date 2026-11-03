import numpy as np

# ==========================================
# Non-linear XOR Input Data
# ==========================================
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# ==========================================
# RBF Network Parameters
# ==========================================
input_neurons = 2
hidden_neurons = 4
output_neurons = 1
learning_rate = 0.5
sigma = 1.0
epochs = 1000

# ==========================================
# Hidden Layer Centers
# ==========================================
centers = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# ==========================================
# Initial Manual Weights
# w1 w2 w3 w4
# ==========================================
W = np.array([
    -0.8,  # w1
    0.9,   # w2
    0.9,   # w3
    -0.8   # w4
])

bias = 0.1

print("Initial Weights")
print("w1 =", W[0])
print("w2 =", W[1])
print("w3 =", W[2])
print("w4 =", W[3])
print("Initial Bias =", bias)


# ==========================================
# Gaussian RBF Activation Function
# ==========================================
def gaussian_rbf(x, center, sigma):
    distance = np.linalg.norm(x - center)
    activation = np.exp(
        -(distance ** 2) /
        (2 * sigma ** 2)
    )
    return activation


# ==========================================
# Function to generate XOR target
# No Y array used
# ==========================================
def xor_target(x):
    if np.array_equal(x, [0, 0]):
        return 0
    elif np.array_equal(x, [1, 1]):
        return 0
    else:
        return 1


# ==========================================
# Training using Gradient Descent
# ==========================================
for epoch in range(epochs):
    total_error = 0

    for x in X:
        # Generate target internally
        target = xor_target(x)

        # --------------------------
        # Hidden Layer
        # --------------------------
        H = []
        for c in centers:
            H.append(
                gaussian_rbf(
                    x,
                    c,
                    sigma
                )
            )
        H = np.array(H)

        # --------------------------
        # Output Layer
        # y=w1h1+w2h2+w3h3+w4h4+b
        # --------------------------
        y = np.dot(W, H) + bias

        # --------------------------
        # Error
        # --------------------------
        error = target - y
        total_error += error ** 2

        # --------------------------
        # Gradient Descent Update
        # --------------------------
        W = W + learning_rate * error * H
        bias = bias + learning_rate * error

    # Print every 100 epochs
    if epoch % 100 == 0:
        print("\nEpoch:", epoch)
        print(
            "Error:",
            round(total_error, 4)
        )
        print("Weights:")
        print("w1 =", round(W[0], 4))
        print("w2 =", round(W[1], 4))
        print("w3 =", round(W[2], 4))
        print("w4 =", round(W[3], 4))
        print(
            "Bias =",
            round(bias, 4)
        )

# ==========================================
# Final Trained Weights
# ==========================================
print("\n==============================")
print("Final Trained Weights")
print("==============================")
print("w1 =", round(W[0], 4))
print("w2 =", round(W[1], 4))
print("w3 =", round(W[2], 4))
print("w4 =", round(W[3], 4))
print(
    "Final Bias =",
    round(bias, 4)
)

# ==========================================
# Testing
# ==========================================
print("\nTesting\n")

for x in X:
    # Hidden Layer
    H = []
    for c in centers:
        H.append(
            gaussian_rbf(
                x,
                c,
                sigma
            )
        )
    H = np.array(H)

    # Output
    y = np.dot(W, H) + bias

    # Classification
    prediction = 1 if y >= 0.5 else 0

    print(
        "Input:",
        x,
        "Raw Output:",
        round(y, 2),
        "Class Output:",
        f"{prediction:.2f}"
    )
