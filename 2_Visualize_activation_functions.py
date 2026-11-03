import numpy as np
import matplotlib.pyplot as plt
from math import erf

# Generate Input
x = np.linspace(-5, 5, 1000)

# Activation Functions

# 1. Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 2. Tanh
def tanh(x):
    return np.tanh(x)

# 3. ReLU
def relu(x):
    return np.maximum(0, x)

# 4. Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# 5. PReLU
def prelu(x, alpha=0.25):
    return np.where(x > 0, x, alpha * x)

# 6. RReLU (using average training slope)
def rrelu(x, lower=1/8, upper=1/3):
    alpha = (lower + upper) / 2
    return np.where(x > 0, x, alpha * x)

# 7. ELU
def elu(x, alpha=1):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


# 8. SELU
def selu(x):
    alpha = 1.67326
    scale = 1.0507
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

# 9. Softplus
def softplus(x):
    return np.log1p(np.exp(x))

# 10. Softsign
def softsign(x):
    return x / (1 + np.abs(x))

# 11. Swish (SiLU)
def swish(x):
    return x * sigmoid(x)

# 12. Hard Swish
def hard_swish(x):
    return x * np.clip((x + 3) / 6, 0, 1)

# 13. Mish
def mish(x):
    return x * np.tanh(np.log1p(np.exp(x)))

# 14. GELU
def gelu(x):
    return 0.5 * x * (1 + np.vectorize(erf)(x / np.sqrt(2)))

# 15. Hard Sigmoid
def hard_sigmoid(x):
    return np.clip((x + 3) / 6, 0, 1)

# 16. Hard Tanh
def hard_tanh(x):
    return np.clip(x, -1, 1)

# 17. TanhShrink
def tanh_shrink(x):
    return x - np.tanh(x)

# 18. SoftShrink
def soft_shrink(x, lambd=0.5):
    return np.where(x > lambd,x - lambd,np.where(x < -lambd, x + lambd,0))

# 19. HardShrink
def hard_shrink(x, lambd=0.5):
    return np.where(np.abs(x) > lambd, x, 0)

# 20. Linear
def linear(x):
    return x

# Function List

functions = [
    ("Sigmoid", sigmoid),
    ("Tanh", tanh),
    ("ReLU", relu),
    ("Leaky ReLU", leaky_relu),
    ("PReLU", prelu),
    ("RReLU", rrelu),
    ("ELU", elu),
    ("SELU", selu),
    ("Softplus", softplus),
    ("Softsign", softsign),
    ("Swish", swish),
    ("Hard Swish", hard_swish),
    ("Mish", mish),
    ("GELU", gelu),
    ("Hard Sigmoid", hard_sigmoid),
    ("Hard Tanh", hard_tanh),
    ("TanhShrink", tanh_shrink),
    ("SoftShrink", soft_shrink),
    ("HardShrink", hard_shrink),
    ("Linear", linear)
]

# Plot

rows = 4
cols = 5

plt.figure(figsize=(15, 22))   # Increase figure size

for i, (name, func) in enumerate(functions):
    plt.subplot(rows, cols, i + 1)
    plt.plot(x, func(x), linewidth=2)
    plt.title(name, fontsize=12, pad=10)
    plt.xlabel("Input", fontsize=8)
    plt.ylabel("Output", fontsize=8)
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

# Add spacing between plots
plt.tight_layout(pad=4.0, h_pad=4.0, w_pad=3.5)

# OR you can additionally use:
plt.subplots_adjust(hspace=0.7, wspace=0.5)

plt.show()

