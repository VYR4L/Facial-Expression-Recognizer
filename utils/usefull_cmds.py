# w = np.random.randn(n) * np.sqrt(1/n) 
# Forward Propagation: z = np.dot(X, w)
# Loss Function: -np.mean(y * np.log(z) + (1 - y) * np.log(1 - z))
# Backward Propagation: dz = z - y
# Gradient Descent: w = w - alpha * np.dot(X.T, dz)/m
# Predict: z = np.dot(X, w)
# Accuracy: np.mean(np.round(z) == y)



import matplotlib.pyplot as plt
import numpy as np

# Sigmoid Function: 1/(1 + np.exp(-z))
x = np.linspace(-5, 5, 50)
z = 1/(1 + np.exp(-x))

plt.subplots(figsize=(8, 5))
plt.plot(x, z)
plt.grid()
plt.show()

# Tanh Function: (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
x = np.linspace(-5, 5, 50)
z = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)) # or np.tanh(x)

plt.subplots(figsize=(8, 5))
plt.plot(x, z)
plt.grid()
plt.show()

# Rectified Linear Unit (ReLU): np.maximum(0, z)
x = np.linspace(-5, 5, 50)
z = np.maximum(0, x)

plt.subplots(figsize=(8, 5))
plt.plot(x, z)
plt.grid()
plt.show()

