# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create sample dataset
np.random.seed(0)

X = np.linspace(-3, 3, 100)
y = np.sin(X) + np.random.normal(0, 0.2, 100)

# Convert to matrix form
X_mat = np.c_[np.ones(len(X)), X]

# Step 2: Define LWLR function
def lwlr(test_point, X, y, tau=0.5):

    m = X.shape[0]
    W = np.eye(m)

    for i in range(m):
        diff = test_point - X[i]
        W[i, i] = np.exp(-np.dot(diff, diff) / (2 * tau ** 2))

    XT_W_X = X.T @ W @ X

    if np.linalg.det(XT_W_X) == 0:
        return 0

    theta = np.linalg.inv(XT_W_X) @ X.T @ W @ y

    return test_point @ theta


y_pred = []

for i in range(len(X)):
    y_pred.append(lwlr(X_mat[i], X_mat, y, tau=0.5))

y_pred = np.array(y_pred)

# Step 3: Plot results
plt.figure(figsize=(8,5))

plt.scatter(X, y, color='blue', label='Original data')
plt.plot(X, y_pred, color='red', label='LWLR fit')

plt.title("Locally Weighted Regression")
plt.legend()

plt.show()
