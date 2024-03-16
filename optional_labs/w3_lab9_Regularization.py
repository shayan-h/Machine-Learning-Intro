"""
This code defines a function `compute_cost_linear_reg` that computes the cost over all examples for linear regression with regularization. It also includes a test case to demonstrate the usage of the function.

The `compute_cost_linear_reg` function takes the following arguments:
- X: Data, an ndarray of shape (m, n) where m is the number of examples and n is the number of features.
- y: Target values, an ndarray of shape (m,) representing the target values for each example.
- w: Model parameters, an ndarray of shape (n,) representing the weights of the linear regression model.
- b: Model parameter, a scalar representing the bias term of the linear regression model.
- lambda_: Regularization parameter, a scalar that controls the amount of regularization.

The function computes the cost by iterating over all examples and calculating the squared difference between the predicted value and the actual value. It then adds a regularization term to the cost based on the weights. Finally, it returns the total cost.

The code also includes a test case where random data is generated and the `compute_cost_linear_reg` function is called with the generated data and parameters. The resulting regularized cost is printed.

Example usage:
--------------
np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
print("X: ", X_tmp)
print("y: ", y_tmp)
print("w: ", w_tmp)
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)
--------------
"""

import numpy as np
np.set_printoptions(precision=3)

def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples for linear regression with regularization.
    
    Args:
      X (ndarray (m,n)): Data, m examples with n features.
      y (ndarray (m,)): Target values.
      w (ndarray (n,)): Model parameters.
      b (scalar): Model parameter.
      lambda_ (scalar): Controls amount of regularization.
      
    Returns:
      total_cost (scalar): Cost.
    """
    cost = 0.0
    reg = 0.0
    m = X.shape[0]
    n = X.shape[1]
    for i in range(m):
        func = np.dot(X[i], w) + b
        cost += (func - y[i])**2
    cost /= 2*m

    for j in range(n):
        reg += w[j]**2
    reg *= (lambda_/2*m)

    cost += reg
    return cost

np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
print("X: ", X_tmp)
print("y: ", y_tmp)
print("w: ", w_tmp)
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)





