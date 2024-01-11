"""

"""

import numpy as np
np.set_printoptions(precision=3)

def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
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





