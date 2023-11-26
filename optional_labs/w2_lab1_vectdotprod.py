import numpy as np

def my_dot(a, b):
    """
    Compute the dot product of two vectors
 
    Args:
      a (ndarray (n,)):  input vector 
      b (ndarray (n,)):  input vector with same dimension as a
    
    Returns:
      x (scalar): 

    """
    x = 0
    n = a.shape[0]
    for i in range(n):
        x += a[i] * b[i]
    return x

a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"Dot Product: {my_dot(a, b)}")
c = np.dot(a, b)
print(f"NP dot product: {c}")