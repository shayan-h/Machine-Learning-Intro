"""
The following code uses sklearn.linear_model.LogisticRegression. It simply takes a
dataset of 6 training examples and 2 features and a binary classification output, meaning
there are only 2 possible outputs, 1 or 0. The model is fitted with this data and
the program proceeds to predict the output given a brand new dataset. The accuracy on the
training set was 1.0
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])
new_X = np.array([[1, 1], [0.5, 0.5], [2.5, 3], [0.9, 2], [1, 2.5], [4, 4]])

lr_model = LogisticRegression()
lr_model.fit(X, y)

y_pred = lr_model.predict(new_X)
print("Prediction on training set:", y_pred)
print("Accuracy on training set:", lr_model.score(X, y))