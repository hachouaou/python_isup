import numpy as np
from ordinary_least_squares import OrdinaryLeastSquares

def test_fit():
    X = np.array([[1, 2], [2, 4], [3, 6]])
    y = np.array([1, 2, 3])
    model = OrdinaryLeastSquares(intercept=True)
    model.fit(X, y)
    assert model.get_coeffs() is not None

def test_r_squared():
    X = np.array([[1, 2], [2, 4], [3, 6]])
    y = np.array([1, 2, 3])
    model = OrdinaryLeastSquares(intercept=True)
    model.fit(X, y)
    r_squared = model.determination_coefficient(y, model.y_pred)
    assert 0 <= r_squared <= 1