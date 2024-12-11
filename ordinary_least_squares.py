import numpy as np
from matplotlib import pyplot as plt

class OrdinaryLeastSquares:
    def __init__(self, intercept = True):
        self.intercept = intercept
        self.coeffs = None
        self.y_pred = None
        self.residuals = None
        self.confidence_intervals = None

        def fit(self, X, y):
            if self.intercept:
                X = np.hstack((np.ones((X.shape[0], 1)), X))

            X_transpose = X.T
            XtX_inv = np.linalg.inv(X_transpose @ X)
            self.coeffs = XtX_inv @ X_transpose @ y

            self.y_pred = X @ self.coeffs

            self.residuals = y-selfy_pred

            n, d = X.shape
            mse = np.sum(self.residuals ** 2) / (n - d)
            se = np.sqrt(mse*np.diag(XtX_inv))
            t_critical = 1.96

            self.confidence_intervals = [
                (self.coeffs[i] - t_critical * se[i], self.coeffs[i] + t_critical * se[i])
                for i in range(d)
            ]