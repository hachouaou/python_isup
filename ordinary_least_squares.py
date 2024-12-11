import numpy as np
from matplotlib import pyplot as plt

class OrdinaryLeastSquares:
    """
    Attributs :
        - intercept : Si True, ajoute une constante au modèle
        - coeffs : Coefficients estimés du modèle
        - y_pred : Prédictions du modèle sur les données d'entrainement
        - residuals : Différence entre les valeurs réelles et prédites
        - confience_intervals : Intervalles de confiance des coefficients
    """
    def __init__(self, intercept = True):
        self.intercept = intercept
        self.coeffs = None
        self.y_pred = None
        self.residuals = None
        self.confidence_intervals = None

    def fit(self, X, y):
        """
        Ajuste le modèle aux données d'entrainement.
        """
        if self.intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        X_transpose = X.T
        try:
            XtX_inv = np.linalg.inv(X_transpose @ X)
        except:
            raise ValueError("La matrice X^T X n'est pas inversible.")
        
        self.coeffs = XtX_inv @ X_transpose @ y
        self.y_pred = X @ self.coeffs
        self.residuals = y-self.y_pred

        n, d = X.shape
        mse = np.sum(self.residuals ** 2) / (n - d)
        se = np.sqrt(mse*np.diag(XtX_inv))
        t_critical = 1.96

        self.confidence_intervals = [
            (self.coeffs[i] - t_critical * se[i], self.coeffs[i] + t_critical * se[i])
            for i in range(d)
        ]
    
    def predict(self, Xt):
        """
        Retourne les prédictions à partir de nouvelles données de la matrice Xt.

        """
        if self.intercept:
            Xt = np.hstack((np.ones((Xt.shape[0], 1)), Xt))

        return Xt @ self.coeffs
    
    def get_coeffs(self):
        """
        Retourne les coefficients estimés.
        """
        return self.coeffs
    
    def determination_coefficient(self, y_true, y_pred):
        """
        Calcule le coefficient de déterminations R^2.
        """
        ss_total = np.sum((y_true - np.mean(y_true))**2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        
        return 1 - (ss_residual / ss_total)
    
    def plot_residuals(self):
        """
        Affiche un graphique des résidus.
        """
        plt.scatter(range(len(self.residuals)), self.residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.title("Graphique des résidus")
        plt.xlabel("Index")
        plt.ylabel("Résidu")
        plt.show()

    def plot_predictions(self, y_true):
        """
        Affiche un graphique des valeurs prédites vs réelles.

        Paramètres :
            - y_true (np.array) : Valeurs réelles.
        """
        plt.scatter(y_true, self.y_pred)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red')
        plt.title("Prédictions vs Valeurs réelles")
        plt.xlabel("Valeurs réelles")
        plt.ylabel("Valeurs prédites")
        plt.show()