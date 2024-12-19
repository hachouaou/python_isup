"""
Fichier pour tester les differentes fonctions avec pytest
"""
import numpy as np
import pytest
from linearmodel.ordinary_least_squares import OrdinaryLeastSquares

def test_fit_and_predict():
    """
    Teste les méthodes fit et predict du modèle.
    """
    # Données d'exemple
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])  # Relation linéaire parfaite : y = 2x

    # Initialisation du modèle
    model = OrdinaryLeastSquares(intercept=True)
    model.fit(X, y)

    # Vérification des coefficients
    coeffs = model.get_coeffs()
    assert np.isclose(coeffs[0], 0, atol=1e-5)  # Intercept doit être proche de 0
    assert np.isclose(coeffs[1], 2, atol=1e-5)  # Coefficient pour x doit être proche de 2

    # Test des prédictions
    y_pred = model.predict(X)
    assert np.allclose(y_pred, y, atol=1e-5)

def test_determination_coefficient():
    """
    Teste le calcul du coefficient de determination.
    """
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])  # Relation linéaire parfaite : y = 2x

    # Initialisation et entraînement du modèle
    model = OrdinaryLeastSquares(intercept=True)
    model.fit(X, y)

    # Vérification de R^2
    r_squared = model.determination_coefficient(y, model.y_pred)
    assert np.isclose(r_squared, 1.0, atol=1e-5)  # R^2 doit être 1.0 pour une relation parfaite

def test_residuals():
    """
    Teste que les résidus sont bien proches de 0.
    """
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    model = OrdinaryLeastSquares(intercept=True)
    model.fit(X, y)

    # Vérification que les résidus sont proches de zéro pour une relation parfaite
    residuals = model.residuals
    assert np.allclose(residuals, 0, atol=1e-5)

def test_non_invertible_matrix():
    """
    Teste comment le modèle agit face à une matrice non inversible.
    """
    # Données non inversibles
    X = np.array([[1, 1], [1, 1], [1, 1]])  # Colonnes identiques
    y = np.array([1, 1, 1])

    model = OrdinaryLeastSquares(intercept=False)

    # Utilisation de re.escape pour une correspondance exacte
    with pytest.raises(ValueError, match=r"La matrice X\^T X n'est pas inversible\."):
        model.fit(X, y)

def test_dimension_mismatch():
    """
    Teste comment le modèle agit lorsque les dimensions de X et y ne correspondent pas.
    """
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2])  # Longueur différente

    model = OrdinaryLeastSquares(intercept=True)

    with pytest.raises(ValueError, match="Les dimensions de X et y ne correspondent pas."):
        model.fit(X, y)
