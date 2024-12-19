## Présentation

Ce fichier permet de décrire le projet $linearmodel$ qui vise à créer un package python pour l'analyse d'un jeu de données et la construction d'un modèle linéaire.

Le fichier 'my_statistics' regroupe différentes fonctions pour les analyses statistiques du jeu de données fourni.

Le fichier 'visualization' regroupe différentes fonctions pour l'affichage et la visualisation des données.

Enfin, le modèle linéaire est codé dans le fichier 'ordinary_least_squares' qui permet d'expliquer le taux d’emission de CO2 en fonction de certaines covariables du jeu de données.

Pour cela, on a implémenter la méthode des moindres carrés ordinaires :

Considérons un modèle linéaire représenté par :

$$
y = X \beta + \varepsilon,


où :
- y est un vecteur de dimension \( n \),
- X est une matrice de dimension \( n \times d \),
- \beta  est un vecteur de paramètres inconnus de dimension \( d \),
- \varepsilon est le vecteur des erreurs de dimension \( n \).

L’estimateur des moindres carrés est donné par :


\hat{\beta} = (X^T X)^{-1} X^T y.
$$

## Installation

Pour exécuter les scripts dans ce fichier certaines bibliothèques python sont nécessaires, exécuter la commande suivante dans le dossier téléchargé avec votre env python activé avant de commencer :
```
pip install .
```

## Utilisation

Le script principal est dans le fichier `main_linear.py`. Pour l'utiliser exécuter :
```
python main_linear.py
```
et suivez les instructions !
