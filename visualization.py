"""
Fichier réunissant les différentes visualisations de la base de données afin de mieux la comprendre
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def donnees_sans_modifs(csv):
    """
    affiche les données brutes
    """
    return pd.read_csv(csv)

def doublons(df):
    """
    Vérifie s'il n'ya pas de doublons dans la bdd
    """
    return df[df.duplicated()]

def nb_class_vehicles(data):
    """
    Affiche le nombre total par type de voitures
    """
    sns.countplot(data, x='Vehicle class', order=data['Vehicle class'].value_counts().index)
    plt.xticks(rotation=90)
    plt.show()

def compte_marque(data):
    """
    Affiche le nombre de voiture pour chaque marque
    """
    sns.countplot(data, x='Make', order=data['Make'].value_counts().index)
    plt.xticks(rotation = 90)
    plt.show()

def compte_annee(data):
    """
    Compte le nombre de voiture par an
    """
    sns.countplot(data, x='Model year')
    plt.show()

def heatmap(data):
    """
    Génère une heatmap afin de mieux visualiser les différentes relations entre chaque colonnes
    """
    numerical_cols = data.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(8,6))
    sns.heatmap(numerical_cols.corr(), annot=True, cmap='coolwarm')
    plt.show()

def relation_moteur_emission(data):
    """
    Affiche la relation entre la taille du moteur et les émissions de CO2
    """
    sns.scatterplot(data, x='Engine size (L)', y='CO2 emissions (g/km)')
    plt.title("Relation entre la taille du moteur et les émissions de CO2")
    plt.xlabel("Taille du moteur (L)")
    plt.ylabel("Émissions de CO2 (g/km)")
    plt.show()

def relation_conso_emission(data):
    """
    Affiche la relation entre la consommation combinée et les émissions de CO2
    """
    sns.scatterplot(data, x='Combined (L/100 km)', y='CO2 emissions (g/km)')
    plt.title("Consommation vs. Émissions de CO2")
    plt.xlabel("Consommation (L/100km)")
    plt.ylabel("Émissions de CO2 (g/km)")
    plt.show()

def relation_cylindres_emission(data):
    """
    Affiche la relation entre le nombre de cylindres et les émissions de CO2
    """
    sns.scatterplot(data, x='Cylinders', y='CO2 emissions (g/km)')
    plt.title("Nombre de cylindres vs. Émissions de CO2")
    plt.xlabel("Nombre de cylindres")
    plt.ylabel("Émissions de CO2 (g/km)")
    plt.show()

def relation_ville_autoroute(data):
    """
    Affiche la relation entre la consommation en ville et la consommation en autoroute
    """
    sns.scatterplot(data, x='City (L/100 km)', y='Highway (L/100 km)')
    plt.title("Consommation en ville vs. autoroute")
    plt.xlabel("Consommation en ville (L/100km)")
    plt.ylabel("Consommation sur autoroute (L/100km)")
    plt.show()

def distrib_emission(data):
    """
    Affiche la distribution des emissions de CO2
    """
    sns.histplot(data, x='CO2 emissions (g/km)', kde=True, bins=20, color='blue')
    plt.title("Distribution des émissions de CO2 (g/km)")
    plt.xlabel("Émissions de CO2 (g/km)")
    plt.ylabel("Fréquence")
    plt.show()

def distrib_combined(data):
    """
    Affiche la distribution de la consommation combinée
    """
    sns.histplot(data, x='Combined (L/100 km)', kde=True,
                 bins=20, color='green')
    plt.title("Distribution de la consommation combinée (L/100km)")
    plt.xlabel("Consommation combinée (L/100km)")
    plt.ylabel("Fréquence")
    plt.show()

def distrib_fuel_type(data):
    """
    Affiche la distribution du type d'essence
    """
    sns.histplot(data, x='CO2 emissions (g/km)', hue='Fuel type',
                 kde=True, bins=20, palette='viridis')
    plt.title("Émissions de CO2 par type de carburant")
    plt.xlabel("Émissions de CO2 (g/km)")
    plt.ylabel("Fréquence")
    plt.show()
