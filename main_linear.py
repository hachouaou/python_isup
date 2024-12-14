"""
Main de la regression linéaire
"""

import visualization as visu
import my_statistics as stats
import ordinary_least_squares as reg

data = 'vehicles.csv'

#Charger les données brutes
data_brut = visu.donnees_sans_modifs(data)

#On drop les differentes colonnes qui ne servent pas au modèle
data_filtered = data_brut.drop(['Model year', 'Make', 'Model', 'Vehicle class',
                                'Transmission', 'Fuel type', 'CO2 rating', 'Smog rating'], axis = 1)

#Résumé statistique
print(stats.summary(data_filtered))

#Vérification des doublons
print(visu.doublons(data_filtered))

#Visualisation des relations
visu.heatmap(data_filtered)
visu.relation_moteur_emission(data_filtered)
visu.relation_conso_emission(data_filtered)
visu.relation_cylindres_emission(data_filtered)
visu.relation_ville_autoroute(data_filtered)

#Visualisation des différentes distribution
visu.distrib_combined(data_filtered)
visu.distrib_emission(data_filtered)

#On met en place le modèle
feature_columns = ["Engine size (L)", "Cylinders", "City (L/100 km)", "Highway (L/100 km)",
                   "Combined (L/100 km)", "Combined (mpg)"]
target_column = "CO2 emissions (g/km)"

#Préparation des données pour la régression
X = data_filtered[feature_columns].values
y = data_filtered[target_column].values

#Création et ajustement du modèle
model = reg.OrdinaryLeastSquares(intercept=True)
model.fit(X, y)

#Calul du coefficient de déterminatation
r_squared = model.determination_coefficient(y, model.y_pred)
print(f"Le coefficient de détermination (R^2): {r_squared}")

#Visualisation des résultats
print("Visualisation des résultats...")
model.plot_residuals()
model.plot_predictions(y)
model.plot_absolute_residuals()
model.plot_residual_distribution()
