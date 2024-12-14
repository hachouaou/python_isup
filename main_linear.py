import visualization as visu
import my_statistics as stats
import ordinary_least_squares as reg

csv = 'vehicles.csv'

#Charger les données brutes
data_brut = visu.donnees_sans_modifs(csv)
print(data_brut.head(5))

#Résumé statistique
print(stats.summary(data_brut))
print(data_brut.describe())

#Vérification des doublons
print(visu.doublons(data_brut))

feature_columns = ["Engine size (L)", "Cylinders", "City (L/100 km)", "Highway (L/100 km)", "Combined (L/100 km)"] #Variables explicatives
target_column = "CO2 emissions (g/km)"

#Préparation des données pour la régression
X = data_brut[feature_columns].values
y = data_brut[target_column].values

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
