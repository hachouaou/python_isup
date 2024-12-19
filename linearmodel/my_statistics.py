import pandas as pd

def minimum(dt, col):
    """
    Retourne la valeur minimale d'une colonne donnée
    """
    return dt[col].min()

def maximum(dt, col):
    """
    Retourne la valeur maximale d'une colonne donnée
    """
    return dt[col].max()

def mean(dt, col):
    """
    Calcul et retourne la moyenne d'une colonne donnée
    """
    return sum(dt[col]) / len(dt[col])

def quartiles(dt, col):
    """
    Retourne les différents quartiles d'une colonne donnée
    """
    q1 = dt[col].quantile(0.25)
    q2 = dt[col].quantile(0.50) #mediane
    q3 = dt[col].quantile(0.75)

    return q1, q2, q3

def standard_deviation(dt, col):
    """
    Calcule et retourne l'écart type d'une colonne donnée
    """
    moy = mean(dt, col)
    return (sum((dt[col] - moy) ** 2) / len(dt[col])) ** (1/2)

def summary(dt):
    """
    Retourne le résumé statistique de chaque colonne de la base de données
    """
    summary_data = []

    for col in dt.columns:
        # Vérifier si la colonne est de type numérique
        if pd.api.types.is_numeric_dtype(dt[col]):
            # Calculer les statistiques descriptives
            taille = dt[col].count()  # Nombre de valeurs non nulles
            moyenne = mean(dt, col)  # Moyenne
            ecart_type = standard_deviation(dt, col)  # Écart-type
            min_value = minimum(dt, col)  # Minimum
            Q1, med, Q3 = quartiles(dt, col)
            max_value = maximum(dt, col)  # Maximum

            # Ajouter les statistiques à la liste
            summary_data.append({
                "Column": col,
                "Count": taille,
                "Mean": moyenne,
                "Std Dev": ecart_type,
                "Min": min_value,
                "25%": Q1,
                "50%": med,
                "75%": Q3,
                "Max": max_value
            })

    # Créer un DataFrame pour afficher les résultats
    summary_df = pd.DataFrame(summary_data)
    summary_df.set_index("Column", inplace=True)
    return summary_df
