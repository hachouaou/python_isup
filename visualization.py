import pandas as pd
import seaborn as sns

def donnees_sans_modifs(csv):
    return pd.read_csv(csv)

def doublons(df):
    return df[df.duplicated()]
