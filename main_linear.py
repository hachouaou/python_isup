import visualization as visu
import my_statistics as stats

csv = 'vehicles.csv'

data_brut = visu.donnees_sans_modifs(csv)

#print(data_brut.head(5))

#print(stats.summary(data_brut))
#print(data_brut.describe())

print(visu.doublons(data_brut))