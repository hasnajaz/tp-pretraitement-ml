import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#charger le dataset titanic
df=sns.load_dataset('titanic')
#analyse exploratoire du dataset
print("les premiere lignes du dataset:" , df.head())
print("\nInformations sur le dataset:")
df.info()
print("\nTypes de données:")
print(df.dtypes)
print("\nDimensions du dataset:")
print(df.shape)
print("\nStatistiques descriptives:")
print(df.describe())
#nettoyage des données
print("\nNombre de valeurs manquantes par colonne:")
print(df.isnull().sum())
#calculer le pourcentage de valeurs manquantes par colonne
valeurs_manquantes = (df.isnull().sum()/len(df))*100
print("\nPourcentage de valeurs manquantes par colonne:",valeurs_manquantes)
#supprimer les colonnes avec plus de 40% de valeurs manquantes
colonnes = valeurs_manquantes[valeurs_manquantes > 40].index
df = df.drop(columns=colonnes)
print("Colonnes supprimées :", list(colonnes))
# Division des données
from sklearn.model_selection import train_test_split
X = df.drop(columns=["survived", "alive"])
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y)
print("Taille de X_train :", X_train.shape)
print("Taille de X_test :", X_test.shape)
#identification des colonnes numériques et catégorielles
colonnes_numeriques = X_train.select_dtypes(include="number").columns
colonnes_categorielles = X_train.select_dtypes(include=["object", "category", "bool"]).columns
print("Variables numériques :", list(colonnes_numeriques))
print("Variables catégorielles :", list(colonnes_categorielles))
from sklearn.impute import SimpleImputer
#remplacer les valeurs manquantes par la moyenne pour les colonnes numériques
var_num = SimpleImputer(strategy="median")
X_train[colonnes_numeriques] = var_num.fit_transform(X_train[colonnes_numeriques])
X_test[colonnes_numeriques] = var_num.transform(X_test[colonnes_numeriques])
#remplacer les valeurs manquantes par la valeur la plus fréquente pour les colonnes catégorielles
var_cat = SimpleImputer(strategy="most_frequent")
X_train[colonnes_categorielles] = var_cat.fit_transform(X_train[colonnes_categorielles])
X_test[colonnes_categorielles] = var_cat.transform(X_test[colonnes_categorielles])
#gestion des valeurs aberrantes
for col in colonnes_numeriques:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=X_train[col])
    plt.title("Boxplot de " + col)
    plt.show()
    # Calcul de Q1 et Q3
for col in colonnes_numeriques:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    # Calcul de IQR
    IQR = Q3 - Q1
    # Définir les limites
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR
    # Calculer la médiane sur Train
    mediane = X_train[col].median()
    # Compter les valeurs aberrantes
    nb_aberrantes = ((X_train[col] < limite_inf)|(X_train[col] > limite_sup)).sum()
    print(col, ":", nb_aberrantes, "valeurs aberrantes")
    # Remplacer les valeurs aberrantes dans Train
    X_train.loc[(X_train[col] < limite_inf)|(X_train[col] > limite_sup), col]=mediane
    # Remplacer les valeurs aberrantes dans Test
    X_test.loc[(X_test[col] < limite_inf)|(X_test[col] > limite_sup), col]=mediane
#transformation des donnes
from sklearn.preprocessing import OneHotEncoder
# Identifier les colonnes catégorielles
colonnes_cat= X_train.select_dtypes(include=["object", "category", "bool"]).columns
print("Colonnes catégorielles :")
print(list(colonnes_cat))
# Appliquer One-Hot Encoding
encoder = OneHotEncoder(handle_unknown="ignore",sparse_output=False)
X_train_cat = encoder.fit_transform(X_train[colonnes_cat])
X_test_cat = encoder.transform(X_test[colonnes_cat])
# Transformer les résultats en DataFrame
noms_colonnes = encoder.get_feature_names_out(colonnes_cat)
X_train_cat = pd.DataFrame( X_train_cat,columns=noms_colonnes,index=X_train.index)
X_test_cat = pd.DataFrame( X_test_cat,columns=noms_colonnes,index=X_test.index)
print("Résultat de l'encodage :")
print(X_train_cat.head())
from sklearn.preprocessing import MinMaxScaler
# Identifier les colonnes numériques
colonnes_numeriques = X_train.select_dtypes(include="number").columns
# Normalisation
normaliseur = MinMaxScaler()
X_train_num_minmax = normaliseur.fit_transform(X_train[colonnes_numeriques])
X_test_num_minmax = normaliseur.transform(X_test[colonnes_numeriques])
X_train_num_minmax = pd.DataFrame(X_train_num_minmax,columns=colonnes_numeriques,index=X_train.index)
print("Données après normalisation :")
print(X_train_num_minmax.head())
# Créer le standardiseur
from sklearn.preprocessing import StandardScaler
standardiseur = StandardScaler()
# Standardisation
X_train_num_std = standardiseur.fit_transform(X_train[colonnes_numeriques])
X_test_num_std = standardiseur.transform( X_test[colonnes_numeriques])
# Transformer en DataFrame
X_train_num_std = pd.DataFrame(X_train_num_std,columns=colonnes_numeriques,index=X_train.index)
print("Données après standardisation :")
print(X_train_num_std.head())
# Choisir une variable numérique
col = "age"
# Afficher les histogrammes
plt.figure(figsize=(15, 4))
# Avant transformation
plt.subplot(1, 3, 1)
sns.histplot(X_train[col], kde=True)
plt.title("Avant transformation")
# Après normalisation
plt.subplot(1, 3, 2)
sns.histplot(X_train_num_minmax[col], kde=True)
plt.title("Après MinMaxScaler")
# Après standardisation
plt.subplot(1, 3, 3)
sns.histplot(X_train_num_std[col], kde=True)
plt.title("Après StandardScaler")
plt.tight_layout()
plt.show()
# Gestion du deséquilibre des classes
print("deséquilibre des classes:")
print(y_train.value_counts())
# Créer le sous-échantillonneur
from imblearn.under_sampling import RandomUnderSampler
under = RandomUnderSampler(random_state=42)
# Appliquer le sous-échantillonnage
X_train_minmax = pd.concat([X_train_num_minmax, X_train_cat], axis=1)
X_train_under, y_train_under = under.fit_resample(X_train_minmax,y_train)
# Afficher la nouvelle répartition
print("Aprèssous-échantillonnage :")
print(y_train_under.value_counts())
# Créer le sur-échantillonneur
from imblearn.over_sampling import SMOTE
# Créer SMOTE
smote = SMOTE(random_state=42)
# Appliquer SMOTE
X_train_smote, y_train_smote = smote.fit_resample(X_train_minmax,y_train)
# Afficher la nouvelle répartition
print("Après SMOTE :")
print(y_train_smote.value_counts())