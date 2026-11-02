import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
print("=== Loading Data ===")
df = pd.read_csv("data/cleaned_data.csv")
print(f"Dataset Shape: {df.shape}")

# 2. Chack NaN 
print("\n=== Checking for NaN Values ===")
nan_values = df.isnull().sum()
if nan_values.sum() == 0:
    print("Δεν βρέθηκαν κενές τιμές (No NaN values found).")
else:
    print(nan_values[nan_values > 0])

# 3. Check Outliers 
# Use boxplots 
plt.figure(figsize=(15, 8))
cols_to_plot = df.drop(columns=["Class", "PathOrder"]).columns[:10] 
df[cols_to_plot].boxplot()
plt.title("Boxplot for Outlier Detection (First 10 features)")
plt.xticks(rotation=45)
plt.show()

# 4. Class Balance
print("\n=== Class Distribution ===")
print(df["Class"].value_counts(normalize=True)) # Presentage for each driver
sns.countplot(x='Class', data=df)
plt.title("Distribution of Driver Classes")
plt.show()

# 5. Correlation Analysis 
print("\n=== Computing Correlation Matrix ===")
plt.figure(figsize=(12, 10))
features_only = df.drop(columns=["Class", "PathOrder"])
corr_matrix = features_only.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# 6. Feature Selection with Mutual Information
print("\n=== Calculating Mutual Information Score ===")
X = features_only
y = df["Class"]

# Encoding the target for Mutual Info
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Calculate MI
mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
mi_results = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

print("\nTop 10 Σημαντικότερα Κανάλια (βάσει Mutual Information):")
print(mi_results.head(10))

# Graphs
plt.figure(figsize=(10, 6))
mi_results.head(15).plot(kind='barh')
plt.title("Top 15 Features by Mutual Information")
plt.xlabel("Mutual Information Score")
plt.gca().invert_yaxis()
plt.show()

