import pandas as pd
import matplotlib.pyplot as plt

# Φόρτωση αρχικών δεδομένων
df = pd.read_csv('data/drivers_data.csv')

# We select a feature that exhibits extreme values.
feature = 'Engine_speed'

# Calculation of IQR limits
Q1 = df[feature].quantile(0.25)
Q3 = df[feature].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Creation of the clipped version
df_clipped = df.copy()
df_clipped[feature] = df_clipped[feature].clip(lower, upper)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before the clipping
axes[0].boxplot(df[feature].dropna())
axes[0].set_title('Πριν το IQR clipping')
axes[0].set_ylabel(feature)

# After the clipping
axes[1].boxplot(df_clipped[feature].dropna())
axes[1].set_title('Μετά το IQR clipping')
axes[1].set_ylabel(feature)

plt.suptitle(
    f'Επίδραση του IQR clipping στο χαρακτηριστικό {feature}',
    fontsize=14
)

plt.tight_layout()
plt.show()