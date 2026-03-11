# =====================================
# AQI Prediction ML Project
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# -------------------------------------
# 1. Load Dataset
# -------------------------------------

# Dataset should be placed in project folder
DATA_PATH = "data/city_day.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded")
print(df.head())

# -------------------------------------
# 2. Data Preprocessing
# -------------------------------------

df['Date'] = pd.to_datetime(df['Date'])

# Fill missing numeric values
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# -------------------------------------
# 3. Feature Engineering
# -------------------------------------

df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['year'] = df['Date'].dt.year
df['day_of_week'] = df['Date'].dt.dayofweek

# Convert City to dummy variables
df = pd.get_dummies(df, columns=['City'], drop_first=True)

# -------------------------------------
# 4. Exploratory Data Analysis
# -------------------------------------

plt.figure()
sns.histplot(df['AQI'], bins=50)
plt.title("AQI Distribution")
plt.savefig("aqi_distribution.png")

plt.figure(figsize=(10,6))
sns.heatmap(df[['AQI','PM2.5','PM10','NO2','SO2','CO','O3']].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.savefig("correlation_heatmap.png")

# -------------------------------------
# 5. Prepare ML Data
# -------------------------------------

features = ['PM2.5','PM10','NO2','SO2','CO','O3','month','day_of_week']

X = df[features]
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------
# 6. Train Models
# -------------------------------------

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

pred_lr = lr.predict(X_test)
pred_rf = rf.predict(X_test)
pred_gb = gb.predict(X_test)

# -------------------------------------
# 7. Evaluation
# -------------------------------------

def evaluate(y_true, pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    mae = mean_absolute_error(y_true, pred)
    r2 = r2_score(y_true, pred)

    print("\n", name)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2 Score:", r2)

evaluate(y_test, pred_lr, "Linear Regression")
evaluate(y_test, pred_rf, "Random Forest")
evaluate(y_test, pred_gb, "Gradient Boosting")

print("\nAQI Prediction Project Completed")
