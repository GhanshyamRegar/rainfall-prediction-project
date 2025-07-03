#  Rainfall Prediction Project: Austin Weather Dataset

#  Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Visual settings
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

#  Load Dataset
df = pd.read_csv("austin_weather.csv")
df.head()
#  Data Cleaning

# Check for missing values and datatypes
df.info()

# Replace trace values 'T' with 0.0 (very small values of rainfall)
df.replace('T', 0.0, inplace=True)

# Replace '-' (missing data) with NaN
df.replace('-', np.nan, inplace=True)

# Convert numeric columns to float
for col in df.columns:
    try:
        df[col] = df[col].astype(float)
    except:
        pass  # Ignore non-numeric columns

# Drop irrelevant columns (like Date or ones with too many missing values)
df = df.drop(columns=["Events", "Date"], errors="ignore")

# Drop rows with any NaN values
df.dropna(inplace=True)

# Final structure
df.info()
#  Correlation Analysis
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
#  Feature Selection and Target Variable
X = df.drop("PrecipitationSumInches", axis=1)
y = df["PrecipitationSumInches"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#  Apply Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

#  Visualization - Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Precipitation")
plt.ylabel("Predicted Precipitation")
plt.title("Actual vs Predicted Precipitation")
plt.plot([0, max(y_test)], [0, max(y_test)], color='red')
plt.show()

#  Trend Visualization - Precipitation Over Time (Optional if Date exists)
# If 'Date' column is present in raw dataset before dropping
df_date = pd.read_csv("austin_weather.csv")
df_date["PrecipitationSumInches"] = df_date["PrecipitationSumInches"].replace("T", 0.0).astype(float)
df_date["Date"] = pd.to_datetime(df_date["Date"])
df_date = df_date.dropna(subset=["PrecipitationSumInches"])

plt.plot(df_date["Date"], df_date["PrecipitationSumInches"])
plt.xlabel("Date")
plt.ylabel("Precipitation (inches)")
plt.title("Precipitation Trend Over Time")
plt.show()
