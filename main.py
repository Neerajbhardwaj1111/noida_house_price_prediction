import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import re


df = pd.read_csv("noida_flats1.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)


# Convert price to numeric values (handling 'Cr' and 'L')
def convert_price(price):
    price = price.lower().replace(" ", "")
    if "cr" in price:
        return float(price.replace("cr", "")) * 100  # Convert Cr to Lakh
    elif "l" in price:
        return float(price.replace("l", ""))
    else:
        return np.nan

df["Price"] = df["Price"].apply(convert_price)

# Drop any rows with missing values
df.dropna(inplace=True)

# Encode categorical variable (Location)
df = pd.get_dummies(df, columns=["Location"], drop_first=True)

# Display cleaned data
print(df.head())

X = df.drop(columns=["Price"])
y = df["Price"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)



# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluate Model
print("Linear Regression Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R² Score:", r2_score(y_test, y_pred_lr))



# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate Model
print("Random Forest Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R² Score:", r2_score(y_test, y_pred_rf))




# Train XGBoost Model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate Model
print("XGBoost Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_xgb))
print("MSE:", mean_squared_error(y_test, y_pred_xgb))
print("R² Score:", r2_score(y_test, y_pred_xgb))

house_features = {col: [0] for col in expected_columns}  # Initialize with zeros

# Set actual values for features
house_features["Area in Sqrt"] = [4200]  # Set Area
house_features["No.of Bhk"] = [5]        # Set BHK
house_features["Location_Sector 151, Noida"] = [1]  # Set selected location

# Convert to DataFrame with correct column order
new_house = pd.DataFrame(house_features)

# Ensure order matches the training dataset
new_house = new_house[expected_columns]

# Predict using the best model
predicted_price = lr_model.predict(new_house)
print("Predicted Price:", predicted_price[0], "Lakhs")
