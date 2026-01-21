# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------ Create Sample Dataset ------------------
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2017, 2018],
    'Present_Price': [5.5, 6.0, 7.0, 8.5, 9.0, 10.0, 6.8, 8.0],
    'Kms_Driven': [50000, 40000, 30000, 20000, 15000, 10000, 35000, 22000],
    'Fuel_Type': [0, 1, 0, 1, 1, 1, 0, 1],        # Petrol=0, Diesel=1
    'Seller_Type': [0, 1, 0, 1, 1, 1, 0, 1],     # Dealer=0, Individual=1
    'Transmission': [0, 1, 0, 1, 1, 1, 0, 1],    # Manual=0, Automatic=1
    'Selling_Price': [3.5, 4.0, 5.0, 6.0, 6.8, 7.5, 4.8, 6.2]
}

df = pd.DataFrame(data)

# ------------------ Features & Target ------------------
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# ------------------ Train-Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ------------------ Train Linear Regression Model ------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------ Prediction ------------------
y_pred = model.predict(X_test)

# ------------------ Evaluation ------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Predicted Prices:", y_pred)
print("Mean Squared Error:", mse)
print("R² Score:", r2)
