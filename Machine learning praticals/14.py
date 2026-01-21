# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------ Create Sample House Dataset ------------------
data = {
    'Area_sqft': [800, 900, 1000, 1100, 1200, 1300, 1400, 1500],
    'Bedrooms': [2, 2, 3, 3, 3, 4, 4, 4],
    'Age_years': [15, 12, 10, 8, 6, 5, 4, 3],
    'Price_lakhs': [30, 35, 45, 50, 55, 65, 70, 75]
}

df = pd.DataFrame(data)

# ------------------ Features & Target ------------------
X = df[['Area_sqft', 'Bedrooms', 'Age_years']]
y = df['Price_lakhs']

# ------------------ Train-Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ------------------ Train Linear Regression Model ------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------ Prediction ------------------
y_pred = model.predict(X_test)

# ------------------ Evaluation ------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Predicted House Prices:", y_pred)
print("Mean Squared Error:", mse)
print("R² Score:", r2)
