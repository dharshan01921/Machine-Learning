# Import required libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ------------------ Dataset ------------------
data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8],
    'Sales': [120, 130, 150, 165, 180, 195, 210, 230]
}

df = pd.DataFrame(data)

# ------------------ Prepare Data ------------------
X = df[['Month']]
y = df['Sales']

# ------------------ Train-Test Split ------------------
X_train = X.iloc[:6]
y_train = y.iloc[:6]

X_test = X.iloc[6:]
y_test = y.iloc[6:]

# ------------------ Train Model ------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------ Test Predictions ------------------
y_pred = model.predict(X_test)

# ------------------ Accuracy Evaluation ------------------
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Model Performance:")
print("R² Score (Accuracy-like):", round(r2 * 100, 2), "%")
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))

# ------------------ Future Sales Prediction ------------------
future_months = np.array([[9], [10], [11], [12]])
future_sales = model.predict(future_months)

print("\nFuture Sales Prediction:")
for month, sale in zip(future_months.flatten(), future_sales):
    print(f"Month {month}: Predicted Sales = {round(sale, 2)}")

# ------------------ Visualization ------------------
plt.scatter(X, y, label="Actual Sales")
plt.plot(future_months, future_sales, color='red', marker='o', label="Future Predictions")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Future Sales Prediction using Linear Regression")
plt.legend()
plt.show()
