# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------ Dataset ------------------
data = {
    'Age': [25, 30, 35, 40, 45, 50, 28, 32, 38, 42, 48, 55],
    'Income': [30000, 40000, 50000, 60000, 75000, 90000,
               35000, 45000, 58000, 65000, 80000, 100000],
    'Credit_Score': [600, 650, 700, 720, 750, 780,
                     620, 680, 710, 730, 760, 800],
    'Employment_Years': [1, 3, 5, 7, 10, 15, 2, 4, 6, 8, 12, 20],
    'Loan_Status': [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# ------------------ Features & Target ------------------
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# ------------------ Train-Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ------------------ Feature Scaling ------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ Naive Bayes Model ------------------
model = GaussianNB()
model.fit(X_train, y_train)

# ------------------ Predictions ------------------
y_pred = model.predict(X_test)

# ------------------ Evaluation ------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# ------------------ Prediction Labels ------------------
loan_map = {0: "Loan Rejected", 1: "Loan Approved"}

print("\nPredictions on Test Data:")
for i in range(len(y_test)):
    print(f"Customer {i+1} → Actual: {loan_map[y_test.iloc[i]]}, "
          f"Predicted: {loan_map[y_pred[i]]}")

# ------------------ Future Customer Prediction ------------------
future_customers = pd.DataFrame({
    'Age': [29, 47],
    'Income': [42000, 85000],
    'Credit_Score': [670, 770],
    'Employment_Years': [3, 11]
})

future_scaled = scaler.transform(future_customers)
future_predictions = model.predict(future_scaled)

print("\n🔮 Future Loan Predictions:")
for i in range(len(future_predictions)):
    print(f"Customer {i+1} → {loan_map[future_predictions[i]]}")
