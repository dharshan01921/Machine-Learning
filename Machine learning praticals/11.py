# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------ Sample Dataset ------------------
# Features: Income, Age, LoanAmount, CreditHistory
# Target: CreditScore (Good/Bad)

data = {
    'Income': [50000, 30000, 60000, 35000, 70000, 40000, 80000, 32000, 45000, 65000],
    'Age': [25, 40, 30, 50, 28, 45, 35, 55, 33, 38],
    'LoanAmount': [20000, 15000, 25000, 30000, 10000, 22000, 12000, 28000, 18000, 24000],
    'CreditHistory': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    'CreditScore': ['Good', 'Bad', 'Good', 'Bad', 'Good', 'Bad', 'Good', 'Bad', 'Good', 'Good']
}

# Create DataFrame
df = pd.DataFrame(data)

# Encode target labels: Good = 1, Bad = 0
df['CreditScore'] = df['CreditScore'].map({'Bad': 0, 'Good': 1})

# ------------------ Features & Target ------------------
X = df[['Income', 'Age', 'LoanAmount', 'CreditHistory']]
y = df['CreditScore']

# ------------------ Split Data ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ------------------ Train Logistic Regression Model ------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ------------------ Predict ------------------
y_pred = model.predict(X_test)

# ------------------ Evaluation ------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ------------------ Output ------------------
print("Confusion Matrix:")
print(cm)
print("\nAccuracy:", accuracy * 100, "%")
