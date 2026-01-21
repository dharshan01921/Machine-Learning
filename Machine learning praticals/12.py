# ------------------ Import Libraries ------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------ Updated Dataset (20 samples) ------------------
data = {
    'Income': [50000, 30000, 60000, 35000, 70000, 40000, 80000, 32000, 45000, 65000,
               48000, 52000, 36000, 61000, 33000, 75000, 39000, 58000, 42000, 68000],
    'Age': [25, 40, 30, 50, 28, 45, 35, 55, 33, 38,
            29, 42, 48, 31, 53, 36, 44, 32, 41, 37],
    'LoanAmount': [20000, 15000, 25000, 30000, 10000, 22000, 12000, 28000, 18000, 24000,
                   19000, 21000, 26000, 23000, 27000, 15000, 20000, 25000, 22000, 14000],
    'CreditHistory': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1,
                      1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    'CreditScore': ['Good', 'Bad', 'Good', 'Bad', 'Good', 'Bad', 'Good', 'Bad', 'Good', 'Good',
                    'Good', 'Good', 'Bad', 'Good', 'Bad', 'Good', 'Good', 'Good', 'Bad', 'Good']
}

# ------------------ Create DataFrame ------------------
df = pd.DataFrame(data)

# Encode target labels: Good = 1, Bad = 0
df['CreditScore'] = df['CreditScore'].map({'Bad': 0, 'Good': 1})

# ------------------ Features & Target ------------------
X = df[['Income', 'Age', 'LoanAmount', 'CreditHistory']]
y = df['CreditScore']

# ------------------ Split Data ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=5
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
