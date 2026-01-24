# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------ Load Iris Dataset ------------------
iris = load_iris()
X = iris.data
y = iris.target

# ------------------ Train-Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ------------------ Feature Scaling ------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ Train Perceptron Model ------------------
model = Perceptron(
    max_iter=1000,
    eta0=0.01,
    random_state=42
)
model.fit(X_train, y_train)

# ------------------ Prediction ------------------
y_pred = model.predict(X_test)

# ------------------ Evaluation ------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# ------------------ Display Sample Predictions ------------------
print("\nSample Predictions:")
for i in range(5):
    print("Actual:", iris.target_names[y_test[i]],
          "Predicted:", iris.target_names[y_pred[i]])
