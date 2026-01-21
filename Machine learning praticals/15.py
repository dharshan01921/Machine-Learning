# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------ Load Iris Dataset ------------------
iris = load_iris()
X = iris.data      # Features
y = iris.target    # Labels

# ------------------ Train-Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ------------------ Create Naïve Bayes Classifier ------------------
model = GaussianNB()

# ------------------ Train the Model ------------------
model.fit(X_train, y_train)

# ------------------ Prediction ------------------
y_pred = model.predict(X_test)

# ------------------ Evaluation ------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ------------------ Display Results ------------------
print("Confusion Matrix:")
print(cm)
print("\nAccuracy:", accuracy * 100, "%")
