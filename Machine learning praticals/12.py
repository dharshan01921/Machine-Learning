# Import required libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------ Load Iris Dataset ------------------
iris = load_iris()
X = iris.data      # Features: Sepal length, Sepal width, Petal length, Petal width
y = iris.target    # Target: Species (0=setosa, 1=versicolor, 2=virginica)

# ------------------ Split Data into Training and Test Sets ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ------------------ Create KNN Classifier ------------------
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# ------------------ Train the Model ------------------
knn.fit(X_train, y_train)

# ------------------ Predict on Test Set ------------------
y_pred = knn.predict(X_test)

# ------------------ Evaluate the Model ------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ------------------ Display Results ------------------
print("Confusion Matrix:")
print(cm)
print("\nAccuracy:", accuracy * 100, "%")
