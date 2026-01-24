# Import required libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# ------------------ Load Dataset ------------------
iris = load_iris()
X = iris.data
y = iris.target

# ------------------ Add Noise to Make Problem Harder ------------------
np.random.seed(0)
noise = np.random.normal(0, 0.5, X.shape)   # add noise
X = X + noise

# ------------------ Train-Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# ------------------ Classifiers ------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=3),
    "SVM": SVC(kernel='rbf', C=1)
}

# ------------------ Train, Predict & Evaluate ------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n------------------------------")
    print("Model:", name)
    print("Accuracy :", round(accuracy_score(y_test, y_pred)*100, 2), "%")
    print("Precision:", round(precision_score(y_test, y_pred, average='macro'), 2))
    print("Recall   :", round(recall_score(y_test, y_pred, average='macro'), 2))
    print("F1-Score :", round(f1_score(y_test, y_pred, average='macro'), 2))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
