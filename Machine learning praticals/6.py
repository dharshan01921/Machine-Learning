# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
data = load_iris()
X = data.data        # Features
y = data.target      # Labels

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# Create Naive Bayes classifier
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)
# Predict on test data
y_pred = model.predict(X_test)
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display results
print("Confusion Matrix:")
print(cm)

print("\nAccuracy:")
print(accuracy * 100, "%")
