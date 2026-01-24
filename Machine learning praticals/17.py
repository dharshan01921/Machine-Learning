import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------ Dataset ------------------
data = {
    'Battery_Power': [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1800, 2000, 2200],
    'RAM_MB': [512, 1024, 2048, 2048, 3072, 4096, 4096, 6144, 6144, 8192, 8192, 12288],
    'Internal_Memory_GB': [8, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512],
    'Camera_MP': [5, 8, 12, 13, 16, 20, 24, 32, 40, 48, 64, 108],
    'Price_Range': [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3]
}

df = pd.DataFrame(data)

# ------------------ Features & Target ------------------
X = df.drop('Price_Range', axis=1)
y = df['Price_Range']

# ------------------ Train-Test Split (More realistic) ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=7
)

# ------------------ Feature Scaling ------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ Random Forest (Controlled) ------------------
model = RandomForestClassifier(
    n_estimators=40,      # fewer trees
    max_depth=4,          # limit depth
    min_samples_split=3,
    random_state=7
)

model.fit(X_train, y_train)

# ------------------ Predictions ------------------
y_pred = model.predict(X_test)

# ------------------ Accuracy ------------------
accuracy = accuracy_score(y_test, y_pred)

price_map = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}

print("📊 Predictions on Previous Data:\n")
for i in range(len(X_test)):
    print(f"Mobile {i+1} → Predicted Price:", price_map[y_pred[i]])

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# ------------------ Future Mobile Prediction ------------------
future_mobiles = pd.DataFrame({
    'Battery_Power': [1700, 1300, 950],
    'RAM_MB': [8192, 4096, 2048],
    'Internal_Memory_GB': [256, 128, 32],
    'Camera_MP': [64, 20, 12]
})

future_scaled = scaler.transform(future_mobiles)
future_pred = model.predict(future_scaled)

print("\n🔮 Future Mobile Predictions:\n")
for i in range(len(future_mobiles)):
    print(f"Future Mobile {i+1} → Predicted Price:", price_map[future_pred[i]])
