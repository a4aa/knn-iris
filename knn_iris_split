# ---------------------- Imports ----------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- Load Dataset ----------------------
iris = load_iris()
X = iris.data       # Features
y = iris.target     # Labels

# ---------------------- Required variables for plotting ----------------------
split_values = np.linspace(0.01, 0.99, 99)
# print(split_values) Uesd for debugging; not much more than that.
acc_results = []

for i in split_values:
    # ---------------------- Train-Test Split ----------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=42)

    # ---------------------- KNN Model ----------------------
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, y_train)

    # ---------------------- Predictions ----------------------
    y_pred = model.predict(X_test)

    # ---------------------- Evaluation ----------------------
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy (k=1): {accuracy:.2f}")
    acc_results.append(accuracy*100)

    # ---------------------- Sample Prediction ----------------------
    sample = [[5.1, 3.5, 1.4, 0.2]]  # Setosa-like sample
    predicted_class = model.predict(sample)
    print(f"Predicted class: {iris.target_names[predicted_class[0]]}")

# ---------------------- Plotting ----------------------
plt.plot(split_values, acc_results)
plt.xlabel("Split Ratio")
plt.ylabel("Accuracy %")
plt.title("Split Ratio vs Accuracy")
plt.show()
