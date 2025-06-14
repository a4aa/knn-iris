from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
iris = load_iris()
X = iris.data          # Features: sepal length, sepal width, etc.
y = iris.target        # Labels: 0, 1, 2 (Setosa, Versicolor, Virginica)

acc_array = [] #Stores accuracy data
i_arr = range(1, 121) # stores k; which can be from 1-100
for k in i_arr:
    # 2. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Initialize the model
    model = KNeighborsClassifier(n_neighbors=k)

    # 4. Train the model
    model.fit(X_train, y_train)

    # 5. Predict on test set
    y_pred = model.predict(X_test)

    # 6. Check accuracy
    acc = accuracy_score(y_test, y_pred)
    acc_array.append(acc*100)
    print(f"Accuracy: {acc:.2f}")

    # 7. Predict a custom flower
    sample = [[5.1, 3.5, 1.4, 0.2]]  # [sepal length, sepal width, petal length, petal width]
    prediction = model.predict(sample)
    print(f"Predicted class: {iris.target_names[prediction[0]]}")

best_k = i_arr[np.argmax(acc_array)]
print(f"\nBest k = {best_k} with Accuracy = {max(acc_array):.2f}%")

final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = final_model.predict(sample)
print(f"Final Prediction: {iris.target_names[prediction[0]]}")

plt.plot(i_arr, acc_array)
plt.title("Correlation between k and accuracy")
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show() #plot graph of k vs acc
