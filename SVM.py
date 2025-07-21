from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

fence_types = ['linear', 'poly', 'rbf']
results = {}

for fence_type in fence_types:
    svm = SVC(kernel=fence_type, random_state=42)
    svm.fit(X_train_scaled, y_train)
    predictions = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    results[fence_type] = accuracy
    print(f"SVM with {fence_type} fence: {accuracy:.1%} accurate")

best_fence = max(results, key=results.get)
print(f"\nBest fence type: {best_fence}")
