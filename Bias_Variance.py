import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate data
np.random.seed(42)
x = np.random.rand(100, 1) * 10  # values between 0 and 10
y = x**2 + np.random.randn(100, 1) * 5  # y = x² + noise

# Sort x for smooth plotting
x_plot = np.linspace(0, 10, 100).reshape(-1, 1)

# Models to compare: 1 (underfit), 2 (ok), 15 (overfit)
degrees = [1, 2, 20]

plt.figure(figsize=(16, 4))

for i, degree in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x, y)
    y_pred = model.predict(x_plot)

    plt.subplot(1, 3, i+1)
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(x_plot, y_pred, color='red', label=f'Degree {degree}')
    plt.title(f'Degree {degree}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

plt.suptitle("Bias-Variance Project: Underfit vs Good Fit vs Overfit", fontsize=16)
plt.tight_layout()
plt.show()
