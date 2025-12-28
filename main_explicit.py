import numpy as np
import matplotlib.pyplot as plt
from models import build_autoencoder, MSEHistory
from utils import estimate_polynomial

# Generate Data: y = x^4 + x^2
x = np.linspace(-10, 10, 200).reshape(-1, 1)
y = x**4 + x**2
y_norm = (y - y.mean()) / y.std()

# Train
model = build_autoencoder(input_dim=1)
history = MSEHistory()
model.fit(x, y_norm, epochs=600, verbose=0, callbacks=[history])

# Results
y_pred = model.predict(x) * y.std() + y.mean()
deg, coefs, formula, _ = estimate_polynomial(x, y_pred)

print(f"Detected Degree: {deg}")
print(f"Formula: {formula}")

plt.scatter(x, y, label="Original")
plt.plot(x, y_pred, color='red', label="Encoder-X Fit")
plt.legend()
plt.show()
