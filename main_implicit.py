import numpy as np
import matplotlib.pyplot as plt
from models import build_autoencoder

# Generate Heart Shape Data
x = np.random.uniform(-2, 2, 1000)
y = np.random.uniform(-2, 2, 1000)
F = (x**2 + y**2 - 1)**3 - x**2 * y**3 # Heart equation

# Train
model = build_autoencoder(input_dim=2)
model.fit(np.c_[x, y], F, epochs=500, verbose=0)

# Visualize Contour
xx, yy = np.meshgrid(np.linspace(-2,2,100), np.linspace(-2,2,100))
grid = np.c_[xx.ravel(), yy.ravel()]
zz = model.predict(grid).reshape(xx.shape)

plt.contour(xx, yy, zz, levels=[0], colors='red')
plt.title("Implicit Function Reconstruction")
plt.show()
