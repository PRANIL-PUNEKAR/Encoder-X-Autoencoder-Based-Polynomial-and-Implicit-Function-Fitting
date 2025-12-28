import numpy as np
from sklearn.metrics import mean_squared_error
from numpy.polynomial import Polynomial

def estimate_polynomial(x, y, max_degree=15, mse_threshold=0.05):
    y, x = y.flatten(), x.flatten()
    best_poly = None
    mse_list = []

    for deg in range(1, max_degree + 1):
        p = Polynomial.fit(x, y, deg, domain=[x.min(), x.max()])
        mse = mean_squared_error(y, p(x))
        mse_list.append(mse)
        if mse <= mse_threshold:
            best_poly = p
            break
    
    if best_poly is None: best_poly = Polynomial.fit(x, y, max_degree)
    
    coefs = best_poly.convert().coef[::-1]
    poly_str = "y = " + " ".join([f"{round(c, 4)}x^{len(coefs)-1-i}" for i, c in enumerate(coefs) if abs(c) > 1e-5])
    return len(coefs)-1, coefs, poly_str, mse_list
