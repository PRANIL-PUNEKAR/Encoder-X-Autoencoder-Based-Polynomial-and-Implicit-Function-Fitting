# Encoder-X: Autoencoder-Based Polynomial & Implicit Function Learning

Encoder-X is a deep-learning framework that automatically learns unknown functions,
denoises data, selects optimal polynomial degree, and extracts meaningful coefficients
using an autoencoder architecture.

## Features
- Explicit function learning: y = f(x)
- Implicit function learning: F(x, y) = 0
- Automatic polynomial degree selection using MSE convergence
- Shape reconstruction (circle, heart, ellipse, random curves)

## Project Structure
explicit/   → Explicit autoencoder + polynomial fitting  
implicit/   → Implicit autoencoder for shapes  
utils/      → Helper utilities (MSE callback, normalization)  
results/    → Generated plots  
report/     → Major project report  

## How It Works
1. Train autoencoder to reconstruct signal
2. Use denoised output for polynomial fitting
3. Automatically select degree using ΔMSE criterion
4. Extract coefficients or implicit contours

## Run Instructions
pip install -r requirements.txt

python explicit/run_explicit_demo.py  
python implicit/run_implicit_demo.py  

## Degree Selection Rule
ΔMSE < ε → Optimal degree selected

## Applications
- Signal denoising
- Scientific modeling
- Function discovery
- Shape reconstruction

## Authors

PRANIL PUNEKAR  
NITK Surathkal
