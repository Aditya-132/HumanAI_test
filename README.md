# Learning the Deterministic SIR Model from Stochastic Epidemics

## Overview

This project demonstrates how machine learning and symbolic regression can recover the deterministic **Susceptible–Infected–Removed (SIR)** model from stochastic epidemic simulations.

Pipeline:

1. Simulate many stochastic SIR epidemics using the Gillespie algorithm  
2. Compute the mean epidemic trajectory  
3. Train a neural network to learn the mean dynamics  
4. Use sparse symbolic regression (SINDy) to rediscover the governing ODEs  
5. Extract epidemiological parameters β and γ  

The recovered equations closely match the true deterministic SIR model.

---

## True Deterministic SIR Model

The classical SIR model is:

dS/dt = -β S I  
dI/dt = β S I - γ I  

Where:

- β = infection rate  
- γ = recovery rate  

In this experiment:

- β = 0.3  
- γ = 0.1  

---

## Results

### Neural Network Performance

ML Mean Squared Error:

ML MSE: 0.001893

This shows the neural network successfully learned the mean epidemic trajectory.

---

### Recovered Symbolic Model

Recovered equations:

S = -0.294 S I  
I = -0.098 I + 0.289 S I  

Recovered parameters:

Estimated beta  = 0.2937  
Estimated gamma = 0.0977  

Parameter comparison:

| Parameter | True | Recovered | Error |
|-----------|------|-----------|-------|
| β         | 0.300 | 0.2937 | ~2% |
| γ         | 0.100 | 0.0977 | ~2% |

The deterministic SIR model is successfully recovered from purely stochastic simulations.

---

## Generated Visualizations

The script produces:

- Sample stochastic infected trajectories  
- Mean epidemic curves (S and I)  
- Neural network prediction vs true mean  
- Symbolic model simulation vs true mean  
- Sparse regression coefficient visualization  

These confirm:

- Convergence of stochastic simulations to deterministic behavior  
- Accurate ML learning  
- Sparse recovery of governing equations  

---

## Methodology

### 1. Stochastic Simulation

- Gillespie algorithm  
- 300–500 independent epidemic realizations  
- Interpolation onto a common time grid  
- Averaging to compute mean dynamics  

### 2. Machine Learning

- MLPRegressor (2 hidden layers)  
- Learns mapping:  
  (S(t), I(t)) → (S(t+1), I(t+1))  

### 3. Symbolic Regression

- Derivatives computed via Savitzky–Golay smoothing  
- Polynomial feature library (degree 2)  
- Sparse regression (STLSQ thresholding)  
- Enforced sparsity to obtain interpretable equations  

---

## Dependencies

Install required packages:

```bash
pip install numpy matplotlib scikit-learn scipy pysindy
