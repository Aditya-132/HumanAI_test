# =====================================
# 1. Imports
# =====================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.integrate import solve_ivp
import pysindy as ps


# =====================================
# 2. Stochastic SIR (Gillespie)
# =====================================
def gillespie_sir(beta, gamma, S0, I0, R0, N, t_max):
    S, I, R = S0, I0, R0
    t = 0
    times = [t]
    states = [[S, I, R]]

    while t < t_max and I > 0:
        infection_rate = beta * S * I / N
        recovery_rate = gamma * I
        total_rate = infection_rate + recovery_rate

        if total_rate == 0:
            break

        dt = np.random.exponential(1 / total_rate)
        t += dt

        if np.random.rand() < infection_rate / total_rate:
            S -= 1
            I += 1
        else:
            I -= 1
            R += 1

        times.append(t)
        states.append([S, I, R])

    return np.array(times), np.array(states)


# =====================================
# 3. Simulation Parameters
# =====================================
beta = 0.3
gamma = 0.1
N = 1000

S0, I0, R0 = 990, 10, 0
t_max = 60
num_simulations = 300

time_grid = np.linspace(0, t_max, 600)
all_trajectories = []


# =====================================
# 4. Run Simulations
# =====================================
for sim in range(num_simulations):
    t_sim, states_sim = gillespie_sir(beta, gamma, S0, I0, R0, N, t_max)

    interp_S = interp1d(t_sim, states_sim[:,0], fill_value="extrapolate")
    interp_I = interp1d(t_sim, states_sim[:,1], fill_value="extrapolate")
    interp_R = interp1d(t_sim, states_sim[:,2], fill_value="extrapolate")

    S_interp = interp_S(time_grid)
    I_interp = interp_I(time_grid)
    R_interp = interp_R(time_grid)

    all_trajectories.append(np.column_stack([S_interp, I_interp, R_interp]))

all_trajectories = np.array(all_trajectories)
mean_trajectory = np.mean(all_trajectories, axis=0)

S_mean = mean_trajectory[:,0] / N
I_mean = mean_trajectory[:,1] / N


# =====================================
# GRAPH 1: Sample Stochastic Trajectories
# =====================================
plt.figure()
for i in range(10):
    plt.plot(time_grid, all_trajectories[i,:,1] / N)
plt.title("Sample Stochastic Infected Trajectories")
plt.xlabel("Time")
plt.ylabel("Infected Fraction")
plt.show()


# =====================================
# GRAPH 2: Mean Epidemic Curve
# =====================================
plt.figure()
plt.plot(time_grid, S_mean, label="Mean S")
plt.plot(time_grid, I_mean, label="Mean I")
plt.title("Mean Stochastic SIR Dynamics")
plt.xlabel("Time")
plt.ylabel("Population Fraction")
plt.legend()
plt.show()


# =====================================
# 5. Train ML Model
# =====================================
X = np.column_stack([S_mean[:-1], I_mean[:-1]])
y = np.column_stack([S_mean[1:], I_mean[1:]])

ml_model = MLPRegressor(hidden_layer_sizes=(32,32), max_iter=5000)
ml_model.fit(X, y)

ml_pred = ml_model.predict(X)
ml_I_pred = ml_pred[:,1]

print("ML MSE:", mean_squared_error(y, ml_pred))


# =====================================
# GRAPH 3: ML Prediction vs True
# =====================================
plt.figure()
plt.plot(time_grid[:-1], I_mean[:-1], label="True Mean I")
plt.plot(time_grid[:-1], ml_I_pred, "--", label="ML Predicted I")
plt.title("ML Prediction vs True Mean")
plt.xlabel("Time")
plt.ylabel("Infected Fraction")
plt.legend()
plt.show()


# =====================================
# 6. Symbolic Regression (SINDy)
# =====================================
dt = time_grid[1] - time_grid[0]

dSdt = savgol_filter(S_mean, 31, 3, deriv=1, delta=dt)
dIdt = savgol_filter(I_mean, 31, 3, deriv=1, delta=dt)

states = np.column_stack([S_mean, I_mean])
derivatives = np.column_stack([dSdt, dIdt])

poly_library = ps.PolynomialLibrary(degree=2, include_bias=False)
optimizer = ps.STLSQ(threshold=0.03)

sindy_model = ps.SINDy(feature_library=poly_library,
                        optimizer=optimizer)

sindy_model.fit(states, t=dt, x_dot=derivatives)

print("\nRecovered Deterministic Model:")
sindy_model.print(lhs=["S", "I"])


coefficients = sindy_model.coefficients()
feature_names = poly_library.get_feature_names(["S", "I"])

si_index = feature_names.index("S I")
i_index = feature_names.index("I")

beta_est = -coefficients[0, si_index]
gamma_est = -coefficients[1, i_index]

print("Estimated beta:", beta_est)
print("Estimated gamma:", gamma_est)


# =====================================
# Simulate Learned Model
# =====================================
def learned_sir(t, y):
    S, I = y
    dS = -beta_est * S * I
    dI = beta_est * S * I - gamma_est * I
    return [dS, dI]

sol = solve_ivp(learned_sir,
                [0, t_max],
                [S_mean[0], I_mean[0]],
                t_eval=time_grid)

S_learned = sol.y[0]
I_learned = sol.y[1]


# =====================================
# GRAPH 4: Symbolic Model vs True
# =====================================
plt.figure()
plt.plot(time_grid, I_mean, label="True Mean I")
plt.plot(time_grid, I_learned, "--", label="Symbolic Model I")
plt.title("Symbolic Model vs True Dynamics")
plt.xlabel("Time")
plt.ylabel("Infected Fraction")
plt.legend()
plt.show()


# =====================================
# GRAPH 5: SINDy Coefficient Magnitudes
# =====================================
plt.figure()
plt.bar(feature_names, coefficients[0])
plt.xticks(rotation=45)
plt.title("Recovered Coefficients for dS/dt")
plt.show()