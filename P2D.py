import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
R = 8.314  # Universal gas constant (J/mol-K)
F = 96485  # Faraday's constant (C/mol)
T_ref = 298  # Reference temperature (K)

# Battery parameters
C_max = 3.2  # Initial capacity in Ah
k_SEI = 1e-4  # SEI growth rate (Ah per cycle^0.5)
k_plating = 1e-2  # Lithium plating rate (Ah per cycle)
R_SEI_0 = 0.01  # Initial SEI resistance (Ohms)
k_R = 1e-4  # SEI resistance growth rate (Ohms per cycle)
n_cycles = 500  # Total number of cycles

# Degradation model including SEI + lithium plating
def degradation_model(y, t, k_SEI, k_plating, k_R, T, C_rate):
    C, R_SEI = y
    # SEI-driven capacity fade with temperature effect (growth rate decreases with time)
    dC_SEI_dt = -k_SEI * C * (t ** 0.5) * C_rate  # SEI growth-driven capacity loss, scaled by C-rate
    # Lithium plating with temperature dependency (Arrhenius-like effect)
    dC_plating_dt = -k_plating * np.exp(-T / T_ref) * C * C_rate  # Temperature-dependent plating loss, scaled by C-rate
    # Total capacity loss (sum of both SEI and plating effects)
    dC_dt = dC_SEI_dt + dC_plating_dt
    # SEI resistance growth (constant rate)
    dR_SEI_dt = k_R  # SEI resistance grows at a constant rate per cycle
    return [dC_dt, dR_SEI_dt]

# Time array (1 step per cycle)
t = np.linspace(1, n_cycles, n_cycles)

y0 = [C_max, R_SEI_0] # Initial conditions

temperatures = [298]

C_rates = [0.5, 1, 2] 

plt.figure(figsize=(10, 5))

# Loop over different temperatures and simulate degradation for each C-rate
for C_rate in C_rates:
    for T in temperatures:
        # Solve ODEs for the given temperature and C-rate
        solution = odeint(degradation_model, y0, t, args=(k_SEI, k_plating, k_R, T, C_rate))
        
        capacity_evolution = solution[:, 0]
        
        plt.plot(t, capacity_evolution, label=f"T = {T} K, C-rate = {C_rate}C")

# Customize plot
plt.xlabel("Cycle Number")
plt.ylabel("Capacity (Ah)")
plt.title("Battery Degradation (SEI + Li-Plating) at different C-rates")
plt.legend()
plt.xlim(0, 500)
plt.ylim(0, C_max) 
plt.grid(True)
plt.show()