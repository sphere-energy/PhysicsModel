import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the Equivalent Circuit Model (ECM)
def ecm_model(f, Rs, Rct, Cdl, W, tau_W):
    omega = 2 * np.pi * f  # Angular frequency
    Z_dl = Rct / (1 + 1j * omega * Rct * Cdl)  # Impedance of the RC part (charge transfer)
    Z_W = W / (1 + 1j * omega * tau_W)  # Warburg element for diffusion
    Z_total = Rs + Z_dl + Z_W
    return Z_total

# Example EIS data (frequency, Z_real, Z_imag)
frequencies = np.logspace(1, 5, 100)  # Frequency range from 10Hz to 100kHz
Z_real_exp = np.real(ecm_model(frequencies, 0.1, 5, 1e-5, 0.5, 0.01))  # Simulated real part
Z_imag_exp = np.imag(ecm_model(frequencies, 0.1, 5, 1e-5, 0.5, 0.01))  # Simulated imaginary part

# Combine real and imaginary parts into a complex impedance vector
Z_exp = Z_real_exp + 1j * Z_imag_exp

# Define the fitting function
def fit_function(f, Rs, Rct, Cdl, W, tau_W):
    omega = 2 * np.pi * f
    Z_dl = Rct / (1 + 1j * omega * Rct * Cdl)
    Z_W = W / (1 + 1j * omega * tau_W)
    Z_total = Rs + Z_dl + Z_W
    return np.real(Z_total) + 1j * np.imag(Z_total)

# Initial guess for parameters [Rs, Rct, Cdl, W, tau_W]
initial_guess = [0.1, 5, 1e-5, 0.5, 0.01]

# Perform the curve fitting
params_opt, params_cov = curve_fit(lambda f, Rs, Rct, Cdl, W, tau_W: np.real(fit_function(f, Rs, Rct, Cdl, W, tau_W)),
                                   frequencies, Z_real_exp, p0=initial_guess)

# Extract the fitted parameters
Rs_fit, Rct_fit, Cdl_fit, W_fit, tau_W_fit = params_opt
print("Fitted parameters:")
print(f"Rs = {Rs_fit:.3f} Ohms")
print(f"Rct = {Rct_fit:.3f} Ohms")
print(f"Cdl = {Cdl_fit:.3e} F")
print(f"W = {W_fit:.3f} Ohm·s^0.5")
print(f"tau_W = {tau_W_fit:.3f} s")

# Plot the experimental and fitted data
Z_fit = fit_function(frequencies, *params_opt)
plt.figure(figsize=(8, 5))
plt.plot(frequencies, Z_real_exp, label="Experimental Real Part", linestyle='--', color='r')
plt.plot(frequencies, np.real(Z_fit), label="Fitted Real Part", color='b')
plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Impedance (Ohms)')
plt.title('ECM Fitting: Real Part of Impedance')
plt.legend()
plt.grid(True)
plt.show()

# Plot imaginary parts
plt.figure(figsize=(8, 5))
plt.plot(frequencies, Z_imag_exp, label="Experimental Imaginary Part", linestyle='--', color='r')
plt.plot(frequencies, np.imag(Z_fit), label="Fitted Imaginary Part", color='b')
plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Impedance (Ohms)')
plt.title('ECM Fitting: Imaginary Part of Impedance')
plt.legend()
plt.grid(True)
plt.show()
