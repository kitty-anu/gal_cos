import numpy as np
import matplotlib.pyplot as plt

# Constants
dtau = 0.01  # Step size in Gigayears
Omega_m0 = 0.3
Omega_lambda0 = 0.7
Omega_k0 = 1 - Omega_m0 - Omega_lambda0
H_0 = 0.07  # in units of 1/Gigayear
c = 299792.458 * 1e-3  # Speed of light in Mpc/Gyr

# Number of steps and maximum lookback time (Gigayears)
tau_max = 10  # Maximum lookback time in Gigayears
num_steps = int(tau_max / dtau)

# Differential equation for da/dtau
def da_dtau(a):
    return -H_0 * np.sqrt(Omega_m0 / a + Omega_k0 + Omega_lambda0 * a**2)

# Hubble parameter as a function of a
def H_a(a):
    return H_0 * np.sqrt(Omega_m0 / a**3 + Omega_k0 / a**2 + Omega_lambda0)

# Initialize arrays
tau = np.linspace(0, tau_max, num_steps)
a = np.zeros(num_steps)
z = np.zeros(num_steps)
H_z = np.zeros(num_steps)
D_C = np.zeros(num_steps)
D_A = np.zeros(num_steps)
D_P = np.zeros(num_steps)
D_L = np.zeros(num_steps)

a[0] = 1  # Initial condition: a = 1 at tau = 0 (today)

# Euler's method to solve da/dtau and coming distance integral for distances
for i in range(1, num_steps):
    a[i] = a[i-1] + dtau * da_dtau(a[i-1])
    z[i] = 1 / a[i] - 1
    H_z[i] = H_a(a[i])
    D_C[i] = D_C[i-1] + (c / H_a(a[i-1])) * dtau  # Comoving distance integral
    D_P[i] = D_C[i] # Proper distance
    D_A[i] = D_C[i] / (1 + z[i])  # Angular diameter distance
    D_L[i] = D_C[i] * (1 + z[i])  # Luminosity distance

# Plot for scale factor a(tau)
plt.plot(tau, a)
plt.xlabel('Lookback Time τ (Gigayears)')
plt.ylabel('Scale Factor a(τ)')
plt.title('Scale Factor a(τ)')
plt.grid(True)
plt.savefig('plot1.png')

# Plot for redshift z(tau)
plt.clf()
plt.plot(tau, z)
plt.xlabel('Lookback Time τ (Gigayears)')
plt.ylabel('Redshift z')
plt.title('Redshift z(τ)')
plt.grid(True)
plt.savefig('plot2.png')

# Plot for Hubble parameter H(z)
plt.clf()
plt.plot(tau, H_z)
plt.xlabel('Lookback Time τ (Gigayears)')
plt.ylabel('Hubble Parameter H(z) (Gyr⁻¹)')
plt.title('Hubble Parameter H(z)')
plt.grid(True)
plt.savefig('plot3.png')

# Plot for angular diameter distance D_A
plt.clf()
plt.plot(tau, D_A / 1000)  # Convert to Mpc
plt.xlabel('Lookback Time τ (Gigayears)')
plt.ylabel('Angular Diameter Distance D_A (Mpc)')
plt.title('Angular Diameter Distance D_A(τ)')
plt.grid(True)
plt.savefig('plot4.png')

# Plot for comoving distance D_P
plt.clf()
plt.plot(tau, D_P / 1000)  # Convert to Mpc
plt.xlabel('Lookback Time τ (Gigayears)')
plt.ylabel('Comoving Distance D_C (Mpc)')
plt.title('Comoving Distance D_C(τ)')
plt.grid(True)
plt.savefig('plot5.png')

# Plot for luminosity distance D_L
plt.clf()
plt.plot(tau, D_L / 1000)  # Convert to Mpc
plt.xlabel('Lookback Time τ (Gigayears)')
plt.ylabel('Luminosity Distance D_L (Mpc)')
plt.title('Luminosity Distance D_L(τ)')
plt.grid(True)
plt.savefig('plot6.png')