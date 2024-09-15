import numpy as np
import matplotlib.pyplot as plt

# Constants
dtau = 0.01  # Step size in Gigayears
Omega_m0 = 0.0
Omega_lambda0 = 0.0
Omega_k0 = 1 - Omega_m0 - Omega_lambda0
H_0 = 0.07  # in units of 1/Gigayear
c = 299792.458 * 1e-3  # Speed of light in Mpc/Gyr

# Number of steps and maximum lookback time (Gigayears)
tau_max = 14  # Maximum lookback time in Gigayears
num_steps = int(tau_max / dtau)

# Differential equation for da/dtau
def da_dtau(a):
    if Omega_k0 == 1:  # Empty Universe
        return -H_0 * np.sqrt(Omega_k0)
    elif Omega_m0 == 1: # Matter-Only Univerese
        return -H_0 * np.sqrt(Omega_m0 / a)
    elif Omega_lambda0 == 1: # Lambda-Only Universe
        return -H_0 * np.sqrt(Omega_lambda0 * a**2)    
    else:
        return -H_0 * np.sqrt(Omega_m0 / a + Omega_k0 + Omega_lambda0 * a**2)

# Hubble parameter as a function of z
def H(z):
    if Omega_k0 == 1:  # Empty Universe
        return H_0 * np.sqrt(Omega_k0 / (1/(1+z))**2)
    elif Omega_m0 == 1: # Matter-Only Univerese
        return H_0 * np.sqrt(Omega_m0 / (1/(1+z))**3)
    elif Omega_lambda0 == 1: # Lambda-Only Universe
        return H_0 * np.sqrt(Omega_lambda0)    
    else:    
        return H_0 * np.sqrt(Omega_m0 / (1/(1+z))**3 + Omega_k0 / (1/(1+z))**2 + Omega_lambda0)

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
    a[i] = max(a[i-1] + dtau * da_dtau(a[i-1]), 1e-10)  # Prevent a from becoming zero or negative
    z[i] = 1 / a[i] - 1
    H_z[i] = H(z[i])
    D_C[i] = D_C[i-1] + (c / a[i-1]) * dtau  # Comoving distance integral
    D_P[i] = D_C[i] # Proper distance
    D_A[i] = D_C[i] / (1 + z[i])  # Angular diameter distance
    D_L[i] = D_C[i] * (1 + z[i])  # Luminosity distance

# Analytical solutions for single-component universes
def analytical_distances(z, Omega_m0, Omega_lambda0):
    Omega_k0 = 1 - Omega_m0 - Omega_lambda0
    if Omega_k0 == 1:  # Empty Universe
        D_P = (c / H_0) * np.log(1+z)
    elif Omega_m0 == 1:  # Matter-Only Universe
        D_P = (2 * c / H_0) * (1 - 1 / np.sqrt(1 + z))  
    elif Omega_lambda0 == 1:  # Lambda-Only Universe
        D_P = (c / H_0) * z 
    elif Omega_lambda0 >= 0.5:  # Lambda-Dominated Universe
        D_P = (c / H_0) * z      
    
    D_A = D_P / (1 + z)  # Angular Diameter Distance
    D_L = (1 + z) * D_P  # Luminosity Distance
    
    return D_P, D_A, D_L    

# Function to extract numerical distances at a specific redshift
def numerical_distances(z_value, z_array, D_C_array, D_A_array, D_L_array):
    # Using np.interp to interpolate values at the given redshift
    D_C_val = np.interp(z_value, z_array, D_C_array)
    D_A_val = np.interp(z_value, z_array, D_A_array)
    D_L_val = np.interp(z_value, z_array, D_L_array)
    return D_C_val, D_A_val, D_L_val

# Redshift values
z_values = [0.01, 0.1, 1] 

# Comparing analytical and numerical for each component universe
def compare_distances(z_values, Omega_m0, Omega_lambda0):
    print(f"Comparison for Omega_m0 = {Omega_m0}, Omega_lambda0 = {Omega_lambda0}")
    
    for z_value in z_values:
        # Numerical distances
        D_C_num, D_A_num, D_L_num = numerical_distances(z_value, z, D_C, D_A, D_L)
        
        # Analytical distances
        D_P_an, D_A_an, D_L_an = analytical_distances(z_value, Omega_m0, Omega_lambda0)
        
        # Print comparison
        print(f"Redshift z = {z_value}:")
        print(f"  Numerical: D_P = {D_C_num:.4f} Mpc, D_A = {D_A_num:.4f} Mpc, D_L = {D_L_num:.4f} Mpc")
        print(f"  Analytical: D_P = {D_P_an:.4f} Mpc, D_A = {D_A_an:.4f} Mpc, D_L = {D_L_an:.4f} Mpc")
        print("\n")

# Comparing distances for each type of universe
compare_distances(z_values, Omega_m0, Omega_lambda0)


# Plot for scale factor a(tau)
plt.plot(tau, a)
plt.xlabel('Lookback Time τ (Gigayears)')
plt.ylabel('Scale Factor a(τ)')
plt.title('Scale Factor a(τ)')
plt.grid(True)
plt.savefig('scale factor.png')

# Plot for redshift z(tau)
plt.clf()
plt.plot(z,tau)
plt.xlabel('Lookback Time τ (Gigayears)')
plt.ylabel('Redshift z')
plt.title('Redshift z(τ)')
plt.grid(True)
plt.savefig('redshift.png')

# Plot for Hubble parameter H(z)
plt.clf()
plt.plot(tau, H_z)
plt.xlabel('Lookback Time τ (Gigayears)')
plt.ylabel('Hubble Parameter H(z) (Gyr⁻¹)')
plt.title('Hubble Parameter H(z)')
plt.grid(True)
plt.savefig('hubble parameter.png')

# Plot for angular diameter distance D_A
plt.clf()
plt.plot(tau, D_A / 1000)  # Convert to Mpc
plt.xlabel('Lookback Time τ (Gigayears)')
plt.ylabel('Angular Diameter Distance D_A (Mpc)')
plt.title('Angular Diameter Distance D_A(τ)')
plt.grid(True)
plt.savefig('D_A.png')

# Plot for proper distance D_P
plt.clf()
plt.plot(tau, D_P / 1000)  # Convert to Mpc
plt.xlabel('Lookback Time τ (Gigayears)')
plt.ylabel('Proper Distance D_P (Mpc)')
plt.title('Proper Distance D_P(τ)')
plt.grid(True)
plt.savefig('D_P.png')

# Plot for luminosity distance D_L
plt.clf()
plt.plot(tau, D_L / 1000)  # Convert to Mpc
plt.xlabel('Lookback Time τ (Gigayears)')
plt.ylabel('Luminosity Distance D_L (Mpc)')
plt.title('Luminosity Distance D_L(τ)')
plt.grid(True)
plt.savefig('D_L.png')