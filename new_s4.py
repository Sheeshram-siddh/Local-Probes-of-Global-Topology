import numpy as np
import matplotlib.pyplot as plt


def rate_2D(Delta_E, L1, L2, vz, Lambda_cut):

    
    if Delta_E >= 0:
        return 0.0

    gamma = 1.0 / np.sqrt(1.0 - vz**2)

   
    rate_mink = -Delta_E / (2.0 * np.pi)

    #Defining maximum n and m value required for radial distance based sorting

    m_max = int(Lambda_cut / L1) + 1
    n_max = int(Lambda_cut / L2) + 1

# arrange all values of n and m in an array

    m_all = np.arange(-m_max, m_max + 1)
    n_all = np.arange(-n_max, n_max + 1)

    # A n*m matrix to include all values of n and m out of which we will choose pairing of n amd m

    M, N = np.meshgrid(m_all, n_all)

    #Make flat array that has all possible combinationation, so it will just like chossing value from an array

    M = M.flatten()
    N = N.flatten()

    # Remove (0,0)
    nonzero = (M != 0) | (N != 0)
    M = M[nonzero]
    N = N[nonzero]

    
    l_mn = np.sqrt((M * L1)**2 + (N * L2)**2)

    # Keep only those values that are inside our raadl distance wise cutoff

    inside = l_mn < Lambda_cut
    M = M[inside]
    N = N[inside]

  # Sort by increasing l_nm (sum nearest shells first)
    order = np.argsort(l_mn[inside])
    M = M[order]
    N = N[order]

    
    # Relativistic distance
    R_mn = np.sqrt((gamma * N * L2)**2 + (M * L1)**2)

    # The exponential velocity phase term
    vel_phase = Delta_E * gamma * vz * N * L2

    
    terms = (-1.0 / (2.0 * np.pi)) * np.sin(Delta_E * R_mn) / R_mn * np.exp(1j * vel_phase)
    correction = np.real(np.sum(terms))

    return rate_mink + correction


#Functions done and now plots

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

LAMBDA = 10000.0
fixed_DE = -10.0


# Panel 1: Rate vs L1

ax = axes[0, 0]
L1_values = np.linspace(10.0, 1000.0, 300)
fixed_L2 = 500.0
vz_list = [0.25, 0.5, 0.75]

for vz in vz_list:
    rates = [rate_2D(fixed_DE, L1, fixed_L2, vz, LAMBDA) for L1 in L1_values]
    ax.plot(L1_values, rates, lw=1.2, label=f"$v_z = {vz}$")

ax.axhline(-fixed_DE / (2*np.pi), color='k', ls='--', lw=0.8, alpha=0.4, label="Minkowski")
ax.set_title(f"Varying Transverse Size $L_1$ ($L_2 = {fixed_L2:.0f}$)")
ax.set_xlabel("$L_1$")
ax.set_ylabel("Transition Rate")
ax.legend()
ax.grid(True, alpha=0.2)


# Panel 2: Rate vs L2

ax = axes[0, 1]
L2_values = np.linspace(10.0, 1000.0, 300)
fixed_L1 = 500.0

for vz in vz_list:
    rates = [rate_2D(fixed_DE, fixed_L1, L2, vz, LAMBDA) for L2 in L2_values]
    ax.plot(L2_values, rates, lw=1.2, label=f"$v_z = {vz}$")

ax.axhline(-fixed_DE / (2*np.pi), color='k', ls='--', lw=0.8, alpha=0.4, label="Minkowski")
ax.set_title(f"Varying Longitudinal Size $L_2$ ($L_1 = {fixed_L1:.0f}$)")
ax.set_xlabel("$L_2$")
ax.set_ylabel("Transition Rate")
ax.legend()
ax.grid(True, alpha=0.2)


# Panel 3: Rate vs velocity

ax = axes[1, 0]
vz_values = np.linspace(0.01, 0.95, 150)

configs = [
    (1000.0, 1000.0, "Square"),
    (200.0, 1000.0, "Tube"),
    (1000.0, 200.0, "Pancake"),
]

for L1, L2, label in configs:
    rates = [rate_2D(fixed_DE, L1, L2, vz, LAMBDA) for vz in vz_values]
    ax.plot(vz_values, rates, lw=1.2, label=label)

ax.set_title("Varying Velocity $v_z$")
ax.set_xlabel("Velocity $v_z$")
ax.set_ylabel("Transition Rate")
ax.legend()
ax.grid(True, alpha=0.2)


# Panel 4: Rate vs energy gap 

ax = axes[1, 1]
DE_values = np.linspace(-10.0, -0.1, 150)
fixed_vz = 0.5

size_configs = [
    (10.0, 10.0, "Small"),
    (100.0, 100.0, "Medium"),
    (1000.0, 1000.0, "Large"),
]

for L1, L2, label in size_configs:
    rates = [rate_2D(dE, L1, L2, fixed_vz, LAMBDA) for dE in DE_values]
    ax.plot(DE_values, rates, lw=1.2, label=label)

ax.set_title(f"Varying Energy Gap $\\Delta E$ ($v_z = {fixed_vz}$)")
ax.set_xlabel("Energy Gap $\\Delta E$")
ax.set_ylabel("Transition Rate")
ax.legend()
ax.grid(True, alpha=0.2)


plt.tight_layout()
plt.savefig("Section_4_v2.png", dpi=300)
print("Saved: Section_4_v2.png")
plt.show()