import numpy as np
import matplotlib.pyplot as plt




def build_lattice(L1, L2, Lambda):
    #same technique to build the requireed arrays using radial cutoff 
    m_max = int(Lambda / L1) + 1
    n_max = int(Lambda / L2) + 1
    m_list = []
    n_list = []
    lmn_list = []
    for m in range(-m_max, m_max + 1):
        for n in range(-n_max, n_max + 1):
            if m == 0 and n == 0:
                continue
            lmn = np.sqrt((m * L1)**2 + (n * L2)**2)
            if lmn < Lambda:
                m_list.append(m)
                n_list.append(n)
                lmn_list.append(lmn)
    M_out = np.array(m_list)
    N_out = np.array(n_list)
    order = np.argsort(np.array(lmn_list))
    return M_out[order], N_out[order]


def unruh_planck(DeltaE, alpha):
   #Planck spectrum term
    x = 2.0 * np.pi * alpha * DeltaE
    if x > LAMBDA:
        return (DeltaE / (2.0 * np.pi)) * np.exp(-x)
    elif x < -LAMBDA:
        return -DeltaE / (2.0 * np.pi)
    else:
        return (DeltaE / (2.0 * np.pi)) / (np.exp(x) - 1.0)


def deexcitation_rate(DeltaE, alpha, L1, L2, Lambda):
    

    # Unruh piece (same for excitation and de-excitation)
    rate = unruh_planck(DeltaE, alpha)

    # Topological correction only for de-excitation 
    if DeltaE >= 0:
        return rate

    absDE = abs(DeltaE)
    M_arr, N_arr = build_lattice(L1, L2, Lambda)
    ell = np.sqrt((M_arr * L1)**2 + (N_arr * L2)**2)

    # Each (m,n) contributes: sin(2 alpha |DE| arcsinh(ell/2alpha)) / (pi ell sqrt(1 + (ell/2alpha)^2))
    ratio = ell / (2.0 * alpha)
    arg = 2.0 * alpha * absDE * np.arcsinh(ratio)
    denom = np.pi * ell * np.sqrt(1.0 + ratio**2)
    correction = np.sum(np.sin(arg) / denom)

    return rate - correction



# Plots


fig, axes = plt.subplots(2, 2, figsize=(15, 11))

LAMBDA   = 1000.0
fixed_DE = -1.0


# Panel 1: Rate vs L1 at fixed L2 
print("Panel 1: Rate vs L1...")
ax = axes[0, 0]

L1_range = np.linspace(5.0, 200.0, 800)
fixed_L2 = 100.0

alpha_configs = [
    (5.0,   r'$\alpha = 5/|\Delta E|$',   '#8B0000'),
    (20.0,  r'$\alpha = 20/|\Delta E|$',  '#E05C00'),
    (50.0,  r'$\alpha = 50/|\Delta E|$',  '#009933'),
    (100.0, r'$\alpha = 100/|\Delta E|$', '#006699'),
]

for alph, lbl, col in alpha_configs:
    rates = [deexcitation_rate(fixed_DE, alph, L1, fixed_L2, LAMBDA) for L1 in L1_range]
    ax.plot(L1_range, rates, color=col, lw=1.3, label=lbl)

pu = unruh_planck(fixed_DE, 50.0)
ax.axhline(pu, color='black', ls='--', lw=0.9, alpha=0.4, label=r'Unruh ($\alpha=50$)')

ax.set_xlabel(r'$L_1\;(1/|\Delta E|)$', fontsize=12)
ax.set_ylabel(r'$\dot{F}_{\rm eq}(\Delta E)\;(|\Delta E|)$', fontsize=12)
ax.set_title(rf'$\Delta E = -1$, $L_2 = {int(fixed_L2)}/|\Delta E|$', fontsize=10)
ax.legend(loc='upper right', frameon=False, fontsize=8.5)
ax.grid(True, alpha=0.15)


# Panel 2: Rate vs L2 at fixed L1 
print("Panel 2: Rate vs L2...")
ax = axes[0, 1]

L2_range = np.linspace(5.0, 200.0, 800)
fixed_L1 = 100.0

for alph, lbl, col in alpha_configs:
    rates = [deexcitation_rate(fixed_DE, alph, fixed_L1, L2, LAMBDA) for L2 in L2_range]
    ax.plot(L2_range, rates, color=col, lw=1.3, label=lbl)

ax.axhline(pu, color='black', ls='--', lw=0.9, alpha=0.4, label=r'Unruh ($\alpha=50$)')

ax.set_xlabel(r'$L_2\;(1/|\Delta E|)$', fontsize=12)
ax.set_ylabel(r'$\dot{F}_{\rm eq}(\Delta E)\;(|\Delta E|)$', fontsize=12)
ax.set_title(rf'$\Delta E = -1$, $L_1 = {int(fixed_L1)}/|\Delta E|$', fontsize=10)
ax.legend(loc='upper right', frameon=False, fontsize=8.5)
ax.grid(True, alpha=0.15)


# Panel 3: Rate vs acceleration 
print("Panel 3: Rate vs acceleration...")
ax = axes[1, 0]

accel_range = np.linspace(0.005, 0.5, 600)
alpha_range = 1.0 / accel_range

torus_configs = [
    (30.0,  30.0,  r'$L_1=30,\;L_2=30$  (square)',        '#8B0000'),
    (50.0,  50.0,  r'$L_1=50,\;L_2=50$  (square)',        '#E05C00'),
    (30.0,  100.0, r'$L_1=30,\;L_2=100$ (rect.)',         '#009933'),
    (100.0, 100.0, r'$L_1=100,\;L_2=100$ (large square)', '#006699'),
]

for L1, L2, lbl, col in torus_configs:
    rates = [deexcitation_rate(fixed_DE, alph, L1, L2, LAMBDA) for alph in alpha_range]
    ax.plot(accel_range, rates, color=col, lw=1.3, label=lbl)

rates_pu = [unruh_planck(fixed_DE, alph) for alph in alpha_range]
ax.plot(accel_range, rates_pu, 'k--', lw=1.2, label=r'Pure Unruh ($L\to\infty$)', alpha=0.6)

ax.set_xlabel(r'Acceleration $a = 1/\alpha\;(|\Delta E|^2)$', fontsize=12)
ax.set_ylabel(r'$\dot{F}_{\rm eq}(\Delta E)\;(|\Delta E|)$', fontsize=12)
ax.set_title(r'$\Delta E = -1$', fontsize=10)
ax.legend(loc='upper left', frameon=False, fontsize=8.5)
ax.grid(True, alpha=0.15)


# Panel 4: Rate vs |DeltaE| 
print("Panel 4: Rate vs |DeltaE|...")
ax = axes[1, 1]

DE_range    = np.linspace(-0.1, -10.0, 500)
absDE_range = np.abs(DE_range)
fixed_alpha = 30.0

spectral_configs = [
    (15.0, 15.0, r'$L_1=15,\;L_2=15$', '#8B0000'),
    (30.0, 30.0, r'$L_1=30,\;L_2=30$', '#E05C00'),
    (30.0, 60.0, r'$L_1=30,\;L_2=60$', '#009933'),
    (60.0, 60.0, r'$L_1=60,\;L_2=60$', '#006699'),
]

rates_pu_4 = [unruh_planck(dE, fixed_alpha) for dE in DE_range]
ax.plot(absDE_range, rates_pu_4, 'k--', lw=1.2, label=r'Pure Unruh ($L\to\infty$)', alpha=0.6)

for L1, L2, lbl, col in spectral_configs:
    rates = [deexcitation_rate(dE, fixed_alpha, L1, L2, LAMBDA) for dE in DE_range]
    ax.plot(absDE_range, rates, color=col, lw=1.3, label=lbl)

ax.set_xlabel(r'$|\Delta E|$', fontsize=12)
ax.set_ylabel(r'$\dot{F}_{\rm eq}(\Delta E)\;(|\Delta E|)$', fontsize=12)
ax.set_title(rf'$\alpha = {int(fixed_alpha)}/|\Delta E|$', fontsize=10)
ax.legend(loc='upper left', frameon=False, fontsize=9)
ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.3)
ax.grid(True, alpha=0.15)


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Section_6_figure.png', dpi=300, bbox_inches='tight')
print("Saved: Section_6_figure.png")
plt.show()