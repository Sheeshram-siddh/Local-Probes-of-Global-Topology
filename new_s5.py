import numpy as np
import matplotlib.pyplot as plt #For plotting
from scipy.integrate import quad #For numerical integration 
import warnings #To avoid any sigularity relted warnings
warnings.filterwarnings("ignore")



E     = 1.0
alpha = 1.0


def build_lattice(L1, L2, Lambda):
    #Max n amd m value required for particular radial cutoff
    m_max = int(Lambda / L1) + 1
    n_max = int(Lambda / L2) + 1
    #arrays
    m_list = []
    n_list = []
    lmn_list = []
    for m in range(-m_max, m_max + 1):
        for n in range(-n_max, n_max + 1):
            if m == 0 and n == 0:
                continue
            l_mn = np.sqrt((m * L1)**2 + (n * L2)**2)
            if l_mn < Lambda:
                m_list.append(m)
                n_list.append(n)
                lmn_list.append(l_mn)
    M_out = np.array(m_list)
    N_out = np.array(n_list)
    #Sorting of the values
    order = np.argsort(np.array(lmn_list))
    return M_out[order], N_out[order]


def find_critical_time(tau0, alpha, L1, L2, Lambda, delta_max=6.0):
    #Calling required arrays of n and m 
    M_arr, N_arr = build_lattice(L1, L2, Lambda)
    
    smallest_sc = delta_max

    for m, n in zip(M_arr, N_arr):
        lmn_sq = (m * L1)**2 + (n * L2)**2

        def f(s):
            K = n * (L2 / alpha) * np.sinh((tau0 + s / 2.0) / alpha)
            D = lmn_sq / (4 * alpha**2)
            X = (K + np.sqrt(K**2 + 4 * D)) / 2.0
            return s - 2 * alpha * np.arcsinh(X) # This is basically our root or say f(s)
#Root finding using method of disection (changing of sign)
        s_lo = 1e-5
        s_hi = 20.0
        if f(s_lo) * f(s_hi) > 0:
            continue

        for _ in range(60):
            s_mid = (s_lo + s_hi) / 2.0
            if f(s_mid) * f(s_lo) < 0:
                s_hi = s_mid
            else:
                s_lo = s_mid

        s_root = (s_lo + s_hi) / 2.0
        if 0.001 < s_root < smallest_sc:
            smallest_sc = s_root

    return smallest_sc

#Complete integrand, first finding summations in limited tolerence limit and then multiple and sum the cos and s terms
def integrand(s, tau, alpha, L1, L2, E, M_arr, N_arr):
    

    if s < 1e-4:
        result = 1.0 / (12 * alpha**2) + (E**2) / 2.0
        if len(M_arr) > 0:
            lmn_sq = (M_arr * L1)**2 + (N_arr * L2)**2
            result += np.sum(1.0 / lmn_sq)
        return result

    sinh_half = np.sinh(s / (2 * alpha))
    S = -1.0 / (4 * alpha**2 * sinh_half**2)

    if len(M_arr) > 0:
        piece1 = 4 * alpha**2 * sinh_half**2
        piece2 = 4 * alpha * N_arr * L2 * np.sinh((2*tau - s) / (2*alpha)) * sinh_half
        piece3 = (M_arr * L1)**2 + (N_arr * L2)**2
        denoms = -piece1 - piece2 + piece3

        if np.any(np.abs(denoms) < 1e-12):
            return np.nan

        S += np.sum(1.0 / denoms)

    return np.cos(E * s) * S + 1.0 / s**2

#Assemble everything at one place and perform integration using quad function
def transition_rate(delta, tau0, alpha, L1, L2, E, M_arr, N_arr):
    
    tau = tau0 + delta
    integral_val, _ = quad(
        integrand, 0, delta,
        args=(tau, alpha, L1, L2, E, M_arr, N_arr),
        limit=500, epsabs=1e-5, epsrel=1e-5
    )
    return -E / (4 * np.pi) + integral_val / (2 * np.pi**2) + 1.0 / (2 * np.pi**2 * delta)

#Finally stroring the values of the rate corresponding to each value in n and m arrays. 
def compute_curve(tau0, L1, L2, alpha, E, Lambda, num_points=150, delta_max=5.5):
    
    sc = find_critical_time(tau0, alpha, L1, L2, Lambda, delta_max)
    d_end = min(sc - 0.005, delta_max)

    if d_end <= 0.05:
        return np.array([]), np.array([])

    M_arr, N_arr = build_lattice(L1, L2, Lambda)

    t = np.linspace(0, 1, num_points)
    deltas = 0.02 + (d_end - 0.02) * (1 - (1 - t)**3)

    rates = []
    for d in deltas:
        try:
            r = transition_rate(d, tau0, alpha, L1, L2, E, M_arr, N_arr)
            if np.isfinite(r):
                rates.append(r)
            else:
                rates.append(np.nan)
        except:
            rates.append(np.nan)

    return deltas, np.array(rates)



# Plots


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

tau0_fixed = 0.0
L1_fixed   = 100.0
L2_fixed   = 5.0
LAMBDA     = 500.0

ylim = (-0.022, 0.045)
xlim = (0, 5.0)


# LEFT PANEL: Fix L1 = 100, vary L2
print("Left panel: varying L2...")
ax = axes[0]

L2_configs = [
    (5.0,     r'$L_2=5/\Delta E$',   '#8B0000'),
    (10.0,    r'$L_2=10/\Delta E$',  '#E05C00'),
    (30.0,    r'$L_2=30/\Delta E$',  '#CC9900'),
    (50.0,    r'$L_2=50/\Delta E$',  '#009933'),
    (10000.0, r'$L_2=\infty$',       '#006699'),
]

for L2, label, color in L2_configs:
    deltas, rates = compute_curve(tau0_fixed, L1_fixed, L2, alpha, E, LAMBDA)
    if len(deltas):
        ax.plot(deltas, rates, color=color, lw=1.6, label=label)

ax.axhline(0, color='gray', lw=0.6, ls='--', alpha=0.5)
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xlabel(r'$\Delta = \tau - \tau_0 \;(1/\Delta E)$', fontsize=11)
ax.set_ylabel(r'$\dot{F}_\tau(\Delta E)\;(\Delta E)$', fontsize=11)
ax.set_title(r'$\Delta E>0,\;\alpha=1/\Delta E$' + '\n'
             + r'$\tau_0=0,\;L_1=100/\Delta E$,  varying $L_2$', fontsize=10)
ax.legend(loc='upper right', frameon=False, fontsize=8.5)


# MIDDLE PANEL: Fix L2 = 5, vary L1 
print("Middle panel: varying L1...")
ax = axes[1]

L1_configs = [
    (5.0,     r'$L_1=5/\Delta E$',   '#8B0000'),
    (10.0,    r'$L_1=10/\Delta E$',  '#E05C00'),
    (30.0,    r'$L_1=30/\Delta E$',  '#CC9900'),
    (50.0,    r'$L_1=50/\Delta E$',  '#009933'),
    (10000.0, r'$L_1=\infty$',       '#006699'),
]

for L1, label, color in L1_configs:
    deltas, rates = compute_curve(tau0_fixed, L1, L2_fixed, alpha, E, LAMBDA)
    if len(deltas):
        ax.plot(deltas, rates, color=color, lw=1.6, label=label)

ax.axhline(0, color='gray', lw=0.6, ls='--', alpha=0.5)
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xlabel(r'$\Delta = \tau - \tau_0 \;(1/\Delta E)$', fontsize=11)
ax.set_ylabel(r'$\dot{F}_\tau(\Delta E)\;(\Delta E)$', fontsize=11)
ax.set_title(r'$\Delta E>0,\;\alpha=1/\Delta E$' + '\n'
             + r'$\tau_0=0,\;L_2=5/\Delta E$,  varying $L_1$', fontsize=10)
ax.legend(loc='upper right', frameon=False, fontsize=8.5)


# RIGHT PANEL: Fix L1 = 100, L2 = 5, vary tau0 
print("Right panel: varying tau0...")
ax = axes[2]

tau0_configs = [
    (1.5,   '#6600CC'),
    (1.0,   '#1A6B9A'),
    (0.5,   '#009933'),
    (0.0,   '#000000'),
    (-0.5,  '#E05C00'),
    (-1.0,  '#CC0000'),
    (-1.5,  '#8B0000'),
]

for tau0, color in tau0_configs:
    deltas, rates = compute_curve(tau0, L1_fixed, L2_fixed, alpha, E, LAMBDA)
    label = fr'$\tau_0={tau0}/\Delta E$'
    if len(deltas):
        ax.plot(deltas, rates, color=color, lw=1.6, label=label)

ax.axhline(0, color='gray', lw=0.6, ls='--', alpha=0.5)
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xlabel(r'$\Delta = \tau - \tau_0 \;(1/\Delta E)$', fontsize=11)
ax.set_ylabel(r'$\dot{F}_\tau(\Delta E)\;(\Delta E)$', fontsize=11)
ax.set_title(r'$\Delta E>0,\;\alpha=1/\Delta E$' + '\n'
             + r'$L_1=100/\Delta E, L_2=5/\Delta E$,  varying $\tau_0$', fontsize=10)
ax.legend(loc='upper right', frameon=False, fontsize=8.5,
          title=r'$L_1=100/\Delta E, L_2=5/\Delta E$')

plt.tight_layout()
plt.savefig("Section_5_final.png", dpi=300)
print("Saved: Section_5_final.png")
plt.show()