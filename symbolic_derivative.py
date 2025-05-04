import sympy as sp
import sys

V, theta_m, sigma_m, g_Ca, E_Ca, I, g_L, E_L, theta_w, sigma_w, miu, omega, g_K, E_K = sp.symbols('V theta_m sigma_m g_ca E_Ca I g_L E_L theta_w sigma_w miu omega g_K E_K')

# Model configuration
model = sys.argv[1]
neuron = sys.argv[2]

if model not in ("LT", "HT"):
    raise ValueError("Invalid model: must be 'LT' or 'HT'")
if neuron not in ("IZH", "ML"):
    raise ValueError("Invalid neuron: must be 'IZH' or 'ML'")


# Main equations
m_inf = (1/2) * (1 + sp.tanh((V - theta_m)/sigma_m))
w_inf = (1/2) * (1 + sp.tanh((V - theta_w)/sigma_w))

F = I - g_Ca * m_inf * (V - E_Ca) - g_L * (V - E_L)
G = w_inf

print(sp.latex(F))
print(sp.latex(G))

# Derivatives and define secondary equations
F_prime = sp.diff(F, V)
F_second = sp.diff(F_prime, V)
F_third = sp.diff(F_second, V)

G_prime = sp.diff(G, V)
G_second = sp.diff(G_prime, V)

g_xx = (miu * g_K / omega) * (2 * G_prime + (V - E_K) * G_second) - (miu / omega) * F_second
a = F_third / 16 - (F_second * g_xx) / (16 * omega)


# Model parameters
subs_dict = {
    V: -56.48148543,
    omega: 2.13747717307879,
    theta_m: -20,
    sigma_m: 30,  # 2 * k_m
    sigma_w: 10,  # 2 * k_w
    miu: 1,
    g_K: 10,
    g_L: 8,
    E_K: -90
}

subs_dict[theta_w] = -45 if model == "LT" else -25
subs_dict[E_L] = -78 if model == "LT" else -80

subs_dict[E_Ca] = 60 if neuron == "IZH" else 120
subs_dict[g_Ca] = 20 if neuron == "IZH" else 10

# Evaluate
a_evaluated = a.subs(subs_dict).evalf()
print(a_evaluated)