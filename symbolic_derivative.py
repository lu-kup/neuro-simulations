import sympy as sp
import sys

# Define the symbols
V, theta_m, sigma_m, g_Ca, E_Ca, I, g_L, E_L, theta_w, sigma_w, miu, omega, g_K, E_K = sp.symbols('V theta_m sigma_m g_ca E_Ca I g_L E_L theta_w sigma_w miu omega g_K E_K')


# Get command-line arguments
model = sys.argv[1]  # 'LT' or 'HT'
neuron = sys.argv[2]  # 'IZH' or 'ML'

# Validate inputs
if model not in ("LT", "HT"):
    raise ValueError("Invalid model: must be 'LT' or 'HT'")
if neuron not in ("IZH", "ML"):
    raise ValueError("Invalid neuron: must be 'IZH' or 'ML'")


# Define w_inf in terms of tanh
m_inf = (1/2) * (1 + sp.tanh((V - theta_m)/sigma_m))
w_inf = (1/2) * (1 + sp.tanh((V - theta_w)/sigma_w))

F = I - g_Ca * m_inf * (V - E_Ca) - g_L * (V - E_L)
G = w_inf

# Differentiate w_inf with respect to V
F_prime = sp.diff(F, V)
F_second = sp.diff(F_prime, V)
F_third = sp.diff(F_second, V)

G_prime = sp.diff(G, V)
G_second = sp.diff(G_prime, V)

# Display the derivative

print(sp.latex(F))
print()
print(sp.latex(G))

g_xx = (miu * g_K / omega) * (2 * G_prime + (V - E_K) * G_second) - (miu / omega) * F_second
a = F_third / 16 - (F_second * g_xx) / (16 * omega)



# Build symbol-to-value substitution dictionary
subs_dict = {
    V: 0,
    omega: 1,
    theta_m: -20,
    sigma_m: 30,  # 2 * k_m
    sigma_w: 10,  # 2 * k_w
    miu: 1,
    g_K: 10,
    g_L: 8,
    E_K: -90
}

# Model-specific entries
subs_dict[theta_w] = -45 if model == "LT" else -25
subs_dict[E_L] = -78 if model == "LT" else -80

# Neuron-specific entries
subs_dict[E_Ca] = 60 if neuron == "IZH" else 120
subs_dict[g_Ca] = 20 if neuron == "IZH" else 10

# Evaluate symbolic expression with substitutions
a_evaluated = a.subs(subs_dict).evalf()
print(a_evaluated)