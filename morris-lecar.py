import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

# Bifurcation parameter
I = 40

# Parameters
theta_m = -20
V_half_m = 15
sigma_m = V_half_m * 2

theta_w = -45
V_half_w = 5
sigma_w = V_half_w * 2

tau_m_bar = 10 # Unused
tau_w_bar = 25 # Unused

# Physical constants
C = 1

g_Ca = 20
g_K = 10
g_L = 8

E_Ca = 60
E_K = -90
E_L = -78

def m_inf(V):
    return 1/2 * (1 + math.tanh((V - theta_m) / sigma_m))

def w_inf(V):
    return 1/2 * (1 + math.tanh((V - theta_w) / sigma_w))

def tau_m(V):
    # return tau_m_bar / math.cosh((V - theta_m) / (2*sigma_m))
    return 1

def tau_w(V):
    # return tau_w_bar / math.cosh((V - theta_w) / (2*sigma_w))
    return 1

# ODEs
def V_dot(V, w):
    return (I - g_Ca * m_inf(V) * (V - E_Ca) - g_K * w * (V - E_K) - g_L * (V - E_L)) / C

def w_dot(V, w):
    return -(w - w_inf(V)) / tau_w(V)

ts = np.linspace(0, 50, 10**5)

# y = (V, w)
def f(t, y):
    V = y[0]
    w = y[1]

    return [V_dot(V, w), w_dot(V, w)]

solution = solve_ivp(f, [0, 50], [-30, 0.1], t_eval=ts)

print(solution.t)
print(solution.y[0])
print(solution.t.size)


def visualize(solution):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(solution.y[0], solution.y[1], marker=',', markersize=0.1, color='black')
    ax1.set_xlabel('V')
    ax1.set_ylabel('w')

    # ax1.plot(solution.y[0, -1], solution.y[1, -1], marker='o', fillstyle='left', markersize=7, color='black')

    ax2.plot(solution.t, solution.y[0], marker=',', markersize=0.3, color='black')
    ax2.set_xlabel('t')
    ax2.set_ylabel('V')
    ax2.grid()


    X = np.arange(-80, 60, 5)
    Y = np.arange(0, 1, 0.05)

    V_grid, W_grid = np.meshgrid(X, Y)
    dV_grid, dW_grid = np.meshgrid(X, Y)

    for i in range(V_grid.shape[0]):
        for j in range(V_grid.shape[1]):
            dV_grid[i, j] = V_dot(V_grid[i, j], W_grid[i, j])
            dW_grid[i, j] = w_dot(V_grid[i, j], W_grid[i, j])

    norm = np.sqrt(dV_grid**2 + dW_grid**2)
    dV_norm = dV_grid / norm
    dW_norm = dW_grid / norm

    q = ax1.quiver(X, Y, dV_norm, dW_norm, color='purple')

    plt.show()

visualize(solution)

""" 
# Grid for vector field
V_vals = np.linspace(-80, 60, 30)
w_vals = np.linspace(0, 1, 30)


# Compute derivatives
dV = []
for V in V_vals:
    dV.append(V_dot()) 


dW = []
for w in w_vals:
    dW.append(w_dot(w)) 

# Normalize arrows for better visualization
magnitude = np.sqrt(dV**2 + dW**2)
dV_norm = dV / magnitude
dW_norm = dW / magnitude

# Plotting vector field
plt.figure(figsize=(8, 6))
plt.quiver(V_vals, w_vals, dV_norm, dW_norm, angles='xy', scale=25, width=0.002)
plt.xlabel('Membrane Potential V (mV)')
plt.ylabel('Gating Variable w')
plt.title('Morris–Lecar Phase Space Vector Field')
plt.grid(True)
plt.tight_layout()
plt.show() """