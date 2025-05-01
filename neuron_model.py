import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
import sys

#-----------------------------------------------
# MODEL INPUTS

I_app = 14.65904006 #4.51 #
I_spike = I_app
spike_intervals = [[0, 1]]

V_0 = 0
w_0 = 0.3
T = 50

#-----------------------------------------------
# MODEL PARAMETERS

model = sys.argv[1]
neuron = sys.argv[2]
if model != "LT" and model != "HT":
    raise ValueError("Invalid model selection")

theta_m = -20
k_m = 15
sigma_m = k_m * 2

theta_w = -45 if model == "LT" else -25
k_w = 5
sigma_w = k_w * 2

tau_m_bar = 10 # Unused
tau_w_bar = 25 # Unused

#-----------------------------------------------
# PHYSICAL CONSTANTS

C = 1

g_Ca = 20 if neuron == "IZH" else 10
g_K = 10
g_L = 8

E_Ca = 60 if neuron == "IZH" else 120
E_K = -90
E_L = -78 if model == "LT" else -80

def I(t):
    for interval_bounds in spike_intervals:
        if t >= interval_bounds[0] and t < interval_bounds[1]:  
            return I_spike
    return I_app

def m_inf(V):
    return 1/2 * (1 + np.tanh((V - theta_m) / sigma_m))

def w_inf(V):
    return 1/2 * (1 + np.tanh((V - theta_w) / sigma_w))

def tau_m(V):
    # return tau_m_bar / math.cosh((V - theta_m) / (2*sigma_m))
    return 1

def tau_w(V):
    # return tau_w_bar / math.cosh((V - theta_w) / (2*sigma_w))
    return 1

#-----------------------------------------------
# ODEs

def V_dot(V, w, t):
    return (I(t) - g_Ca * m_inf(V) * (V - E_Ca) - g_K * w * (V - E_K) - g_L * (V - E_L)) / C

def w_dot(V, w):
    return -(w - w_inf(V)) / tau_w(V)

# y = (V, w)
def f(t, y):
    V = y[0]
    w = y[1]

    return [V_dot(V, w, t), w_dot(V, w)]

def f0(y):
    return f(1, y)

#-----------------------------------------------
# NULLCLINES

def calculate_w_nullcline():
    Vs = np.linspace(-80, 60, 10**4)
    ws = [w_inf(V_value) for V_value in Vs]

    return [Vs, ws]

def V_nullcline_formula(V):
    w = (I_app + g_Ca * m_inf(V) * (E_Ca - V) + g_L * (E_L - V)) / (g_K * (V - E_K))

    return w

def calculate_V_nullcline():
    Vs = np.linspace(-100, 100, 10**4)
    ws = [V_nullcline_formula(V_value) for V_value in Vs]

    return [Vs, ws]

#-----------------------------------------------
# MAIN CODE

def visualize(solution, ts):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    w_nullcline = calculate_w_nullcline()
    ax1.plot(w_nullcline[0], w_nullcline[1], marker=',', markersize=0.02, color='grey')

    V_nullcline = calculate_V_nullcline()
    ax1.plot(V_nullcline[0], V_nullcline[1], marker=',', markersize=0.02, color='grey')

    ax1.plot(solution.y[0], solution.y[1], marker=',', markersize=0.1, color='black')
    ax1.set_xlabel('V')
    ax1.set_ylabel('w')

    ax1.set_ylim(-0.2, 1)
    ax1.set_xlim(-80, 40)

    # ax1.plot(solution.y[0, -1], solution.y[1, -1], marker='o', fillstyle='left', markersize=7, color='black')
    # vect_I = np.vectorize(I)

    ax2.plot(solution.t, solution.y[0], marker=',', markersize=0.3, color='black')
    ax2.set_xlabel('t')
    ax2.set_ylabel('V')
    ax2.grid()

    X = np.arange(-80, 60, 5)
    Y = np.arange(-0.2, 1, 0.05)

    V_grid, W_grid = np.meshgrid(X, Y)
    dV_grid, dW_grid = np.meshgrid(X, Y)

    for i in range(V_grid.shape[0]):
        for j in range(V_grid.shape[1]):
            dV_grid[i, j] = V_dot(V_grid[i, j], W_grid[i, j], ts[0])
            dW_grid[i, j] = w_dot(V_grid[i, j], W_grid[i, j])

    norm = np.sqrt(dV_grid**2 + dW_grid**2)
    dV_norm = dV_grid / norm
    dW_norm = dW_grid / norm

    q = ax1.quiver(X, Y, dV_norm, dW_norm, color='purple')

    #plt.show()
    return fig, (ax1, ax2)

if __name__ == '__main__':
    ts = np.linspace(0, T, 10**5)

    solution = solve_ivp(f, [0, T], [V_0, w_0], t_eval=ts)
    fig, (ax1, ax2) = visualize(solution, ts)

    plt.show()


""" 
Saddle-node bifurcation Izhikevich model:

I, V:  [  4.51286763 -60.93251761]
w:  0.0007561581829480524
Jacobian:
[[ 4.39259892e-02 -2.90674824e+02]
 [ 1.51117282e-04 -1.00000000e+00]]
[[   0.04392599 -290.67482386]
 [   0.00015112   -1.        ]]
Eigenvalues and eigenvectors:
[ 0.         -0.95607401]
[[0.99999999 0.99999408]
 [0.00015112 0.00344025]]
 """