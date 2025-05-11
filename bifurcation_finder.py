import numpy as np
import math
from neuron_model import *
from scipy.optimize import fsolve
from scipy.differentiate import jacobian 

def saddle_node(x):
    I_x = x[0]
    V_x = x[1]
    print(f"Current params: I={I_x} V={V_x}, cosh arg is ", (V_x - theta_m)/sigma_m)

    eq1 = (I_x + g_Ca * m_inf(V_x) * (E_Ca - V_x) + g_L * (E_L - V_x)) / (g_K * (V_x - E_K)) - w_inf(V_x)
    diff1 = m_inf(V_x) - (E_Ca - V_x) / (2 * sigma_m * (np.cosh((V_x - theta_m)/sigma_m))**2)
    expression_num = (g_Ca * diff1 + g_L) * (V_x - E_K) + I_x + g_Ca * m_inf(V_x) * (E_Ca - V_x) + g_L * (E_L - V_x)
    expression_den = g_K * (V_x - E_K)**2
    eq2 = 1 / (2 * sigma_w * (np.cosh((V_x - theta_w) / sigma_w)**2)) + expression_num / expression_den
    print(f"Current error: {1 / (2 * sigma_w * (np.cosh((V_x - theta_w) / sigma_w)**2))} {expression_num / expression_den}")

    return [eq1, eq2]

def andronov_hopf(x):
    I_x = x[0]
    V_x = x[1]
    print(f"Current params: I={I_x} V={V_x}, cosh arg is ", (V_x - theta_m)/sigma_m)

    eq1 = (I_x + g_Ca * m_inf(V_x) * (E_Ca - V_x) + g_L * (E_L - V_x)) / (g_K * (V_x - E_K)) - w_inf(V_x)
    jac = jacobian(f0, [V_x, w_inf(V_x)])
    eq2 = np.trace(jac.df)

    return [eq1, eq2]

def steady_state(V_x, I_x):
    return I_x + g_Ca * m_inf(V_x) * (E_Ca - V_x) + g_L * (E_L - V_x) + g_K * (E_K - V_x) * w_inf(V_x)

def steady_state_V(I_x):
    return fsolve(steady_state, x0=-50, args=I_x)[0]

def get_jacobian_eigen(V_solution, w_solution):
    print("Jacobian:")
    jac = jacobian(f0, [V_solution, w_solution])
    print(jac.df)
    eval, evec = np.linalg.eig(jac.df)
    print("Eigenvalues and eigenvectors:")
    print(eval)
    print(evec)
    print("Determinant:")
    print(np.linalg.det(jac.df))

    np.set_printoptions(suppress=True)
    print("Jacobian:")
    print(jac.df)
    print("Eigenvalues and eigenvectors:")
    print(eval)
    print(evec)

    print("Trace:")
    print(np.trace(jac.df))

def solve_bifurcation(f):
    solution = fsolve(f, [47, -56])
    print("\nRESULT:")
    print("I, V: ", solution)
    w_solution = w_inf(solution[1])
    print("w: ", w_solution)
    get_jacobian_eigen(solution[1], w_solution)
    return [solution[1], w_solution]

def current_second_ca(V):
    sh = 1 / np.cosh((V - theta_m)/sigma_m)
    th = np.tanh((V - theta_m)/sigma_m)
    return g_Ca * sh**2 / sigma_m * (1 - (V - E_Ca) * th / sigma_m)

def current_second_k(V):
    sh = 1 / np.cosh((V - theta_w)/sigma_w)
    th = np.tanh((V - theta_w)/sigma_w)
    return g_K * sh**2 / sigma_w * (1 - (V - E_K) * th / sigma_w)

def current_prime_ca(V):
    sh = 1 / np.cosh((V - theta_m)/sigma_m)
    return g_Ca * (V - E_Ca) * sh**2 / (2 * sigma_m) + g_Ca * m_inf(V)

def current_prime_k(V):
    sh = 1 / np.cosh((V - theta_w)/sigma_w)
    return g_K * (V - E_K) * sh**2 / (2 * sigma_w) + g_K * w_inf(V)

def det_L(V):
    return (current_prime_ca(V) + current_prime_k(V) + g_L) / tau_w(V)

def tr_L(V, w):
    return -1 * (current_prime_ca(V) + g_K * w + g_L + 1/tau_w(V))

def c(V, w):
    return tr_L(V, w) / 2

def omega(V, w):
    return np.sqrt(det_L(V) - c(V, w)**2)

def c_appr(I):
    V = steady_state_V(I)
    w = w_inf(V)
    return c(V, w)

def omega_appr(I):
    V = steady_state_V(I)
    w = w_inf(V)
    return omega(V, w)

def current_inf_second(V):
    return (current_second_ca(V) + current_second_k(V)) / C

def total_current_inf_second(V):
    return -1 * current_inf_second(V)

def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

if __name__ == '__main__':
    # get_jacobian_eigen(-56.5, 0.09)

    solution = solve_bifurcation(andronov_hopf)
    print(solution)
    ats_c = c(solution[0], solution[1])
    print("c =", ats_c)
    ats_omega = omega(solution[0], solution[1])
    print("omega =", ats_omega)

    solved_V = steady_state_V(14.65904006)
    print(solved_V)

    print(numerical_derivative(omega_appr, 14.65904006))
    print(numerical_derivative(c_appr, 14.65904006))
