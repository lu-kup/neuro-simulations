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

def get_jacobian_eigen(V_solution, w_solution):
    print("Jacobian:")
    jac = jacobian(f0, [V_solution, w_solution])
    print(jac.df)
    eval, evec = np.linalg.eig(jac.df)
    print("Eigenvalues and eigenvectors:")
    print(eval)
    print(evec)

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

if __name__ == '__main__':
    # get_jacobian_eigen(-56.5, 0.09)
    solve_bifurcation(andronov_hopf)
