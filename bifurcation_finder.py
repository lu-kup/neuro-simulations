import numpy as np
import math
from morris_lecar import *
from scipy.optimize import fsolve

def saddle_node(x):
    I_x = x[0]
    V_x = x[1]
    eq1 = (I_x + g_Ca * m_inf(V_x) * (E_Ca - V_x) + g_L * (E_L - V_x)) / (g_K * (V_x - E_K)) - w_inf(V_x)
    print(f"DEBUG: I={I_x} V={V_x}, cosh arg is ", (V_x - theta_m)/sigma_m)

    diff1 = m_inf(V_x) - (E_Ca - V_x) / (2 * sigma_m * (np.cosh((V_x - theta_m)/sigma_m))**2)

    expression_num = (g_Ca * diff1 + g_L) * (V_x - E_K) + I_x + g_Ca * m_inf(V_x) * (E_Ca - V_x) + g_L * (E_L - V_x)
    expression_den = g_K * (V_x - E_K)**2
    eq2 = 1 / (2 * sigma_w * (np.cosh((V_x - theta_w) / sigma_w)**2)) + expression_num / expression_den
    print(f"EQUATION RESULTS: {1 / (2 * sigma_w * (np.cosh((V_x - theta_w) / sigma_w)**2))} {expression_num / expression_den}")

    return [eq1, eq2]

if __name__ == '__main__':
    solution = fsolve(saddle_node, [47, -56])
    print(solution)
