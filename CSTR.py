import numpy as np
import casadi as ca
from DRO_class_multiple_constraints import Simulation

def CSTR_ode(x, u):
    '''
    state = [cA, cB, theta, theta_K]

    input = [V_dot/VR, QK_dot]

    ki(theta) = ki0 * exp(Ei / (theta + 273.15))
    '''
    k10 = 1.287e12
    k20 = 1.287e12
    k30 = 9.043e9

    E1 = -9758.3
    E2 = -9758.3
    E3 = -8560
    Delta_H_RAB = 4.2
    Delta_H_RBC = -11.0
    Delta_H_RAD = -41.85

    rho = 0.9342

    Cp = 3.01
    kw = 4032

    AR = 0.215
    VR = 10.01
    mK = 5.0
    CPK = 2.0

    cA0 = 5.10
    theta0 = 130.0

    cA = x[0]
    cB = x[1]
    theta = x[2]
    theta_K = x[3]

    k1 = k10 * ca.exp(E1 / (theta + 273.15))
    k2 = k20 * ca.exp(E2 / (theta + 273.15))
    k3 = k30 * ca.exp(E3 / (theta + 273.15))

    x1p = u[0] * (cA0 - cA) - k1 * cA - k3 * cA ** 2
    x2p = -u[0] * cB + k1 * cA - k2 * cB
    x3p = u[0] * (theta0 - theta) - 1 / (rho * Cp) * (
                k1 * cA * Delta_H_RAB + k2 * cB * Delta_H_RBC + k3 * cA ** 2 * Delta_H_RAD) + (kw * AR) / (
                      rho * Cp * VR) * (theta_K - theta)
    x4p = 1 / (mK * CPK) * (u[1] + kw * AR * (theta - theta_K))

    rhs = [x1p,
           x2p,
           x3p,
           x4p
           ]

    return ca.vertcat(*rhs)

if __name__ == "__main__":
    # np.random.seed(2)
    Nx = 4
    Nu = 2
    x_SX = ca.SX.sym("x_SX", Nx)
    u_SX = ca.SX.sym("u_SX", Nu)

    ode = ca.Function("ode_func", [x_SX, u_SX], [CSTR_ode(x_SX, u_SX)])

    D = np.array([[0.01,0],[0,0],[0,0],[0,1]])


    Nd = np.shape(D)[1]
    N = 4

    delta_t = 0.01

    x_init = np.array([[1.235], [1.0], [134.14], [128.95]])
    u_0 = np.array([[18.83],[-4495.7]])

    xr = np.array([[1.2345],
           [0.89987685],
           [134.15212038],
           [128.96587743]])

    ur = np.array([[18.83],[-4495.7]])


    F = np.array([[-1, 0, 0, 0]])
    f = np.array([[-1.233]])
    G = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    g = np.array([[-5], [35], [8500], [0]])
    F_t = np.array([[-1, 0, 0, 0]])
    f_t = np.array([[-1.233]])

    Q = np.diag([10, 1, 1, 1])
    R = np.diag([1e-1, 1e-1])
    Qf = np.diag([10, 1, 1, 1])

    K = np.array([[-6.33159126, -4.08697435, -1.1737697, -0.5021129],
          [-0.04402787, -0.03782597, -0.01618324, -0.00892606]])

    sin_const = 0.1
    N_sim = 100
    N_sample = 5
    epsilon = 5e-4

    H = np.vstack((np.diag([1] * N * Nd), np.diag([-1] * N * Nd)))
    h = np.vstack((sin_const * np.ones([N * Nd, 1]), sin_const * np.ones([N * Nd, 1])))

    mass_sim = Simulation(ode, delta_t, N, x_init, D, F, f, G, g, F_t, f_t, H, h, Q, Qf, R, K, cont_time=True, nonlinear=True, u_0=u_0,
     xr=xr, ur=ur, collect=True, est=False, sin_const=sin_const, N_sample=N_sample, epsilon=epsilon, N_sim=N_sim, data_set=None, N_sample_max=None)

    mass_sim.plot_state()