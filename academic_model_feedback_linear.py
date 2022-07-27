import numpy as np
import casadi as ca
from DRO_class_feedback_linearization import Simulation


def academic_ode(x, u):
    '''
    state = [cA, cB, theta, theta_K]

    input = [V_dot/VR, QK_dot]

    ki(theta) = ki0 * exp(Ei / (theta + 273.15))
    '''

    m = 2  # [kg]
    k1 = 3  # [N/m]
    k2 = 2  # [N/m]

    x1p = x[1]
    x2p = -k2 / m * x[0] ** 3 - k1 / m * x[1] + 1 / m * u[0]

    rhs = [x1p,
           x2p
           ]

    return ca.vertcat(*rhs)

def academic_ode_linear(x, nu):
    '''
    state = [cA, cB, theta, theta_K]

    input = [V_dot/VR, QK_dot]

    ki(theta) = ki0 * exp(Ei / (theta + 273.15))
    '''

    m = 2  # [kg]
    k1 = 3  # [N/m]
    k2 = 2  # [N/m]

    x1p = x[1]
    x2p = nu - k1 / m * x[1]

    rhs = [x1p,
           x2p
           ]

    return ca.vertcat(*rhs)

def policy(x, nu):
    '''
    u=k_{2} x_{1}^{3}+m \nu_{i}
    '''

    m = 2  # [kg]
    k1 = 3  # [N/m]
    k2 = 2  # [N/m]

    u = k2 * x[0]**3 + m * nu

    return u


if __name__ == "__main__":
    # np.random.seed(2)
    Nx = 2
    Nu = 1
    N_nu = 1
    x_SX = ca.SX.sym("x_SX", Nx)
    u_SX = ca.SX.sym("u_SX", Nu)
    nu_SX = ca.SX.sym("u_SX", N_nu)

    ode = ca.Function("ode_func", [x_SX, u_SX], [academic_ode(x_SX, u_SX)])
    ode_linear = ca.Function("linear_ode_func", [x_SX, nu_SX], [academic_ode_linear(x_SX, nu_SX)])
    policy_fn = ca.Function("linear_ode_func", [x_SX, nu_SX], [policy(x_SX, nu_SX)])

    D = np.array([[0],[0.1]])


    Nd = np.shape(D)[1]
    N = 5

    delta_t = 0.1

    x_init = np.array([[-2.0], [0]])
    u_0 = np.array([[0.]])

    xr = np.array([[0.], [0.]])
    ur = np.array([[0.]])


    F = np.array([[0, 1]])
    f = np.array([[0.5]])


    G = np.array([[-1], [1]])
    g = np.array([[10], [10]])
    F_t = np.array([[0, 1]])
    f_t = np.array([[0.5]])

    Q = np.diag([100, 1])
    R = np.diag([1])
    Qf = np.diag([100, 1])

    K = np.array([[-2.92030722, -1.50654301]])

    sin_const = 1
    N_sim = 100
    N_sample = 1
    epsilon = 1e-3


    H = np.vstack((np.diag([1] * N * Nd), np.diag([-1] * N * Nd)))
    h = np.vstack((sin_const * np.ones([N * Nd, 1]), sin_const * np.ones([N * Nd, 1])))

    # nu = [0, 2, 0, 0, 0] * N +[0, 0, 0, 0, 0]

    mass_sim = Simulation(ode, ode_linear, policy_fn, delta_t, N, x_init, D, F, f, G, g, F_t, f_t, H, h, Q, Qf, R, K, cont_time=True, nonlinear=True, u_0=u_0,
     xr=xr, ur=ur, collect=True, est=False, sin_const=sin_const, N_sample=N_sample, epsilon=epsilon, N_sim=N_sim, data_set=None, N_sample_max=None)

    mass_sim.plot_state()