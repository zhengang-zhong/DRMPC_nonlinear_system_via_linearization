import numpy as np
import casadi as ca
from DRO_class_multiple_constraints_learning import Simulation


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

if __name__ == "__main__":
    # np.random.seed(2)
    Nx = 2
    Nu = 1
    x_SX = ca.SX.sym("x_SX", Nx)
    u_SX = ca.SX.sym("u_SX", Nu)

    ode = ca.Function("ode_func", [x_SX, u_SX], [academic_ode(x_SX, u_SX)])

    D = np.array([[0], [0.05]])

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
    g = np.array([[100], [100]])
    F_t = np.array([[0, 1]])
    f_t = np.array([[0.5]])

    Q = np.diag([100, 1])
    R = np.diag([1])
    Qf = np.diag([100, 1])

    K = np.array([[-9.03373672, -3.84972813]])


    sin_const = 1
    N_sim = 100
    N_cmax = 4
    N_sample = 1
    epsilon = 5


    H = np.vstack((np.diag([1] * N * Nd), np.diag([-1] * N * Nd)))
    h = np.vstack((sin_const * np.ones([N * Nd, 1]), sin_const * np.ones([N * Nd, 1])))

    mass_sim = Simulation(ode, delta_t, N, x_init, D, F, f, G, g, F_t, f_t, H, h, Q, Qf, R, K, cont_time=True,
                          nonlinear=True, u_0=u_0,
                          xr=xr, ur=ur, collect=True, est=False, sin_const=sin_const,
                          N_sample=N_sample, epsilon=epsilon, N_sim=N_sim, data_set=None, N_sample_max=None, N_cmax = N_cmax)

    mass_sim.plot_state()