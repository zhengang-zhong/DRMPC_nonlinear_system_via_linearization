import numpy as np
import casadi as ca
import cvxpy as cp
import mosek
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power
import gurobipy

class Model:
    def __init__(self, sys_fn, delta_t, cont_time=True, nonlinear=False):
        self.sys_fn = sys_fn
        self.delta_t = delta_t
        self.cont_time = cont_time
        self.nonlinear = nonlinear

        self.Nx = sys_fn.sx_in()[0].shape[0]
        self.Nu = sys_fn.sx_in()[1].shape[0]

        self.xk_SX = ca.SX.sym("xk_SX", self.Nx)
        self.uk_SX = ca.SX.sym("uk_SX", self.Nu)

        if cont_time == True:
            self.dt_sys_fn = self.discretize_sys()
        else:
            self.dt_sys_fn = self.sys_fn

    def discretize_sys(self):
        xk_SX = self.xk_SX
        uk_SX = self.uk_SX
        sys_fn = self.sys_fn
        delta_t = self.delta_t
        x_next = self.integrator_rk4(sys_fn, xk_SX, uk_SX, delta_t)
        dt_sys_fn = ca.Function("dt_sys_fn", [xk_SX, uk_SX], [x_next])
        return dt_sys_fn

    def integrator_rk4(self, f, x, u, delta_t):
        '''
        This function calculates the integration of stage cost with RK4.
        '''

        k1 = f(x, u)
        k2 = f(x + delta_t / 2 * k1, u)
        k3 = f(x + delta_t / 2 * k2, u)
        k4 = f(x + delta_t * k3, u)

        x_next = x + delta_t / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next


class Linear_model(Model):
    '''
    Linearized model
    '''

    def __init__(self, sys_fn, delta_t, cont_time=True, nonlinear=False):
        super().__init__(sys_fn, delta_t, cont_time, nonlinear)

        x0_SX = ca.SX.sym("x0_SX", self.Nx)
        u0_SX = ca.SX.sym("u0_SX", self.Nu)
        self.x0_SX = x0_SX
        self.u0_SX = u0_SX

        xk_SX = self.xk_SX
        uk_SX = self.uk_SX
        self.lin_dt_sys_fn = self.linearize_sys(x0_SX, xk_SX, u0_SX, uk_SX)

    def linearize_sys(self, x0_SX, xk_SX, u0_SX, uk_SX):
        dt_sys = self.dt_sys_fn(xk_SX, uk_SX)
        A = ca.jacobian(dt_sys, xk_SX)
        B = ca.jacobian(dt_sys, uk_SX)

        A_fn = ca.Function("A_fn", [xk_SX, uk_SX], [A])
        B_fn = ca.Function("B_fn", [xk_SX, uk_SX], [B])

        self.A_fn = A_fn
        self.B_fn = B_fn

        if self.nonlinear == True:
            x_next = self.dt_sys_fn(x0_SX, u0_SX) + A_fn(x0_SX, u0_SX) @ (xk_SX - x0_SX) + B_fn(x0_SX, u0_SX) @ (
                    uk_SX - u0_SX)
            x_next_lin_fn = ca.Function("x_next_lin_fn", [x0_SX, xk_SX, u0_SX, uk_SX], [x_next])
            self.delta_fn = ca.Function("delta_fn", [x0_SX, u0_SX], [
                self.dt_sys_fn(x0_SX, u0_SX) - A_fn(x0_SX, u0_SX) @ x0_SX - B_fn(x0_SX, u0_SX) @ u0_SX])
            self.A_fn = A_fn
            self.B_fn = B_fn
        else:
            # Linear system is independent of x0 and u0
            x_next = A_fn(xk_SX, uk_SX) @ xk_SX + B_fn(xk_SX, uk_SX) @ uk_SX
            x_next_lin_fn = ca.Function("x_next_lin_fn", [xk_SX, uk_SX], [x_next])
            self.delta_fn = ca.Function("delta_fn", [x0_SX, u0_SX], [np.zeros(xk_SX.shape)])
            self.delta = np.zeros(xk_SX.shape)
            self.A = ca.DM(A)  # Transfer SX to DM
            self.B = ca.DM(B)
        # print(A)
        # print(B)
        return x_next_lin_fn

class Stack_model(Linear_model):
    def __init__(self, sys_fn, delta_t, N, x_init, D, F, f, G, g, F_t, f_t, K, cont_time=True, nonlinear=False, u_0=None, xr=None,
                 ur=None):
        '''

        x_next = Ak x_k + Bk u_k + Ck w_k
        y_k = D x_k + E w_k

        Args:
            A: discretized and linearized
            B: discretized and linearized
            D: Discrete time

        '''
        super().__init__(sys_fn, delta_t, cont_time, nonlinear)

        self.xr = xr
        self.ur = ur
        self.N = N

        self.D = D
        self.F = F
        self.f = f
        self.G = G
        self.g = g
        self.F_t = F_t
        self.f_t = f_t
        self.x_init = x_init

        self.K = K

        Nx = self.Nx
        Nu = self.Nu
        Nd = np.shape(D)[1]  # Dimension of disturbance

        if nonlinear == True:
            self.A = self.A_fn(x_init, u_0).full()
            self.B = self.B_fn(x_init, u_0).full()
            self.delta = self.delta_fn(x_init, u_0).full()
        else:
            self.A = self.A.full()
            self.B = self.B.full()
            self.delta = self.delta  # delta = 0
        # print(self.A,self.B)


        # number of constraints
        Nc  = np.shape(F)[0]
        Nc_t = np.shape(F_t)[0]
        self.Nd = Nd
        self.Nc = Nc
        self.Nc_t = Nc_t

        # dimension of stacked matrices

        Nx_s = (N + 1) * Nx
        Nu_s = N * Nu
        Nw_s = N * Nd

        self.Nx_s = Nx_s
        self.Nu_s = Nu_s
        self.Nw_s = Nw_s

        self.stack_system()

    def stack_system(self):
        '''
        Stack system matrix for N prediction horizon

        x_next = A x_k + B u_k + D w_k
        y_k = D x_k + E w_k

        '''
        Nx = self.Nx  # Dimension of state
        Nu = self.Nu  # Dimension of input
        Nd = self.Nd  # Dimension of disturbance

        Nx_s = self.Nx_s
        Nu_s = self.Nu_s
        Nw_s = self.Nw_s

        N = self.N

        Ax = np.zeros([Nx_s, Nx])
        Bx = np.zeros([Nx_s, Nu_s])
        Dx = np.zeros([Nx_s, Nw_s])
        Du = np.zeros([Nu_s, Nw_s])
        A_ext = np.zeros([Nx_s, Nx])

        A = self.A
        B = self.B
        D = self.D


        K = self.K
        A_cl = A + B @ K

        # Ax
        for i in range(N + 1):
            Ax[i * Nx:(i + 1) * Nx, :] = matrix_power(A, i)
        # Bx
        for i in range(N):
            mat_temp = B
            for j in range(i + 1):
                Bx[(i + 1) * Nx: (i + 2) * Nx, (i - j) * Nu: (i - j + 1) * Nu] = mat_temp  # could be problematic
                mat_temp = A @ mat_temp
        # Dx
        for i in range(N):
            mat_temp = D
            for j in range(i + 1):
                Dx[(i + 1) * Nx: (i + 2) * Nx, (i - j) * Nd: (i - j + 1) * Nd] = mat_temp
                mat_temp = A_cl @ mat_temp
        # Du
        for i in range(N-1):
            mat_temp = D
            for j in range(i + 1):
                Du[(i + 1) * Nu: (i + 2) * Nu, (i - j) * Nd: (i - j + 1) * Nd] = K @ mat_temp
                mat_temp = A_cl @ mat_temp

        self.Ax = Ax
        # print(Ax)
        self.Bx = Bx
        self.Dx = Dx
        self.Du = Du



class Opt_problem(Stack_model):
    def __init__(self, sys_fn, delta_t, N, x_init, D, F, f, G, g, F_t, f_t, H, h, Q, Qf, R, K, cont_time=True, nonlinear=True,
                 u_0=None, xr=None, ur=None, collect=False,  est=False, sin_const=1, N_sample=1, epsilon=1,
                 W_sample_matrix = None):
        super().__init__(sys_fn, delta_t, N, x_init, D, F, f, G, g, F_t, f_t, K, cont_time, nonlinear, u_0, xr, ur)

        N = self.N
        Nx = self.Nx  # Dimension of state
        Nu = self.Nu  # Dimension of input
        Nd = self.Nd  # Dimension of disturbance

        self.xr = xr
        self.ur = ur

        self.epsilon = epsilon

        self.x_init = x_init

        self.Q = Q
        self.Qf = Qf
        self.R = R

        self.F = F
        self.f = f
        self.G = G
        self.g = g
        self.F_t = F_t
        self.f_t = f_t

        self.NG = np.shape(G)[0]
        self.NF = np.shape(F)[0]

        self.H = H
        self.h = h
        self.K = K

        self.A_cl = self.A + self.B @ K

        if collect == False:
            self.N_sample = N_sample
            W_sample_matrix = self.gene_disturbance(N_sample, sin_const)
            self.W_sample_matrix = W_sample_matrix
        elif collect == True and W_sample_matrix is not None:
            self.W_sample_matrix = W_sample_matrix
            # print(W_sample_matrix.shape)
            self.N_sample = np.shape(W_sample_matrix)[1]
        else:
            print("No sample available")

        self.define_loss_func()
        self.define_constraint()
        loss_func = self.loss_func
        constraint = self.constraint

        self.obj = cp.Minimize(loss_func)
        self.prob = cp.Problem(self.obj, constraint)
        # print(self.prob)

    def gene_disturbance(self, N_sample, sin_const):
        # Generate data: const * sinx

        N = self.N
        Nd = self.Nd
        w_sample = []
        for i in range(N_sample):
            w_temp = sin_const * np.sin(np.random.randn(N * Nd))
            w_sample += [w_temp]
        W_sample_matrix = np.array(w_sample).T
        return W_sample_matrix

    def define_loss_func(self):
        xr = self.xr
        ur = self.ur

        N = self.N

        Nx = self.Nx  # Dimension of state
        Nu = self.Nu  # Dimension of input
        Nd = self.Nd  # Dimension of disturbance
        Nc = self.Nc
        Nc_t = self.Nc_t

        Nx_s = self.Nx_s
        Nu_s = self.Nu_s
        Nw_s = self.Nw_s

        Q = self.Q
        R = self.R
        Qf = self.Qf

        Z = cp.Variable([Nx_s, 1])
        V = cp.Variable([Nu_s, 1])

        self.Z = Z
        self.V = V

        loss = 0
        for i in range(N):
            z_temp = Z[i * Nx: (i + 1) * Nx, :]
            v_temp = V[i * Nu : (i + 1) * Nu, :]

            loss += cp.quad_form(z_temp, Q)
            loss += cp.quad_form(v_temp, R)
            loss += - 2 * xr.T @ Q @ z_temp
            loss += - 2 * ur.T @ R @ v_temp

        z_temp = Z[N * Nx: (N + 1) * Nx, :]
        loss += cp.quad_form(z_temp, Qf)
        loss += - 2 * xr.T @ Qf @ z_temp

        self.loss_func = loss
    def define_constraint(self):
        N = self.N

        Nx = self.Nx  # Dimension of state
        Nu = self.Nu  # Dimension of input
        Nd = self.Nd  # Dimension of disturbance
        Nc = self.Nc
        Nc_t = self.Nc_t

        Nx_s = self.Nx_s
        Nu_s = self.Nu_s

        Q = self.Q
        R = self.R
        NQ = np.shape(Q)[0]
        NR = np.shape(R)[0]


        G = self.G
        g = self.g
        F = self.F
        F_t = self.F_t
        f = self.f
        f_t = self.f_t

        NF = np.shape(F)[0]
        NG = np.shape(G)[0]

        N_Ft = np.shape(F_t)[0]

        Q_stack = np.zeros([Nx_s - Nx, Nx_s - Nx])
        for i in range(N):
            Q_stack[i * Nx : (i+1) * Nx, i * Nx : (i+1) * Nx] = Q

        R_stack = np.zeros([Nu_s, Nu_s])
        for i in range(N):
            R_stack[i * Nu : (i+1) * Nu, i * Nu : (i+1) * Nu] = R

        G_stack = np.zeros([NG * N, Nu * N])
        g_stack = np.zeros([NG * N, 1])
        for i in range(N):
            G_stack[i * NG : (i+1) * NG, i * Nu : (i+1) * Nu] = G
            g_stack[i * NG : (i+1) * NG,:] = g


        x_init = self.x_init

        H = self.H
        h = self.h
        N_xi_dual = np.shape(H)[0]

        epsilon = self.epsilon
        N_sample = self.N_sample

        Ax = self.Ax
        Bx = self.Bx
        Dx = self.Dx
        Du = self.Du

        A = self.A
        B = self.B
        delta = self.delta
        K = self.K

        Z = self.Z
        V = self.V

        self.V = V
        self.Z = Z

        constraint = []

        # Initial state
        constraint += [Z[:Nx,:] == x_init]

        # Nominal System evolution
        for i in range(N):
            constraint += [Z[(i+1) * Nx : (i+2) * Nx,:] == A @ Z[i * Nx : (i+1) * Nx,:] + B @ V[i * Nu : (i+1) * Nu, :] + delta]

        # Reformulate input constraints
        N_Gstack = np.shape(G_stack)[0]
        xi_dual1_list = [cp.Variable([N_xi_dual, 1], nonneg=True) for i in range(N_Gstack)]
        for i in range(N_Gstack):
            constraint += [h.T @ xi_dual1_list[i] <= (g_stack - G_stack @ V)[i, 0]]
            constraint += [H.T @ xi_dual1_list[i] == (G_stack @ Du)[[i], :].T]

        # Define state constraints
        W_sample_matrix = self.W_sample_matrix
        N_sk = np.shape(W_sample_matrix)[1]

        lambda_list_2D = []
        s_list_3D = []
        gamma_list_3D = []

        for i in range(1,N):
            gamma_list_3D += [[]]
            s_list_3D += [[]]
            lambda_list_2D += [[]]
            for j in range(NF):
                Ij = np.zeros([NF,1])
                Ij[j] = 1

                gamma_list_3D[i-1] += [[]]
                s_list_3D[i-1] += [[]]

                lambda_var = cp.Variable(nonneg=True)
                lambda_list_2D[i-1] += [lambda_var]
                for l in range(N_sk):
                    gamma_var = cp.Variable((N_xi_dual, 1), nonneg=True)
                    gamma_list_3D[i-1][j] += [gamma_var]

                    s_var = cp.Variable()
                    s_list_3D[i-1][j] += [s_var]
                    xi_k =  W_sample_matrix[:,[l]]
                    constraint += [Ij.T @ (F @ Z[i * Nx: (i + 1) * Nx, :] - f + F @ Dx[i * Nx: (i + 1) * Nx,:] @ xi_k) + gamma_var.T @ (h - H @ xi_k) <= s_var]
                    constraint += [cp.norm(H.T @ gamma_var - (Ij.T @ F @ Dx[i *Nx : (i+1)*Nx, :]).T, p=np.inf) <= lambda_var]
                constraint += [lambda_var * epsilon + 1/N_sk *  cp.sum(s_list_3D[i-1][j]) <= 0]

        # Reformulate terminal constraint
        gamma_N_list_2D = []
        s_N_list_2D = []
        lambda_N_list_1D = []
        for j in range(N_Ft):
            Ij = np.zeros([N_Ft,1])
            Ij[j] = 1

            gamma_N_list_2D += [[]]
            s_N_list_2D += [[]]

            lambda_var = cp.Variable(nonneg=True)
            lambda_N_list_1D += [lambda_var]
            for l in range(N_sk):
                gamma_var = cp.Variable((N_xi_dual, 1), nonneg=True)
                gamma_N_list_2D[j] += [gamma_var]

                s_var = cp.Variable()
                s_N_list_2D[j] += [s_var]
                xi_k =  W_sample_matrix[:,[l]]
                constraint += [Ij.T @ (F_t @ Z[N * Nx : (N+1) * Nx,:] - f_t + F_t @ Dx[N *Nx : (N+1)*Nx, :] @ xi_k )+ gamma_var.T @ (h - H @ xi_k) <= s_var ]
                constraint += [cp.norm(H.T @ gamma_var - Ij.T @ F_t @ Dx[N *Nx : (N+1)*Nx, :], p=np.inf) <= lambda_var]
            constraint += [lambda_var * epsilon + 1/N_sk *  cp.sum(s_N_list_2D[j])<= 0]


        self.constraint = constraint


class Simulation():
    '''
    @Arg

        mode: "collect" data and incorporate the collected data into constraint
              "gene" data at each time instant and use fixed number of data to solve opt problem

    '''


    def __init__(self, sys_fn, delta_t, N, x_init, D, F, f, G, g, F_t, f_t, H, h, Q, Qf, R, K, cont_time=True, nonlinear=True, u_0=None,
                 xr=None, ur=None, collect=False, est=False, sin_const=1, N_sample=5, epsilon=1, N_sim=80, data_set=None, N_sample_max=None):

        self.sys_fn = sys_fn
        self.delta_t = delta_t
        self.N = N
        self.x_init = x_init
        self.D = D
        # For constraints: Fx + Gu <= 1
        self.F = F
        self.f = f
        self.G = G
        self.g = g
        self.H = H
        self.h = h
        self.F_t = F_t
        self.f_t = f_t

        self.K = K

        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.cont_time = cont_time
        self.nonlinear = nonlinear
        self.u_0 = u_0
        self.xr = xr
        self.ur = ur
        self.collect = collect
        self.est = est
        self.sin_const = sin_const
        self.N_sample = N_sample
        self.epsilon = epsilon
        self.N_sim = N_sim
        self.data_set = data_set
        self.N_sample_max = N_sample_max

        self.x_sim, self.u_sim = self.simulation_gene(x_init, N_sim)

        # TODO: finish coding collect

    def simulation_gene(self, x_init, N_sim):

        sys_fn = self.sys_fn
        delta_t = self.delta_t
        N = self.N
        x_init = self.x_init
        D = self.D
        F = self.F
        f = self.f
        G = self.G
        g = self.g
        H = self.H
        h = self.h
        F_t = self.F_t
        f_t = self.f_t
        Q = self.Q
        Qf = self.Qf
        R = self.R
        K = self.K
        cont_time = self.cont_time
        nonlinear = self.nonlinear
        u_0 = self.u_0
        xr = self.xr
        ur = self.ur
        collect = self.collect
        est = self.est
        sin_const = self.sin_const
        N_sample = self.N_sample
        epsilon = self.epsilon
        N_sim = self.N_sim
        data_set = self.data_set
        N_sample_max = self.N_sample_max

        ode = sys_fn

        Nd = np.shape(D)[1]

        t0 = 0
        xk = x_init
        uk = u_0
        t = t0

        error_flag = False


        x_list = []
        x_list += xk.flatten().tolist()
        u_list = []
        obj_list = []

        xk = x_init

        W_sample_matrix = self.gene_disturbance(N, Nd, N_sample, sin_const)

        for i in range(N_sim):
            opt_problem = Opt_problem(sys_fn, delta_t, N, xk, D, F, f, G, g, F_t, f_t, H, h, Q, Qf, R, K,
                                      cont_time=cont_time, nonlinear=nonlinear, u_0=uk, xr=xr, ur=ur, collect=collect, est=est,
                                      sin_const=sin_const, N_sample=N_sample, epsilon=epsilon, W_sample_matrix = W_sample_matrix)
            Nu = opt_problem.Nu
            prob = opt_problem.prob
            prob.solve(solver=cp.GUROBI)
            wk = sin_const * np.random.uniform(-0.8, 1, Nd).reshape(Nd, 1)
            if nonlinear is False:
                uk = opt_problem.V.value[0:Nu,0].reshape(Nu,1)
            else:
                uk = opt_problem.V.value[0:Nu, 0].reshape(Nu, 1)
            u_list += uk.flatten().tolist()
            x_kp1 = opt_problem.dt_sys_fn(xk, uk).full()
            xk = x_kp1
            xk += D @ wk
            x_list += xk.flatten().tolist()
            obj_list += [prob.value]

        if error_flag is True:
            x_list = None
            u_list = None
        self.Nx = opt_problem.Nx
        self.Nu = opt_problem.Nu

        self.obj_list = obj_list
        # print(obj_list)
        return x_list, u_list

    def gene_disturbance(self, N, d, N_sample, sin_const):
        # Generate data: const * sinx

        w_sample = []
        for i in range(N_sample):
            w_temp = sin_const * np.sin(np.random.randn(N * d))
            w_sample += [w_temp]
        W_sample_matrix = np.array(w_sample).T

        return W_sample_matrix

    def plot_state(self):
        delta_t = self.delta_t
        Nx = self.Nx
        Nu = self.Nu

        x_traj = self.x_sim
        u_traj = self.u_sim

        Nt = np.shape(x_traj[::Nx])[0]
        t_plot = [delta_t * i for i in range(Nt)]

        plt.figure(1, figsize=(10, 20))
        plt.clf()
        # Print states
        for i in range(Nx):
            plt.subplot(Nx + Nu, 1, i + 1)
            plt.grid()
            x_traj_temp = x_traj[i::Nx]
            plt.plot(t_plot, x_traj_temp)
            plt.ylabel('x' + str(i + 1))

            # Print reference
            ref_plot_temp = [self.xr[i]] * Nt
            plt.plot(t_plot, ref_plot_temp, color="k")

        for i in range(Nu):
            plt.subplot(Nx + Nu, 1, i + 1 + Nx)
            plt.grid()
            u_traj_temp = u_traj[i::Nu]
            plt.plot(t_plot[:-1], u_traj_temp)
            plt.ylabel('u' + str(i + 1))

            # Print reference
            ref_plot_temp = [self.ur[i]] * Nt
            plt.plot(t_plot, ref_plot_temp, color="k")

            # Print constraint
            # if i == self.i_th_state:
            #     v_constr = self.i_state_ub
            #     constr_plot_temp = [v_constr] * Nt
            #     plt.plot(t_plot, constr_plot_temp, color="r")

        plt.xlabel('t')
        plt.show()