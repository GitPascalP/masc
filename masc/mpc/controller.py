""" implementations for mpc and rl controller/ agents """

import os
import numpy as np
import scipy
from scipy.optimize import minimize, LinearConstraint
import qpsolvers as qp
import polytope


class MPC:
    """
    Implementation of a basic MPC controller, solving the optimization with
    a simple qp-solver

    TODO:
        - add functionality that tests the feasibility, feasible set from x0
    """

    def __init__(self, A, B, Q, R, S, Mx, wx, Mu, wu, N, F0=None):
        """ """
        self.A, self.B, self.Q, self.R, self.S = A, B, Q, R, S
        self.Mx_init, self.wx_init, self.Mu_init, self.wu_init = Mx, wx, Mu, wu
        self.H, self.G, self.E, self.F, self.d = None, None, None, None, None
        self.n, self.m = self.B.shape
        self.N = N
        # define feasible set
        self.F0 = F0

        self.update(np.zeros(self.B.shape[0]))
        self.training_mode = False

    def update(self, x_target):
        """ update mpc parameters during controle process """
        self.A_N, self.B_N, self.Q_N, self.R_N = None, None, None, None
        self.get_block_matrices()

        # update contraints acording to new target state
        self.wx = self.wx_init - self.Mx_init @ x_target
        self.wu = self.wu_init
        self.Mx_N, self.wx_N, self.Mu_N, self.wu_N = None, None, None, None
        self.get_constraint_matrices()

        if self.H is None:
            self.H = 2 * (self.R_N + self.B_N.T @ self.Q_N @ self.B_N)
            self.G = np.concatenate((self.Mx_N @ self.B_N, self.Mu_N))
            #self.G = self.F0.A
            self.E = np.concatenate(
                (-self.Mx_N @ self.A_N, np.zeros((self.wu_N.shape[0], self.n)))
            )
            self.F = 2 * self.B_N.T @ self.Q_N @ self.A_N
            self.d = np.concatenate((self.wx_N, self.wu_N))

    def control(self, x, reference):
        """ return control action, depending on current state """
        self.update(reference)
        x_k = x - reference
        # find solution from qp-problem
        u_hat = self.optimize(x_k)

        # TODO: maybe add predicted state horizons
        # used control input and resulting state
        try:
            u_k = np.array([u_hat[0]])
        except:
            print(f'x_k: {x_k}, x: {x}, uhat: {u_hat}')
            raise TypeError

        return u_k

    def optimize(self, x):
        """ solve qp optimization problem to find optimal control input """
        self.f = 2 * self.B_N.T @ self.Q_N @ self.A_N @ x
        self.e = np.concatenate((self.wx_N - self.Mx_N @ self.A_N @ x, self.wu_N))
        # input and state trajectory for specified horizon
        u_hat = qp.solve_qp(self.H, self.f, self.G, self.e)

        return u_hat

    def get_block_matrices(self,):
        """ build blockmatrices for the qp-solver """
        for i in np.arange(0, self.N + 1):
            # build colums
            if self.A_N is None:
                self.A_N = np.linalg.matrix_power(self.A, i)
            else:
                self.A_N = np.concatenate(
                    [self.A_N, np.linalg.matrix_power(self.A, i)], axis=0
                )

            B_N_row = None

            for j in range(self.N):
                # build rows
                power = i - j - 1
                if power < 0:
                    if B_N_row is None:
                        B_N_row = np.zeros((self.n, self.m))
                    else:
                        B_N_row = np.concatenate(
                            [B_N_row, np.zeros((self.n, self.m))], axis=-1
                        )
                else:
                    if B_N_row is None:
                        B_N_row = np.linalg.matrix_power(self.A, power) @ self.B
                    else:
                        B_N_row = np.concatenate(
                            [B_N_row, np.linalg.matrix_power(self.A, power) @ self.B],
                            axis=-1,
                        )
            B_N_row = np.squeeze(B_N_row)

            if self.B_N is None:
                self.B_N = B_N_row
            else:
                self.B_N = np.concatenate([self.B_N, B_N_row])

        self.Q_N = scipy.linalg.block_diag(*([self.Q] * self.N))
        self.Q_N = scipy.linalg.block_diag(self.Q_N, self.S)
        self.R_N = scipy.linalg.block_diag(*([self.R] * self.N))

    def get_constraint_matrices(self,):
        """ build block constraint matrices to give to the qp-solver """
        self.Mx_N = scipy.linalg.block_diag(*([self.Mx_init] * (self.N + 1)))
        self.wx_N = np.concatenate([*([self.wx_init] * (self.N + 1))])

        self.Mu_N = scipy.linalg.block_diag(*([self.Mu_init] * self.N))
        self.wu_N = np.concatenate([*([self.wu_init] * self.N)])

    def get_matrices(self,):
        matrices = {
            "system": (self.A_N, self.B_N, self.Q_N, self.R_N),
            "constraints": (self.Mx_N, self.wx_N, self.Mu_N, self.wu_N),
            "qp": (self.H, self.G, self.f, self.e),
            "pqp": (self.H, self.F, self.G, self.E, self.d),
        }
        return matrices


def safe_costs(u, u_rl):
    """ cost function for safeguard """
    cost = np.abs((u - u_rl))**2
    return cost


def safe_costs_derivative(u, u_rl):
    """ derivative of the quadratic cost function """
    derivative = 2 * (u - u_rl)
    return derivative


class Safeguard:
    """ safeguard to ensure that the RL-Agent takes only safe actions """

    def __init__(self, constraints, set_scaler=None, savepath=None):
        """
        args:
            constraints (polytope): polytope describing the feasible set
            set_scaler (float): scaling factor to manually scaler the
                                constraint-polytope
            savepath (str): estimated polytopes are saved to path during
                            learning if not None
        """
        # attributes, that have to be updated during process
        self.constraints_F0 = constraints
        self.guard_constraints = None
        self.set_scaler = set_scaler or 0.05

        if self.constraints_F0 is not None:
            self.update(
                constraints=self.constraints_F0,
                mode='update',
            )

    def scale(self, constraints, scaler):
        """ scale constraint polytope """
        if constraints.A.shape[-1] == 2:
            weights = np.array(
                [1 + self.set_scaler + np.squeeze(scaler[0]),
                 1 + self.set_scaler + np.squeeze(scaler[1])])
            weights = np.squeeze(weights)
        elif constraints.A.shape[-1] == 3:
            weights = np.array(
                [1 + self.set_scaler + np.squeeze(scaler[0]),
                 1 + self.set_scaler + np.squeeze(scaler[1]),
                 1])
            weights = np.squeeze(weights)

        scaled_constraints = polytope.Polytope(
            constraints.A * weights, constraints.b
        )
        return scaled_constraints

    def update(
            self,
            constraints=None,
            constraints2d=None,
            fit_error_margin=np.array([0, 0]),
            mode='adapt',
    ):
        """
        update Safeguard constraints during learning process, by scaling
        the polytope with different coefficients
        """

        if mode == 'adapt':
            self.guard_constraints = self.scale(
                self.constraints_F0, scaler=fit_error_margin
            )
        elif mode == 'update':
            # set new feasible set given in arguments
            self.guard_constraints = self.scale(
                constraints, scaler=fit_error_margin
            )

    def guide(self, action, state):
        """ guide the action to keep constraints and solve QP if necessary """
        state = np.atleast_1d(np.squeeze(state))
        action = np.atleast_1d(np.squeeze(action))
        n, m = state.shape[0], action.shape[0]
        state_act = np.concatenate([state, action])

        if self.check_constraints(state_act):
            # rl action is safeguard is inactive
            sg_active = False
            u_safe = action
        else:
            # rl actions will violate constraints and guard is actuated
            sg_active = True

            # construct constraint for scipy solver
            Fg, Fe = self.guard_constraints.A, self.guard_constraints.b
            Fg_u = Fg[:, -1]
            Fe_u = Fe - (Fg[:, :n] @ state)
            Fe_u_l = -np.ones(Fe_u.shape) #* (-np.inf)
            Fg_u = Fg_u.reshape(1, -1).T

            ineq_cons = {'type': 'ineq',
                         'fun': lambda action: (Fe - (
                                     Fg[:, :n] @ state)) - Fg_u @ action,
                         }
            lin_constraints = LinearConstraint(A=Fg_u, lb=Fe_u_l, ub=Fe_u)
            # solve constrained opt-problem to find safe action
            solution = scipy.optimize.minimize(
                safe_costs,
                x0=np.array([0]),
                args=(action),
                constraints=lin_constraints,
                method='COBYLA',
            )

            u_safe = solution.x
            if not solution.success:
                # clip action to action limits (-1, 1)
                u_safe = np.clip(u_safe, -1, 1)

        return u_safe, sg_active

    def check_constraints(self, state_action):
        """ check if state-action vector lies within feasible set """
        return state_action in self.guard_constraints
