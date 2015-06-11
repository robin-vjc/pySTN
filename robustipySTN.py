"""
Implementation of a scheduling system based on the STN model.
"""

import numpy as np
import cvxpy as cvx

class NominalSTN(object):
    def __init__(self):
        # Definition of the STN (order in the list matters!)
        self.units = ['Heater', 'Reactor 1', 'Reactor 2', 'Column']
        self.tasks = ['Heating', 'Rea. 1', 'Rea. 2', 'Rea. 3', 'Separation']
        self.states = ['Feed A', 'Feed B', 'Feed C', 'Hot A', 'Intermediate AB',
                       'Intermediate BC', 'Impure E', 'Product 1', 'Product 2']
        self.horizon = 11  # horizon in number of of steps

        # Aliases
        self.I = self.units
        self.J = self.tasks
        self.S = self.states
        self.T = self.horizon

        # Units capability (what tasks can be processed on which unit)
        # row = unit, column = task
        self.J_i = np.array([[1, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0],
                             [0, 1, 1, 1, 0],
                             [0, 0, 0, 0, 1]])

        # Recipes and timing
        # fractions of input state s (column) required to execute task j (row)
        self.rho_in = np.array([[1,   0,   0,   0,   0,   0,   0,   0,   0],
                                [0,   0.5, 0.5, 0,   0,   0,   0,   0,   0],
                                [0,   0,   0,   0.4, 0,   0.6, 0,   0,   0],
                                [0,   0,   0.2, 0,   0.8, 0,   0,   0,   0],
                                [0,   0,   0,   0,   0,   0,   1,   0,   0]])
        # fractions of output state s (column) produced by task j (row)
        self.rho_out = np.array([[0,   0,   0,   1,   0,   0,   0,   0,   0],
                                 [0,   0,   0,   0,   0,   1,   0,   0,   0],
                                 [0,   0,   0,   0,   0.6, 0,   0,   0.4, 0],
                                 [0,   0,   0,   0,   0,   0,   1,   0,   0],
                                 [0,   0,   0,   0,   0.1, 0,   0,   0,   0.9]])
        # time (in # of steps) required to produce output state s (column) from task j (row)
        self.P = np.array([[0,   0,   0,   1,   0,   0,   0,   0,   0],
                           [0,   0,   0,   0,   0,   2,   0,   0,   0],
                           [0,   0,   0,   0,   2,   0,   0,   2,   0],
                           [0,   0,   0,   0,   0,   0,   1,   0,   0],
                           [0,   0,   0,   0,   2,   0,   0,   0,   1]])
        # total execution time of task j (row-wise max of P)
        self.P_j = np.amax(self.P, axis=1)

        # Capacities
        # max capacity of unit i (row) to process task j (column), in e.g. kg
        self.V_max = np.array([[100, 100, 100, 100, 100],
                               [80,  80,  80,  80,  80],
                               [50,  50,  50,  50,  50],
                               [200, 200, 200, 200, 200]])
        self.V_min = np.array([[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]])
        # storage capacity for state s
        self.C_max = np.array([np.infty, np.infty, np.infty, 100, 200, 150, 100, np.infty, np.infty])

        # objective to maximize (revenue from the states)
        c = np.array([0, 0, 0, -1, -1, -1, -1, 10, 10])

        # Optimization problem structure
        self.problem = 0


    def construct_nominal_model(self):
        """ constructs the standard model for the STN
        :return: list(A_eq, b_eq, A_ineq, b_ineq, c, int_index) """
        I = self.units.__len__()
        J = self.tasks.__len__()
        S = self.states.__len__()
        T = self.T

        # TODO BIG-M for allocation
        BIG_M = 4

        # Variables
        # ---------
        x_ijt = {}  # allocation variable (bool)
        y_ijt = {}  # batch size [kg]
        y_st = {}   # state quantity [kg]
        for t in range(self.T):
            for i in range(I):
                for j in range(J):
                    # x_ijt[(i,j,t)] = cvx.Bool()
                    x_ijt[(i,j,t)] = cvx.Variable()
                    y_ijt[(i,j,t)] = cvx.Variable()
            # TODO maybe not used:
            for s in range(S):
                y_st[(s,t)] = cvx.Variable()

        # Constraints
        # -----------
        # Unit allocation
        cstr_allocation = []
        for i in range(I):
            for t in range(T):
                cstr_allocation.append( sum( [x_ijt[(i,j,t)] for j in range(J)] ) <= 1 )
            for j in range(J):
                for t in range(T-self.P_j[j]):
                    cstr_allocation.append(
                        sum ( sum( [[x_ijt[(i,jj,tt)] for jj in range(J)] for tt in range(2)], [] ) )
                        <=
                        self.P_j[j]*BIG_M*(1 - x_ijt[(i,j,t)]) +1
                    )

        # Box constraints (for testing)
        cstr_box = []
        for i in range(I):
            for j in range(J):
                for t in range(T):
                    cstr_box.append( [0 <= x_ijt[(i,j,t)],
                                      x_ijt[(i,j,t)] <= 1,
                                      0 <= y_ijt[(i,j,t)],
                                      y_ijt[(i,j,t)] <= 1] )

        # Objective
        # ---------
        objective = cvx.Maximize(x_ijt[(1,1,1)])

        self.problem = cvx.Problem(objective, cstr_box)


def main():
    model = NominalSTN()
    model.construct_nominal_model()


if __name__ == '__main__':
    main()
