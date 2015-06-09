"""
Implementation of a scheduling system based on the STN model.
"""

import numpy as np
import cxvpy as cvx

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

    def construct_nominal_model(self):
        """ constructs the standard model for the STN
        :return: list(A_eq, b_eq, A_ineq, b_ineq, c, int_index) """
        I_size = self.units.__len__()
        J_size = self.tasks.__len__()
        S_size = self.states.__len__()

        # Variables
        # ---------
        x_ijt = cvx.Variable(I_size, J_size, self.T)  # allocation variable (bool)
        y_ijt = cvx.Variable(I_size, J_size, self.T)  # batchsize
        # TODO maybe not used:
        y_st = cvx.Variable(S_size, self.T)  # state quantity

        # Constraints
        # -----------






def main():
    pass


if __name__ == '__main__':
    main()
