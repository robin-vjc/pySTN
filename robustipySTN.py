"""
Implementation of a scheduling system based on the STN model.
"""

import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

# We pack everything in a class, mainly to avoid having to implement functions (now the
# class methods) with very long signatures (can just pass in self, which contains all
# the data).
class STN(object):
    def __init__(self):
        # Definition of the STN (order in the list matters!)
        self.units = ['Heater', 'Reactor 1', 'Reactor 2', 'Column']
        self.tasks = ['Heat', 'Rea. 1', 'Rea. 2', 'Rea. 3', 'Sep.']
        self.states = ['Feed A', 'Feed B', 'Feed C', 'Hot A', 'Intermediate AB',
                       'Intermediate BC', 'Impure E', 'Product 1', 'Product 2']
        self.input_states = [0, 1, 2]  # indices of the input states
        self.horizon = 11  # horizon in number of of steps

        # Aliases
        self.I = self.units.__len__()
        self.J = self.tasks.__len__()
        self.S = self.states.__len__()
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
        self.C_min = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

        # objective to maximize (revenue from the states)
        self.c = np.array([0, 0, 0, -1, -1, -1, -1, 10, 10])

        # Optimization problem structure
        self.problem = 0

        # Variables
        # ---------
        self.x_ijt = {}  # allocation variable (bool)
        self.y_ijt = {}  # batch size [kg]
        self.y_s = {}   # state quantity [kg]
        for t in range(self.T):
            for i in range(self.I):
                for j in range(self.J):
                    self.x_ijt[i,j,t] = cvx.Bool()
                    self.y_ijt[i,j,t] = cvx.Variable()
        # state equations are eliminated to allow for the robust counterpart. We only store
        # the iniatial state y_s(t=0), which we name y_s
        for s in range(self.S):
            self.y_s[s] = cvx.Variable()
        # auxiliary expressions used in the state equations
        self.y_st_inflow = {}
        self.y_st_outflow = {}
        # local variables to store results
        self.X_ijt = np.zeros((self.I,self.J,self.T))
        self.Y_ijt = np.zeros((self.I,self.J,self.T))
        self.Y_st = np.zeros((self.S,self.T))
        self.Y_st_inflow = np.zeros((self.S,self.T))
        self.Y_st_outflow = np.zeros((self.S,self.T))

    def construct_allocation_constraint(self):
        ''' construct the allocation constraints:
            1) each unit i is processing at most one task j at each t
            2) units can only perform tasks that are compatible with self.J_i
        :return: list of constraints
        '''
        I = self.units.__len__()
        J = self.tasks.__len__()
        S = self.states.__len__()
        T = self.T
        # TODO BIG-M for allocation
        BIG_M = 40

        constraint_allocation = []
        # 1) each unit i is processing at most one task j at each t
        for i in range(I):
            for t in range(T):
                constraint_allocation.append( sum( [self.x_ijt[(i,j,t)] for j in range(J)] ) <= 1 )
            for j in range(J):
                for t in range(T-self.P_j[j]+1):
                    constraint_allocation.append(
                        sum ( sum( [[self.x_ijt[(i,jj,tt)] for jj in range(J)] for tt in range(t,t+self.P_j[j])], [] ) )
                        <=
                        self.P_j[j]*BIG_M*(1 - self.x_ijt[(i,j,t)]) + 1
                    )
        # 2) units can only perform tasks that are compatible with self.J_i, and tasks should be started
        # early enough such that they are completed within T, i.e., no tasks can start after T-P_j
        for i in range(I):
            for j in range(J):
                for t in range(T):
                    constraint_allocation.append( self.x_ijt[i,j,t]*(1-self.J_i[i,j]) == 0 )
                    if t >= T-self.P_j[j]:
                        constraint_allocation.append( self.x_ijt[i,j,t] == 0 )

        return constraint_allocation

    def construct_box_constraint(self):
        I = self.units.__len__()
        J = self.tasks.__len__()
        T = self.T
        constraint_box = []
        for i in range(I):
            for j in range(J):
                for t in range(T):
                    constraint_box.append(self.x_ijt[i,j,t] >= 0)
                    constraint_box.append(self.x_ijt[i,j,t] <= 1)
                    constraint_box.append(self.y_ijt[i,j,t] >= 0)
                    constraint_box.append(self.y_ijt[i,j,t] <= 1)

        return constraint_box

    def construct_units_capacity_constraint(self):
        constraint_capacity = []
        for i in range(self.I):
            for j in range(self.J):
                for t in range(self.T):
                    constraint_capacity.append( self.x_ijt[i,j,t]*self.V_min[i,j] <= self.y_ijt[i,j,t] )
                    constraint_capacity.append( self.y_ijt[i,j,t] <= self.x_ijt[i,j,t]*self.V_max[i,j] )

        return constraint_capacity

    def construct_state_equations_and_storage_constraint(self):
        # implementation of state equations, and states capacities (storages)
        constraint_state_eq = []
        # 1) every intermediate / output state starts with a quantity of 0 kg
        for s in range(self.S):
            if s not in self.input_states:
                constraint_state_eq.append( self.y_s[s] == 0 )

        # 2) state equations and storage capacities
        for t in range(self.T):
            for s in range(self.S):
                self.y_st_inflow[s,t] = cvx.Constant(0)
                self.y_st_outflow[s,t] = cvx.Constant(0)

                for i in range(self.I):
                    for j in range(self.J):
                        if self.J_i[i,j]:
                            # set inflows
                            if (t-self.P[j,s] >= 0):
                                self.y_st_inflow[s,t] += self.rho_out[j,s]*self.y_ijt[i,j,t-self.P[j,s]]
                            # set outflows
                            self.y_st_outflow[s,t] += self.rho_in[j,s]*self.y_ijt[i,j,t]

                constraint_state_eq.append( self.C_min[s] <= self.y_s[s]
                                            + sum( [self.y_st_inflow[(s,tt)] for tt in range(t+1)] )
                                            - sum( [self.y_st_outflow[(s,tt)] for tt in range(t+1)] ))
                constraint_state_eq.append( self.C_max[s] >= self.y_s[s]
                                            + sum( [self.y_st_inflow[(s,tt)] for tt in range(t+1)] )
                                            - sum( [self.y_st_outflow[(s,tt)] for tt in range(t+1)] ))

        return constraint_state_eq

    def construct_konidili_solution_enforce(self):
        """ the nominal model with the data of Kondili's paper has several optimizers. The following
        constraints force the exact same solution as in the paper (the objective is unaffected) """
        constraint_kondili = []
        constraint_kondili.append( [self.y_ijt[0,0,1] == 52] )
        constraint_kondili.append( [self.y_ijt[1,1,0] == 80] )
        return constraint_kondili

    def construct_objective(self):
        return sum( [self.c[s]*( sum([self.y_st_inflow[s,t] for t in range(self.T)])
                                 - sum([self.y_st_outflow[s,t] for t in range(self.T)]))
                     for s in range(self.S)] )

    def unpack_results(self):
        ''' once model is solved, transform the solution dictionaries into np.arrays for
        easier inspection/plotting '''
        for t in range(self.T):
            for j in range(self.J):
                for i in range(self.I):
                    self.X_ijt[i,j,t] = self.x_ijt[i,j,t].value
                    self.Y_ijt[i,j,t] = self.y_ijt[i,j,t].value
        for t in range(self.T):
            for s in range(self.S):
                self.Y_st_inflow[s,t] = self.y_st_inflow[s,t].value
                self.Y_st_outflow[s,t] = self.y_st_outflow[s,t].value
                self.Y_st[s,t] = self.y_s[s].value + (sum([self.y_st_inflow[s,tt].value for tt in range(t+1)])
                                                      - sum([self.y_st_outflow[s,tt].value for tt in range(t+1)]))

    def plot_schedule(self):
        ## TODO warning if problem is not solved
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        for i, unit in enumerate(self.Y_ijt):
            for j, task in enumerate(unit):
                for t, time in enumerate(task):
                    if t < self.T:
                        slot_x = np.array([t, t+1])
                        slot_y = np.array([i, i])
                        slot_y2 = slot_y+1
                        slot_y3 = np.array([i, i+1])
                        # don't plot blocks where Y_ijt is just some epsilon, residual of the optimization
                        if self.Y_ijt[i,j,t] >= 1:
                            plt.fill_between(slot_x, slot_y, y2=slot_y2, color='red')
                            plt.text(np.mean(slot_x), np.mean(slot_y3), "{0}\n{1}".format(self.Y_ijt[i,j,t], self.tasks[j]),
                                     horizontalalignment='center', verticalalignment='center' )

        # plt.ylim(self.I, 0)
        plt.yticks(range(self.I), self.units, verticalalignment='center')
        plt.minorticks_on()
        plt.show()

    def construct_nominal_model(self):
        """ constructs the standard model for the STN """
        # Constraints
        # -----------
        constraints = []
        constraints.append(self.construct_allocation_constraint())
        constraints.append(self.construct_units_capacity_constraint())
        constraints.append(self.construct_state_equations_and_storage_constraint())
        # can force Kondili's solution with
        # constraints.append(self.construct_konidili_solution_enforce())

        constraints = sum(constraints, [])

        # Objective
        # ---------
        objective = cvx.Maximize(self.construct_objective())

        self.problem = cvx.Problem(objective, constraints)

    def solve(self):
        self.construct_nominal_model()
        self.problem.solve(verbose=True, solver='GUROBI')
        self.unpack_results()

if __name__ == '__main__':
    model = STN()
    model.solve()
    model.plot_schedule()

