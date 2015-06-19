"""
Implementation of a scheduling system based on the STN model. The data of the problem is
in the __init__() method, and can be changed. Additional constraints can be included as functions
that return cvx.Constraints, and added in the STN.construst_nominal_model() method.

We pack everything in a class to avoid having to implement functions (now the class methods)
with exceedingly long signatures.
"""

import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

class STN(object):
    def __init__(self):
        # Data
        # ----
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

        # Optimization problem structure (cvx.Problem type)
        self.model = 0

        # Optimization Variables
        # ----------------------
        self.x_ijt = {}  # allocation variable (bool)
        self.y_ijt = {}  # batch size [kg]
        self.y_s = {}   # state quantity [kg]
        for i in range(self.I):
            for j in range(self.J):
                for t in range(self.T):
                    self.x_ijt[i,j,t] = cvx.Bool()
        # we separate creation of x_ijt and y_ijt in two loops, so that the optimization variables
        # in the standard model do not get mixed up
        for i in range(self.I):
            for j in range(self.J):
                for t in range(self.T):
                    self.y_ijt[i,j,t] = cvx.Variable()
        # state equations are eliminated to allow for the robust counterpart. We only store
        # the initial state y_s(t=0), which we name y_s
        for s in range(self.S):
            self.y_s[s] = cvx.Variable()
        # auxiliary expressions used in the state equations
        self.y_st_inflow = {}
        self.y_st_outflow = {}

        # Attributes
        # ----------
        # to store results
        self.X_ijt = np.zeros((self.I,self.J,self.T))
        self.Y_ijt = np.zeros((self.I,self.J,self.T))
        self.Y_st = np.zeros((self.S,self.T))
        self.Y_st_inflow = np.zeros((self.S,self.T))
        self.Y_st_outflow = np.zeros((self.S,self.T))
        # to store the standard model
        # min   c_x'*x + c_y'*y
        # s.t.  A_eq*x + B_eq*y = b_eq
        #       A_ineq*x + B_ineq*y <= b_ineq
        #       x \in {0,1}
        self.c_x = 0
        self.c_y = 0
        self.A_eq = 0
        self.A_ineq = 0
        self.B_eq = 0
        self.B_ineq = 0
        self.b_eq = 0
        self.b_ineq = 0
        self.bool_ix = 0
        self.cont_ix = 0
        self.m_eq = 0
        self.m_ineq = 0

    def construct_allocation_constraint(self):
        ''' construct the allocation constraints:
            1) each unit i is processing at most one task j at each t
            2) units can only perform tasks that are compatible with self.J_i
        :return: list of cvx.Constraint
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
        """ Construct box constraints on x_ijt and y_ijt; useful for testing with continuous
        variables instead of bools.
        :return: list of cvx.Constraint
        """
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
        """ Ensure maximum and minimum sizes of the batches to be processed are within
        unit constraints.
        :return: list of cvx.Constraint
        """
        constraint_capacity = []
        for i in range(self.I):
            for j in range(self.J):
                for t in range(self.T):
                    constraint_capacity.append( self.x_ijt[i,j,t]*self.V_min[i,j] <= self.y_ijt[i,j,t] )
                    constraint_capacity.append( self.y_ijt[i,j,t] <= self.x_ijt[i,j,t]*self.V_max[i,j] )

        return constraint_capacity

    def construct_state_equations_and_storage_constraint(self):
        """ Implementation of state equations, and states capacities (storages)
        :return: list of cvx.Constraint
        """
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
        """ The nominal model with the data of Kondili's paper has several optimizers. The following
        constraints force the exact same solution as in the paper (the objective is unaffected)
        :return: list of cvx.Constraint
        """
        constraint_kondili = []
        constraint_kondili.append( [self.y_ijt[0,0,1] == 52] )
        constraint_kondili.append( [self.y_ijt[1,1,0] == 80] )
        return constraint_kondili

    def construct_objective(self):
        """ Objective encodes c'*(y_s(t=end)-y_s(t=0)), i.e., value of the final products minus
        cost of the input feeds.
        :return: cvx.Objective
        """
        return - sum( [self.c[s]*( sum([self.y_st_inflow[s,t] for t in range(self.T)])
                                 - sum([self.y_st_outflow[s,t] for t in range(self.T)]))
                     for s in range(self.S)] )

    def construct_nominal_model(self):
        """ Constructs the nominal STN model, and saves it in the class attribute self.model as
        a cvx.Problem type. Constraints can be added/removed here.
        :return: None
        """
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
        objective = cvx.Minimize(self.construct_objective())

        # Model
        # -----
        self.model = cvx.Problem(objective, constraints)
        # Also store the matrices of te standard model:
        data = self.model.get_problem_data('ECOS_BB')
        self.retrieve_standard_model(data)

    def retrieve_standard_model(self, data):
        """ Here we store the problem matrices (A_eq, B_eq, b_eq, etc) with ordered columns.
        :param data: dictionary, as returned by cvx.Problem.get_problem_data('ECOS_BB')
        :return: None
        """
        n = data['c'].shape[0]
        self.bool_ix = data['bool_vars_idx']
        self.cont_ix = list(set(range(n)) - set(data['bool_vars_idx']))
        self.n_x = self.bool_ix.__len__()
        self.n_y = self.cont_ix.__len__()
        range_bool_ix = range(self.n_x)
        range_cont_ix = range(self.n_x, self.n_x+self.n_y)
        self.c_x = data['c'][range_bool_ix]
        self.c_y = data['c'][range_cont_ix]
        self.A_eq = data['A'][:,range_bool_ix]
        self.B_eq = data['A'][:,range_cont_ix]
        self.b_eq = data['b']
        self.A_ineq = data['G'][:,range_bool_ix]
        self.B_ineq = data['G'][:,range_cont_ix]
        self.b_ineq = data['h']
        self.b_ineq = data['h']
        self.m_eq = self.A_eq.shape[0]
        self.m_ineq = self.A_ineq.shape[0]

    def solve(self):
        """ Constructs and solved the nominal STN model. The solution is stored in the np.arrays
        - STN.X_ijt (assignments, bool)
        - STN.Y_ijt (batch sizes, float)
        - STN.Y_st (material quantities, float)
        - STN.Y_st_inflow and Y_st_outflow (material flows, float)
        :return: optimal value (float)
        """
        print 'Constructing nominal model...'
        self.construct_nominal_model()
        print 'Solving...'
        self.model.solve(verbose=True, solver='GUROBI')
        self.unpack_results()
        return self.model.value

    def unpack_results(self):
        """ Once model is solved, transform the solution dictionaries (self.x_ijt, self.y_ijt) into
        np.arrays for easier inspection/plotting. The np.arrays are saved within the instance attributes
        - STN.X_ijt (assignments, bool)
        - STN.Y_ijt (batch sizes, float)
        - STN.Y_st (stored material quantities, float)
        - STN.Y_st_inflow and STN.Y_st_outflow (material flows, float),
        and can be accessed from there once the method is executed.
        :return: None
        """
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
        """ Plot the nominal schedule.
        :return: None
        """
        # TODO you should plot the attained objective, and also (maybe) the states evolution, divided in input/int/output
        color = 'red'
        margin = 0.03  # size of margins around the boxes

        if not self.X_ijt.any():
            print 'Please, solve model first by invoking STN.solve()'
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect(1)
            for i, unit in enumerate(self.Y_ijt):
                unit_axis = np.array([i, i+1])
                for j, task in enumerate(unit):
                    for t, time in enumerate(task):
                        if t < self.T:
                            slot_x = np.array([t+margin, t+self.P_j[j]-margin])
                            slot_y = np.array([i+margin, i+margin])
                            slot_y2 = slot_y+0.90+margin
                            # don't plot blocks where Y_ijt is just some epsilon, residual of the optimization
                            if self.Y_ijt[i,j,t] >= 1:
                                plt.fill_between(slot_x, slot_y, y2=slot_y2, color=color)
                                plt.text(np.mean(slot_x), np.mean(unit_axis), "{0}\n{1}".format(self.Y_ijt[i,j,t], self.tasks[j]),
                                         horizontalalignment='center', verticalalignment='center' )
                plt.text(-0.15, np.mean(unit_axis), "{0}".format(self.units[i]),
                         horizontalalignment='right', verticalalignment='center')

            plt.ylim(self.I, 0)
            plt.xlabel('time [h]')
            plt.yticks(range(self.I), "", y=0.5)
            plt.show()

if __name__ == '__main__':
    model = STN()
    model.solve()
    model.plot_schedule()
