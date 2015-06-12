"""
Implementation of a scheduling system based on the STN model.
"""

import numpy as np
import cvxpy as cvx

# We pack everything in a class, mainly to avoid having to implement functions (now the
# class methods) with very long signatures (can just pass in self, which contains all
# the data).
class NominalSTN(object):
    def __init__(self):
        # Definition of the STN (order in the list matters!)
        self.units = ['Heater', 'Reactor 1', 'Reactor 2', 'Column']
        self.tasks = ['Heating', 'Rea. 1', 'Rea. 2', 'Rea. 3', 'Separation']
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
        c = np.array([0, 0, 0, -1, -1, -1, -1, 10, 10])

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
                    # x_ijt[(i,j,t)] = cvx.Bool()
                    self.x_ijt[i,j,t] = cvx.Variable()
                    self.y_ijt[i,j,t] = cvx.Variable()
        # state equations are eliminated to allow for the robust counterpart. We only store
        # the iniatial state y_s(t=0), which we name y_s
        for s in range(self.S):
            self.y_s[s] = cvx.Variable()
        # auxiliary expressions used in the state equations
        self.y_st_inflow = {}
        self.y_st_outflow = {}

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
        BIG_M = 4

        constraint_allocation = []
        # 1) each unit i is processing at most one task j at each t
        for i in range(I):
            for t in range(T):
                constraint_allocation.append( sum( [self.x_ijt[(i,j,t)] for j in range(J)] ) <= 1 )
            for j in range(J):
                for t in range(T-self.P_j[j]):
                    constraint_allocation.append(
                        sum ( sum( [[self.x_ijt[(i,jj,tt)] for jj in range(J)] for tt in range(2)], [] ) )
                        <=
                        self.P_j[j]*BIG_M*(1 - self.x_ijt[(i,j,t)]) +1
                    )
        # 2) units can only perform tasks that are compatible with self.J_i
        for i in range(I):
            for j in range(J):
                for t in range(T):
                    constraint_allocation.append( self.x_ijt[i,j,t]*(1-self.J_i[i,j]) == 0 )

        # TODO maybe necessary?
        # Every task should be started early enough such that it is completed within T, i.e., no
        # tasks can start after T-P_j
        # for j= 1:J.size
        #     constr_alloc = [constr_alloc, x_ijt(:,j,end-P_j(j)+1:end) == 0];
        # end


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
        for t in range(1,self.T):
            for s in range(self.S):
                self.y_st_inflow[s,t] = 0
                self.y_st_outflow[s,t] = 0


# for t=2:T
#     for s=1:S.size
#         help_stp(s,t) = 0;
#         help_stm(s,t) = 0;
#         for i = 1:I.size
#             for j = 1:J.size
#                 if J_i(i,j)
#                     if (t-P_js(j,s)>=1)
#                         help_stp(s,t) = help_stp(s,t) + rho_out_js(j,s)*y_ijt(i,j,t-P_js(j,s));
#                     end
#                     help_stm(s,t) = help_stm(s,t) - rho_in_js(j,s) * y_ijt(i,j,t);
#                 end
#             end
#         end
#
#         constr_state = [constr_state, 0<= y_st(s,1) + sum(help_stp(s,2:t)) + sum(help_stm(s,2:t)) <= capty_s(s)];
#     end
# end



    def construct_nominal_model(self):
        """ constructs the standard model for the STN
        :return: list(A_eq, b_eq, A_ineq, b_ineq, c, int_index) """

        # Constraints
        # -----------
        constraints = []
        # Unit allocation
        constraints.append(self.construct_allocation_constraint())
        # Box constraints (for testing with continuous variables)
        constraints.append(self.construct_box_constraint())
        # Unit capacity
        constraints.append(self.construct_units_capacity_constraint())

        constraints = sum(constraints, [])

        # Objective
        # ---------
        objective = cvx.Maximize(self.x_ijt[(1,1,1)])

        self.problem = cvx.Problem(objective, constraints)


def main():
    model = NominalSTN()
    model.construct_nominal_model()


if __name__ == '__main__':
    main()
