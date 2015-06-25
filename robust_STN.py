__author__ = 'vujanicr'
# when constructing the uncertainty vectors, it is important to know the order of the variables
# in the nominal problem in standard form; this is established in the __init__() method of the
# STN class, when filling the x_ijt, y_ijt and y_st dictionaries with cvx.Variable objects.

from STN import *
import scipy
import copy

# TODO is there a reddit on PSE? chem E?

class robust_STN(object):
    """ robust_STN uses the nominal STN model, and provides
    - example methods to construct appropriate uncertainty sets,
    - a method to build the Robust Counterpart (RC)
    - a method to solve the RC
    - plotting
    """
    def __init__(self, stn=STN()):
        self.stn = stn
        self.m = self.stn.m_eq + self.stn.m_ineq
        # uncertainty sets are stored as matrices, within a list of length n_x (number of bools
        # in the problem).
        self.W = []

        # Optimization Variables
        # ----------------------
        self.x = cvx.Bool(self.stn.n_x, 1)
        self.v = cvx.Variable(self.stn.n_y, 1)
        self.Y = cvx.Variable(self.stn.n_y, self.stn.n_x)  # recourse matrix
        # for the auxiliary vars Phi and Psi we work with dictionaries to have sparse representations
        self.Phi = {}
        self.Psi = {}
        # TODO BIG-M for Psi_bar
        self.Psi_bar = 100000*np.ones((self.m,1))

        # Attributes
        # ----------
        self.X_ijt = np.zeros((self.stn.I,self.stn.J,self.stn.T))
        self.Y_ijt = np.zeros((self.stn.I,self.stn.J,self.stn.T))
        self.Y_recourse = np.zeros((self.stn.n_y, self.stn.n_x))
        self.Y_ijt_after_event = np.zeros((self.stn.I,self.stn.J,self.stn.T))

    def x_ijt_index_to_std(self, x_ijt_ix):
        """ transforms the [i,j,t] index of the variable x_ijt in the index of the optimization variable
         x in the standard model.
        :param x_ijt_ix: [i,j,t] as a list containing the index of x_ijt
        :return: std_ix, the index of the x_ijt variable in vector of the model in standard form
        """
        i = x_ijt_ix[0]
        j = x_ijt_ix[1]
        t = x_ijt_ix[2]
        return i*(self.stn.J*self.stn.T) + j*self.stn.T + t

    def std_index_to_x_ijt(self, std_ix):
        """ transforms a given coordinate in the standard model into the appropriate [i,j,t] coordinate
        :param std_ix: the position within the variable x in the standard model
        :return: [i,j,t] as a list (ints)
        """
        i = int(std_ix/(self.stn.J*self.stn.T))
        remainder = std_ix%(self.stn.J*self.stn.T)
        j = int(remainder/(self.stn.T))
        t = int(remainder%(self.stn.T))
        return [i,j,t]

    def solve(self, solver='GUROBI'):
        if self.W == []:
            print 'Uncertainty sets robust_STN.W are empty. Build them first.'
            print '(call, e.g., robust_STN.W = robust_STN.build_uncertainty_set_for_unit_delay()).'
        else:
            print 'Constructing robust model...'
            self.build_robust_counterpart()
            print 'Solving...'
            self.robust_model.solve(verbose=True, solver=solver)
            self.unpack_results()

    def unpack_results(self):
        for k in range(self.x.size[0]):
            i,j,t = self.std_index_to_x_ijt(k)
            self.X_ijt[i,j,t] = copy.deepcopy(self.x.value[k])
            self.Y_ijt[i,j,t] = copy.deepcopy(self.v.value[k])
        self.Y_recourse = copy.deepcopy(self.Y.value)

    def plot_schedule(self, color='orange', style='ggplot'):
        """ Plot the robust schedule.
        :return: None
        """
        plot_stn_schedule(self.stn, self.Y_ijt, color=color, style=style)  # imported with plotting

    def build_uncertainty_set_for_time_delay(self, units=(0,), tasks=(0,), delay=1, from_t=0, to_t=None):
        # TODO fix doc, it's tuple; you need a comma at the end if it is one single unit (cannot iterate otherwise)
        """ Returns the uncertainty sets corresponding to possible delays of any task (in `tasks') executed
        on a unit (in `units'). The delays amount to up to `delay' time steps. Note: this is not equivalent to
        delays of exactly `delay' time steps, the ``up to'' is important. A version for `exact delays', which
        for instance can model the swapping of a certain operation from day to night is also possible.
        :param units: a (list of) unit(s) from STN.units (string); default='Heater'
        :param tasks: a (list of) task(s) from STN.tasks (string); default='Heat'
        :param delay: time delay, in steps (int); default=1
        :param from_t: delays may occur from time step ``from_t'' (int); default=0
        :param to_t: delays may occur only upt to time step ''to_t'' (int); default=STN.T
        :return: list of matrices [W_k], k=0,...,n_x-1
        """
        if to_t is None:
            to_t = self.stn.T-delay-1

        W = [0 for x in range(self.stn.n_x)]

        for i in units:
            for j in tasks:
                for t in range(from_t, to_t):
                    W_k = np.zeros((self.stn.n_x, delay+1))
                    # one column of the W_k matrix should always be 0, to encode that no event occurs;
                    # declaration left for clarity
                    W_k[:,0] = 0
                    # on to the other columns of W_k
                    for col in range(1, delay+1):
                        # index where we place the ``-1'' and ``+1''
                        minus_index = self.x_ijt_index_to_std([i,j,t])
                        plus_index = self.x_ijt_index_to_std([i,j,t+col])
                        # uncertainty vector w is -1 at (i,j,t), and +1 at (i,j,t+1)
                        w = np.zeros(self.stn.n_x)
                        w[minus_index] = -1
                        w[plus_index] = +1
                        W_k[:,col] = w

                    # assign matrix to the collection (list) of uncertainties
                    W[minus_index] = W_k
        # we return the value (and do not change self.W from within the method), since one may want
        # to combine several different W's before computing the RC
        return W

    def build_uncertainty_set_for_unit_swap(self, from_unit=2, to_unit=1, tasks=(1,), from_t=0, to_t=None):
        if to_t is None:
            to_t = self.stn.T-1

        W = [0 for x in range(self.stn.n_x)]

        for t in range(from_t, to_t):
            # two columns
            W_k = np.zeros((self.stn.n_x, 2))
            # one column of the W_k matrix should always be 0, to encode that no event occurs;
            # declaration left for clarity
            W_k[:,0] = 0

            # the other column encodes the desired unit swap
            for j in tasks:
                # uncertainty vector w is -1 at (from_unit,j,t), and +1 at (to_unit,j,t+1)
                minus_index = self.x_ijt_index_to_std([from_unit,j,t])
                plus_index = self.x_ijt_index_to_std([to_unit,j,t])
                w = np.zeros(self.stn.n_x)
                w[minus_index] = -1
                w[plus_index] = +1
                W_k[:,1] = w
                # assign matrix to the collection (list) of uncertainties
                W[minus_index] = W_k

        # we return the value (and do not change self.W from within the method), since one may want
        # to combine several different W's before computing the RC
        return W


    def build_robust_counterpart(self):
        # fill in the required columns in Phi and Psi (can only be done after a W is constructed)
        k_ix = []  # list of indices with non-zero entries in W[k_ix]
        for k in range(self.stn.n_x):
            if np.any(self.W[k]):
                self.Phi[k] = cvx.Variable(self.m,1)
                self.Psi[k] = cvx.Variable(self.m,1)
                k_ix.append(k)

        # Aggregate Nominal Model
        # -----------------------
        A = scipy.sparse.vstack((self.stn.A_eq,self.stn.A_ineq))
        B = scipy.sparse.vstack((self.stn.B_eq,self.stn.B_ineq))
        D = A

        # Constraints
        # -----------
        constraints = []
        for i, row in enumerate(self.stn.A_eq):
             constraints.append(self.stn.A_eq[i,:]*self.x + self.stn.B_eq[i,:]*self.v +
                                np.sum([self.Phi[k][i] for k in k_ix]) == self.stn.b_eq[i])
        for i, row in enumerate(self.stn.A_ineq):
             constraints.append(self.stn.A_ineq[i,:]*self.x + self.stn.B_ineq[i,:]*self.v +
                                np.sum([self.Phi[k][i+self.stn.m_eq] for k in k_ix]) <= self.stn.b_ineq[i])

        for k in k_ix:
            n_k = self.W[k].shape[1]
            constraints.append(np.ones((n_k,self.m))*cvx.diag(self.Psi[k]) >= ((B*self.Y + D)*self.W[k]).T)
            constraints.append(self.Phi[k] >= 0)
            constraints.append(self.Phi[k] <= self.x[k]*self.Psi_bar)
            constraints.append(self.Psi[k] - self.Phi[k] >= 0)
            constraints.append(self.Psi[k] - self.Phi[k] <= (1-self.x[k])*self.Psi_bar)
            # TODO: the following constraints speed up computations.
            constraints.append(self.v + self.Y*self.W[k][:,1] <=  np.max(self.stn.V_max))
            constraints.append(self.v + self.Y*self.W[k][:,1] >=  0)

        # non-anticipativity
        for t in range(self.stn.T):
            for i in range (self.stn.I):
                for j in range(self.stn.J):
                    row = self.x_ijt_index_to_std([i,j,t])
                    # loop over all necessary columns, forward in time
                    for ii in range(self.stn.I):
                        for jj in range(self.stn.J):
                            for tt in range(t+1,self.stn.T):
                                col = self.x_ijt_index_to_std([ii,jj,tt])
                                constraints.append(self.Y[row,col] == 0)

        objective = cvx.Minimize(self.stn.c_x*self.x + self.stn.c_y*self.v)
        self.robust_model = cvx.Problem(objective, constraints)

    def simulate_uncertain_event(self, event=1, column=1, color='yellow', style='ggplot'):
        # events start from 1

        self.Y_ijt_after_event[:,:,:] = 0
        # 1) extract appropriate column from W
        k_ix = []  # list of indices with non-zero entries in W[k_ix]
        for k in range(self.stn.n_x):
            if np.any(self.W[k]):
                k_ix.append(k)

        w = self.W[k_ix[event-1]][:,column]
        w = np.array([w]).T

        # 2) apply affine recourse
        y_ijt_after_event = self.v.value[:self.stn.n_x] + self.Y.value[:self.stn.n_x,:]*w

        # 3) paste result into self.Y_ijt_after_event
        for k in range(self.stn.n_x):
            i, j, t = self.std_index_to_x_ijt(k)
            self.Y_ijt_after_event[i,j,t] = y_ijt_after_event[k]

        plot_stn_schedule(self.stn, self.Y_ijt_after_event, color=color, style=style)



if __name__ == '__main__':
    from STN import *
    stn = STN()
    stn.solve()
    rSTN = robust_STN(stn)
    rSTN.W = rSTN.build_uncertainty_set_for_time_delay()
    rSTN.solve()
    rSTN.plot_schedule()
