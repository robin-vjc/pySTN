__author__ = 'vujanicr'
# when constructing the uncertainty vectors, it is important to know the order of the variables
# in the nominal problem in standard form; this is established in the __init__() method of the
# STN class, when filling the x_ijt, y_ijt and y_st dictionaries with cvx.Variable objects.

from STN import *
import scipy
import copy

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

    def solve(self):
        if self.W == []:
            print 'Uncertainty sets robust_STN.W are empty. Build them first.'
            print '(call, e.g., robust_STN.W = robust_STN.build_uncertainty_set_for_unit_delay()).'
        else:
            print 'Constructing robust model...'
            self.build_robust_counterpart()
            print 'Solving...'
            self.robust_model.solve(verbose=True, solver='GUROBI')
            self.unpack_results()

    def unpack_results(self):
        for k in range(self.x.size[0]):
            i,j,t = self.std_index_to_x_ijt(k)
            self.X_ijt[i,j,t] = copy.deepcopy(self.x.value[k])
            self.Y_ijt[i,j,t] = copy.deepcopy(self.v.value[k])
        self.Y_recourse = copy.deepcopy(self.Y.value)

    def plot_schedule(self):
        """ Plot the robust schedule.
        :return: None
        """
        # TODO you should plot the attained objective
        color = 'blue'
        margin = 0.03  # size of margins around the boxes

        if not self.X_ijt.any():
            print 'Please, solve model first by invoking robust_STN.solve()'
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect(1)
            for i, unit in enumerate(self.Y_ijt):
                unit_axis = np.array([i, i+1])
                for j, task in enumerate(unit):
                    for t, time in enumerate(task):
                        if t < self.stn.T:
                            slot_x = np.array([t+margin, t+self.stn.P_j[j]-margin])
                            slot_y = np.array([i+margin, i+margin])
                            slot_y2 = slot_y+0.90+margin
                            # don't plot blocks where Y_ijt is just some epsilon, residual of the optimization
                            if self.Y_ijt[i,j,t] >= 1:
                                plt.fill_between(slot_x, slot_y, y2=slot_y2, color=color)
                                plt.text(np.mean(slot_x), np.mean(unit_axis), "{0}\n{1}".format(self.Y_ijt[i,j,t], self.stn.tasks[j]),
                                         horizontalalignment='center', verticalalignment='center' )
                plt.text(-0.15, np.mean(unit_axis), "{0}".format(self.stn.units[i]),
                         horizontalalignment='right', verticalalignment='center')

            plt.ylim(self.stn.I, 0)
            plt.xlabel('time [h]')
            plt.yticks(range(self.stn.I), "", y=0.5)
            plt.show()

    def build_uncertainty_set_for_unit_delay(self, units=(0,), tasks=(0,), delay=1, from_t=0, to_t=None):
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
             constraints.append( self.stn.A_eq[i,:]*self.x + self.stn.B_eq[i,:]*self.v +
                                 np.sum( [self.Phi[k][i] for k in k_ix] ) == self.stn.b_eq[i] )
        for i, row in enumerate(self.stn.A_ineq):
             constraints.append( self.stn.A_ineq[i,:]*self.x + self.stn.B_ineq[i,:]*self.v +
                                 np.sum( [self.Phi[k][i+self.stn.m_eq] for k in k_ix] ) <= self.stn.b_ineq[i] )

        for k in k_ix:
            n_k = self.W[k].shape[1]
            constraints.append( np.ones((n_k,self.m))*cvx.diag(self.Psi[k]) >= ((B*self.Y + D)*self.W[k]).T )
            constraints.append(self.Phi[k] >= 0)
            constraints.append(self.Phi[k] <= self.x[k]*self.Psi_bar)
            constraints.append(self.Psi[k] - self.Phi[k] >= 0)
            constraints.append(self.Psi[k] - self.Phi[k] <= (1-self.x[k])*self.Psi_bar)
            # TODO: these following constraints help a lot computationally.
            constraints.append( self.v + self.Y*self.W[k][:,1] <=  np.max(self.stn.V_max))
            constraints.append( self.v + self.Y*self.W[k][:,1] >=  0)
            # TODO: non-anticipativity

        objective = cvx.Minimize(self.stn.c_x*self.x + self.stn.c_y*self.v)
        self.robust_model = cvx.Problem(objective, constraints)

if __name__ == '__main__':
    from STN import *
    stn = STN()
    stn.solve()
    rSTN = robust_STN(stn)
    rSTN.W = rSTN.build_uncertainty_set_for_unit_delay()
    rSTN.build_robust_counterpart()
    rSTN.robust_model.solve(verbose=True, solver='GUROBI')
