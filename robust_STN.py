__author__ = 'vujanicr'
# when constructing the uncertainty vectors, it is important to know the order of the variables
# in the nominal problem in standard form; this is established in the __init__() method of the
# STN class, when filling the x_ijt, y_ijt and y_st dictionaries with cvx.Variable objects.

from STN import *

class robust_STN(object):
    """ robust_STN uses the nominal STN model, and provides
    - example methods to construct appropriate uncertainty sets,
    - a method to build the Robust Counterpart (RC)
    - a method to solve the RC
    - plotting
    """
    def __init__(self, stn=STN()):
        """
        :type stn: STN
        """
        self.stn = stn

        # uncertainty sets are stored as matrices, within a list of length n_x (number of bools
        # in the problem).
        self.W_k = [0 for x in range(self.stn.n_x)]

    def build_uncertainty_set_for_unit_delay(self, units='Heater', tasks='Heat', delay=1, from_t=0, to_t=None):
        """ Returns the uncertainty sets corresponding to possible delays of any task (in `tasks') executed
        on a unit (in `units'). The delays amount to `delay' time steps.
        :param units: a (list of) units from STN.units (string)
        :param tasks: a (list of) tasks from STN.tasks (string)
        :param delay: time delay, in steps (int)
        :param from_t: delays may occur from time step ``from_t'' (int)
        :param to_t: delays may occur only upt to time step ''to_t'' (int)
        :return: list of matrices [W_k], k=0,...,n_x-1
        """
        if to_t is None:
            to_t = self.stn.T

        units =
        for t in range(self.stn.T):

            # create a column in W_k (where k is the appropriate index, where you have the -1)




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
