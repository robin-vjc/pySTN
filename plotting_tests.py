__author__ = 'vujanicr'
import matplotlib.pyplot as plt
import numpy as np

def plot_schedule(model):
        ## TODO warning if problem is not solved
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        for i, unit in enumerate(model.Y_ijt):
            for j, task in enumerate(unit):
                for t, time in enumerate(task):
                    if t < model.T:
                        slot_x = np.array([t+0.03, t+model.P_j[j]-0.03])
                        slot_y = np.array([i+0.03, i+0.03])
                        slot_y2 = slot_y+0.93
                        slot_y3 = np.array([i, i+1])
                        # don't plot blocks where Y_ijt is just some epsilon, residual of the optimization
                        if model.Y_ijt[i,j,t] >= 1:
                            plt.fill_between(slot_x, slot_y, y2=slot_y2, color='red')
                            plt.text(np.mean(slot_x), np.mean(slot_y3), "{0}\n{1}".format(model.Y_ijt[i,j,t], model.tasks[j]),
                                     horizontalalignment='center', verticalalignment='center' )
                            plt.text(-0.15, np.mean(slot_y3), "{0}".format(model.units[i]), horizontalalignment='right', verticalalignment='center')

        plt.ylim(model.I, 0)
        plt.xlabel('time [h]')
        plt.yticks(range(model.I), "", y=0.5)
        plt.show()