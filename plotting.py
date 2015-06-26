__author__ = 'robin'
"""
Implementation of the plotting function, used in the STN.py and robust_STN.py modules.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_stn_schedule(STN, Y_ijt, color='red', style='ggplot'):
    """
    Plot a schedule.
    :param STN: an instance of the STN class
    :param Y_ijt: the schedule to be plotted (numpy array)
    :param color: color of the blocks
    :param style: style; available options: ['bmh', 'grayscale', 'ggplot', 'fivethirtyeight']
    :return: None
    """
    # TODO you should plot the attained objective, and also (maybe) the states evolution, divided in input/int/output
    margin = 0.03  # size of margins around the boxes
    plt.style.use(style)

    if not Y_ijt.any():
        print 'Please, solve model first by invoking the solve() method'
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        for i, unit in enumerate(Y_ijt):
            unit_axis = np.array([i, i + 1])
            for j, task in enumerate(unit):
                for t, time in enumerate(task):
                    if t < STN.T:
                        slot_x = np.array([t + margin, t + STN.P_j[j] - margin])
                        slot_y = np.array([i + margin, i + margin])
                        slot_y2 = slot_y + 0.90 + margin
                        # don't plot blocks where Y_ijt is just some epsilon, residual of the optimization
                        if Y_ijt[i, j, t] >= 1:
                            plt.fill_between(slot_x, slot_y, y2=slot_y2, color=color)
                            text = format(Y_ijt[i, j, t], '.2f').rstrip('0').rstrip('.')
                            plt.text(np.mean(slot_x), np.mean(unit_axis), "{0}\n{1}".format(text, STN.tasks[j]),
                                     horizontalalignment='center', verticalalignment='center')
            plt.text(-0.15, np.mean(unit_axis), "{0}".format(STN.units[i]),
                     horizontalalignment='right', verticalalignment='center')

        plt.ylim(STN.I, 0)
        plt.xlabel('time [h]')
        plt.yticks(range(STN.I), "", y=0.5)
        plt.show()
