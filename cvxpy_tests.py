import cvxpy as cvx
import numpy as np

x = cvx.Variable(2,2)

interesting_ix = [(0,0),(0,1)]
for i in interesting_ix:
    print i

A = np.array([1, 0])
b = np.array([0.7])

objective = cvx.Maximize(cvx.sum_entries(x[0,:])+0.5*cvx.sum_entries(x[1,:]))
constraints = [0 <= x,
               x <= 1,
               x[0,:] <= 0.7]
# constraints.append(sum([x[i] for i in interesting_ix]) <= 0.5)
constraints.append(sum([x[i,:] for i in range(2)]) <= 0.5)

prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True)

print result
print x.value


I = 4
J = 5
T = 11
S = 9

x_ijt = {}  # allocation variable (bool)
y_ijt = {}  # batch size [kg]
y_st = {}   # state quantity [kg]
for t in range(T):
    for i in range(I):
        for j in range(J):
            x_ijt[(i,j,t)] = cvx.Variable()
            y_ijt[(i,j,t)] = cvx.Variable()
            # TODO maybe not used:
            for s in range(S):
                y_st[(s,t)] = cvx.Variable()