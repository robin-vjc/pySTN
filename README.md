# pySTN
Implementation of a robust scheduling system based on STN (State-Task-Network) models.

## Description
This is supporting code to reproduce the results in [1].
It allows the construction of a schedule optimizer according to the model in [2], and it also
implements a robustification procedure that renders the schedules immune to uncertain events such as
unit or tasks delays an swaps.

## Usage
Please refer to [this IPython notebook](./STN with Python.ipynb).

## References
[1] R. Vujanic, P. Goulart, M. Morari, ``A New Approach for Robust Optimization of Mixed–Integer Programs,
with an Application to Scheduling``, June 2015, submitted.
[2] E. Kondili, C.C. Pantelides, and R.W.H. Sargent, ``A general algorithm for short-term scheduling of batch
operations - I. MILP formulation.``, Computers & Chemical Engineering 17 (1993), no. 2, 211–227.

