import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.stats import norm

"""
Problem 6

"""

def plot_job_availability(A, D, n): 
    ax = plt.figure()
    plt.scatter(A, np.linspace(1,n,num=n), color='k', marker='*')
    plt.scatter(D+1, np.linspace(1,n,num=n), color='k', marker='o')
    for i in range(n):
        plt.plot(np.append(A[i],D[i]+1),np.append(i+1,i+1), 'k-')
    
    plt.xlabel('Time t', fontsize=font_size)
    plt.ylabel('Job i', fontsize=font_size)

def plot_speed(s): 
    pass 

# Given parameters
n = 12   # number of jobs.
T = 16   # number of time periods.
Smin = 1 # min processor speed.
Smax = 4 # max processor speed.
R = 1    # max slew rate.


alpha = 1
beta = 1
gamma = 1

# Job arrival times and deadlines.
A = np.array([1, 3, 4, 6, 9, 9, 11, 12, 13, 13, 14, 15])
D = np.array([3, 6, 11, 7, 9, 12, 13, 15, 15, 16, 14, 16])
# Total work for each job.
W = np.array([2, 4, 10, 2, 3, 2, 3, 2, 3, 4, 1, 4])

# define problem variables
P = cp.Variable((T, n))
s = cp.sum(P, axis=1)
theta = P / cp.reshape(s, (T, 1))

# define constraints
constraints = [
    P >= 0,
    s >= Smin,
    s <= Smax,
    cp.abs(s[1:] - s[:-1]) <= R, # slew-rate constraint
    cp.sum(P, axis=0) >= W
]

# job availability constraints 
for i in range(n):
    for t in range(A[i] - 1): # A[i] indexed at 1
        constraints.append(P[t, i] == 0)
    for t in range(D[i], T):
        constraints.append(P[t, i] == 0)

# define objective function
objective = cp.Minimize(cp.sum(alpha + beta * s + gamma * s**2))

problem = cp.Problem(objective, constraints)
problem.solve()

# optimal Energy
E = problem.value
print("The optimal Energy value is: ", E)

speed = s.value
print("\n The speeds that achive this are: \n", speed)

allocation = theta.value.T
#print("\n allocation: \n ", allocation)


plot_job_availability(A,D,n)