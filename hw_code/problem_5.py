import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.stats import norm

"""
Problem 5
"""

mu1 = 8
mu2 = 20
sigma1 = 6
sigma2 = 17.5
rho = -0.25

n = 100
r_1, r_n = -30, 70

# discretize R1 and R2 by n in [-30, 70]
r = np.linspace(r_1, r_n, n)

# marginal distributions
p1 = np.exp(-(r - mu1)**2 / (2 * sigma1**2))
p1 /= np.sum(p1)

p2 = np.exp(-(r - mu2)**2 / (2 * sigma2**2))
p2 /= np.sum(p2)

# region where R1 + R2 <= 0
r1p, r2p = np.meshgrid(r, r)
loss_region = (r1p + r2p <= 0)

"""
Define optimization problem
"""
P = cp.Variable((n, n), nonneg=True)

# Maximize: 
objective = cp.Maximize(cp.sum(cp.multiply(P, loss_region)))

#Subject to 
constraint1 = cp.sum(P, axis=1) == p1
constraint2 = cp.sum(P, axis=0) == p2
constraint3 = (r - mu1) @ P @ (r - mu2) == rho * sigma1 * sigma2
constraints = [constraint1, constraint2, constraint3]

# solve problem
problem = cp.Problem(objective, constraints)
problem.solve()

p_max = problem.value  # worst case probability of loss
print("Worst-case probability of loss: ", p_max)

# joint probability matrix
P = P.value
print(P)

# Plotting the first figure
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(r1p, r2p, P.T, cmap='viridis')
ax1.set_xlabel('R1')
ax1.set_ylabel('R2')
ax1.set_zlabel('Density')
ax1.set_title('3D Plot')

# Plotting the second figure
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
contour = ax2.contour(r1p, r2p, P.T, levels=20, cmap='viridis')
ax2.set_xlabel('R1')
ax2.set_ylabel('R2')
ax2.set_title('Contour Plot')
ax2.grid(True)

plt.show()
