import numpy as np

n = 1000

A = np.diag(np.ones(n-2), k=-2) + np.diag(-4*np.ones(n-1), k=-1) + \
    np.diag(7*np.ones(n), k=0) + \
    np.diag(-4*np.ones(n-1), k=1) + np.diag(np.ones(n-2), k=2)

b = np.zeros((n,1))
b[0,0] = 3 ; b[1,0] = -1 ; b[n-1,0] = 3 ; b[n-2,0] = -1


gamma = 0.01          # Step size multiplier
precision = 1e-9     # Desired precision of result
max_iters = 1000000   # Maximum number of iterations
cvg = False           # Convergence flag

# your initial vector x guess
next_x = 6*np.ones((n,1))  # here, you chose 6

# derivative function
def df(A,b,x):
    return np.dot(A,x)-b

i=1
while i <= max_iters:
    curr_x = next_x
    next_x = curr_x - gamma * df(A,b,curr_x)

    step = next_x - curr_x
    if np.linalg.norm(step,2) <= precision:
        cvg = True
        break
    i += 1

if cvg :
    print('Minimum reached in ' + str(i) + ' iterations.')
    print('x=',next_x)
else :
    print('No convergence.')
