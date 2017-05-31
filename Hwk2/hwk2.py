import cmath as c
import numpy as np
import scipy as sp
import numpy.linalg as npla
import scipy.linalg as scla
#############
#Question 1b#
#############
n = 16
H = scla.hilbert(n)

q = np.array([complex(i,np.random.rand()) for i in np.random.rand(n)])
q = q/npla.norm(q,2)

steps = 5000
for i in range(0,steps):
    q = np.dot(H,q)
    q = q/npla.norm(q,2)
    l = np.dot(np.conj(q),np.dot(H,q))

print('The largest eigenvalue value is: %s' %str(l.real))
print('It\'s eigenvector is:\n%s' %np.array2string(q))

#############
#Question 1c#
#############
n=16
l1 = l.real #The eigenvalues are all real (since H is Hermitian), so this just ditches any rounding errors
mu = l1
H = scla.hilbert(n)
Hp = H - mu*np.eye(n)

q = np.array([complex(i,np.random.rand()) for i in np.random.rand(n)])
q = q/npla.norm(q,2)

steps = 50000
for i in range(0,steps):
    q = np.dot(Hp,q)
    q = q/npla.norm(q,2)
    l = np.dot(np.conj(q),np.dot(Hp,q))
l2 = l.real + mu
print('The smallest eigenvalue value is: %s' %str(l2))
print('It\'s eigenvector is:\n%s' %np.array2string(q))
