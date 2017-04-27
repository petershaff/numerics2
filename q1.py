import numpy as np
import scipy as sp
import scipy.optimize as spot
import numpy.linalg as npla
import matplotlib.pyplot as plt
from tqdm import *

class ode_obliterator_v2_obsequious_oyster(dict):
    def __init__(self, RHS, init_val, init_time=0, h=1e-1, q=2,n=2):
        self.yp = RHS
        self.iv = np.array(init_val)
        self.init_time = init_time
        self.t = init_time
        self.h = h
        self.q = q
        self.n = n
        self.dim = max(self.iv.shape)

        self.sol = self.iv
        self.times = np.array([self.t])

    def sacramento_scramble(self,h,curr,t):
        v = curr + .5*h*np.dot(self.yp(t),curr)
        A = np.eye(self.dim) - .5*h*self.yp(t+h)
        output = npla.solve(A,v)
        return(output)

    def basic_trap(self,end,h):
        hold = self.iv
        times = np.arange(self.init_time,end,h)
        curr = self.iv
        for t in times:
            curr = self.sacramento_scramble(h,curr,t)
            hold = np.vstack([hold,curr])
        return(hold)
        
    def boston_u_turn(self,init):
        n = len(init)
        table = np.zeros([n,n])
        table[:,0] = init

        for j in range(1,n):
            for i in range(j,n):
                p = 2*j+1
                table[i,j] = table[i,j-1] +(table[i,j-1] - table[i-1,j-1])/float(self.q**p-1) 

        update = table[n-1,n-1]

        return(update)

    def boulder_shimmy(self,end_time):
        t_steps = np.arange(self.init_time,end_time,self.h)
        self.multigrid = np.zeros([self.n,len(t_steps),self.dim])
        for k in range(0,self.n):
            h = self.q**(-1*k)*self.h
            curr = self.iv
            for t in tqdm(range(len(t_steps))):
                time = t_steps[t]
                for i in np.arange(self.q**k):
                    curr = self.sacramento_scramble(h,curr,time)
                    time += h
                
                self.multigrid[k,t,:] = curr
        
    def houston_plug_n_chug(self,end_time):
        self.boulder_shimmy(end_time)
        hold = np.zeros(self.multigrid.shape[1:3])
        for i in tqdm(range(hold.shape[0])):
            for j in range(hold.shape[1]):
                hold[i,j] = self.boston_u_turn(self.multigrid[:,i,j])

        self.sol = hold
        self.times = np.arange(self.init_time,end_time,self.h)

    
def ode(t):
    A = np.array([[0.,1.],[-1.,-3./t]])
    return(A)

y0 = np.array([.5,0.])
trap_solve = ode_obliterator_v2(ode,y0, init_time = 1e-10, h = 1e-5,n=2)
trap_solve.houston_plug_n_chug(3*np.pi)

x = trap_solve.times
y_true = sp.special.jv(1,x)
y_int = x*trap_solve.sol[:,0]

err = max(y_int-y_true)

plt.plot(x,y_true)
plt.plot(x,y_int)
plt.show()

