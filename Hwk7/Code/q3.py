import numpy as np

#y' = l y

def fwd(h,y0,y1,l):
    y2 = 4.*y1 -3.*y0 - 2.*h*l*y0
    return(y2)
    
h = 1e-5
l = -1
t0 = 0
t1 = t0 + h

y0 = np.exp(l*t0)
y1 = np.exp(l*t1)

y = [y0,y1]
for i in range(2,102):
    next = fwd(h,y[i-2],y[i-1],l)
    y.append(next)
    
