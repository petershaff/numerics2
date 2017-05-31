import numpy as np
############
#Question 1#
############
def qr_decomp(A):
    if np.shape(A)[0] != np.shape(A)[1]:
        print('Not a square matrix, fool!')
        return([A,A])

    else:
        dim = np.shape(A)[0]
        R = A.copy()
        Q = np.eye(dim)
        for i in range(0,dim-1):
            a = R[:,i].copy()
            a[0:i] = 0
            r = np.zeros(dim)
            r[i] = np.linalg.norm(a,ord=2)
            u = a - r
            H = np.eye(dim) - np.outer(u,u)*2*(1/np.linalg.norm(u,ord=2)**2)
            Q = np.dot(Q,H)
            R = np.dot(H,R)
        return([Q,R])

def qr_iter(A, tol=1e-6):
    err = tol + 1
    L = A
    t = 0
    dim = np.shape(A)[0]
    mu = np.mean(np.diag(A))
    if np.shape(A)[0] != np.shape(A)[1]:
        print('Still not square(, dork.')
        return(A)

    try:
        while err > tol:
            q,r = qr_decomp(L)# - mu*np.eye(dim))
            L = np.dot(r,q)# + mu*np.eye(dim)
            err = np.linalg.norm(np.tril(L,k=-1),ord=np.inf)
            t+=1
             
    except KeyboardInterrupt:
        return(L)

    return([L,t])

def rand_tridiag(n):
    test = np.zeros([n,n])
    for i in range(0,n):
        if i==0:
            test[i,i] = np.random.rand(1)
            test[i+1,i] = np.random.rand(1)

        elif i < n-1:
            test[i,i] = np.random.rand(1)
            test[i-1,i] = test[i,i-1]
            test[i+1,i] = np.random.rand(1)

        else:
            test[i-1,i] = test[i,i-1]
            test[i,i] = np.random.rand(1)

    return(10*test)

def does_this_script_work(n):
    test = rand_tridiag(n)
    eigs = np.sort(np.diag(qr_iter(test)[0]))
    err = np.linalg.norm(eigs - np.sort(np.linalg.eig(test)[0]))
    return(err/n)
    
############
#Question 2#
############
compare_list = []
for eps in np.arange(1e-4,1e-1,1e-5):
    A = np.array([[2,eps],[eps,2]])
    #Without shift:
    q,r = qr_decomp(A)
    L1 = np.dot(r,q)

    #With shift:
    q,r = qr_decomp(A-np.eye(2))
    L2 = np.dot(r,q) + np.eye(2)

    #Record which did better
    compare_list.append(abs(L1[1,0]) - abs(L2[1,0]))

#Did the shift help at all?
print(any(np.array(compare_list)>0))
