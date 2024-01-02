import numpy as np
import cvxpy as cvx
import dccp


def minsumbhat(H,c,h,P,E,nu,ccpN,k):
    np.random.seed(int(k))
    n = len(P)
    nH = len(H[:,0,0,0])
    x = cvx.Variable(len(c[0,0,:]))

    cost = 0
    k = 0
    for i in range(n-1):
        for j in range(i+1, n):
            for j2 in range(nH):
                cost += np.sqrt(P[i] * P[j]) * cvx.exp(-cvx.quad_form(x, H[j2,k]) - c[j2,k].T @ x - h[j2,k])
            k += 1

    nui = int(nu)
    nconstr = int(len(c[0,0])/nui)  # number of constraints (equals horizon length)
    constr = []
    for i in range(nconstr):
        Q = np.zeros([len(c[0,0, :]),len(c[0,0,:])])
        Q[i*nui:(i+1)*nui, i*nui:(i+1)*nui] = np.eye(nui)
        constr += [cvx.quad_form(x, Q) <= E]

    prob = cvx.Problem(cvx.Minimize(cost), constr)

    prob.solve(method='dccp',ccp_times=int(ccpN))
    return x.value


