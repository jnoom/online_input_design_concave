import numpy as np
import cvxpy as cvx
import dccp

def minsumTaylor(TH,Tc,Th,E,nu,N,k):
    np.random.seed(int(k))
    n = len(Th)
    x = cvx.Variable(len(Tc[0, :]))

    cost = 0
    k = 0
    for i in range(n):
        cost += cvx.quad_form(x, TH[k]) + Tc[k].T @ x + Th[k]
        k += 1

    nui = int(nu)
    nconstr = int(len(Tc[0])/nui)  # number of constraints (equals horizon length)
    constr = []
    for i in range(nconstr):
        Q = np.zeros([len(Tc[0, :]),len(Tc[0,:])])
        Q[i*nui:(i+1)*nui, i*nui:(i+1)*nui] = np.eye(nui)
        constr += [cvx.quad_form(x, Q) <= E]

    prob = cvx.Problem(cvx.Minimize(cost), constr)

    prob.solve(method='dccp', ccp_times=int(N))
    return x.value


