from __future__ import division

import numpy as np

class BilinearFunction(object):
    """
        y_dot^k = sum_{i,j} a_{ij}^k y_i u_j + \sum_j b_j^k u_j
    """
    def __init__(self, y_dim, u_dim):
        self.I = self.K = y_dim
        self.J = u_dim
        self.A = np.zeros((self.K, self.I, self.J))
        self.B = np.zeros((self.K, self.J))

    def eval(self, Y, U):
        ndim = Y.ndim
        assert U.ndim == ndim
        if ndim == 1:
            Y = Y.reshape((1, -1))
            U = U.reshape((1, -1))
        N, I, J = Y.shape[0], self.I, self.J
        assert Y.shape == (N, I)
        assert U.shape == (N, J)
        X = np.einsum("ni,nj->nij", Y, U)
        Y_dot = X.reshape((-1, I*J)).dot(self.A.reshape((-1, I*J)).T) + U.dot(self.B.T)
        if ndim == 1:
            Y_dot = Y_dot.reshape(-1)
        return Y_dot

    def eval2(self, Y, U):
        ndim = Y.ndim
        assert U.ndim == ndim
        if ndim == 1:
            Y = Y.reshape((1, -1))
            U = U.reshape((1, -1))
        N, I, J, K = Y.shape[0], self.I, self.J, self.K
        assert Y.shape == (N, I)
        assert U.shape == (N, J)
        Y_dot = np.zeros((N, K))
        for n in range(N):
            for k in range(K):
                for j in range(J):
                    for i in range(I):
                        Y_dot[n,k] += self.A[k,i,j] * Y[n,i] * U[n,j]
                    Y_dot[n,k] += self.B[k,j] * U[n,j]
        if ndim == 1:
            Y_dot = Y_dot.reshape(-1)
        return Y_dot

    def eval3(self, Y, U):
        N, I, J = Y.shape[0], self.I, self.J
        assert Y.shape == (N, I)
        assert U.shape == (N, J)
        X = np.einsum("ni,nj->nij", Y, U)
        Z = np.c_[X.reshape((-1, I*J)), U]
        C = np.c_[self.A.reshape((-1, I*J)), self.B]
        return Z.dot(C.T)

    def jac_u(self, Y):
        ndim = Y.ndim
        if ndim == 1:
            jac = np.einsum("kij,i->kj", self.A, Y) + self.B
            return jac
        else:
            return np.asarray([self.jac_u(y) for y in Y])

    def fit(self, Y, U, Y_dot):
        N, I, J, K = Y.shape[0], self.I, self.J, self.K
        assert Y.shape == (N, I)
        assert U.shape == (N, J)
        assert Y_dot.shape == (N, K)
        X = np.einsum("ni,nj->nij", Y, U)
        Z = np.c_[X.reshape((-1, I*J)), U]
        C = np.linalg.solve(Z.T.dot(Z), Z.T.dot(Y_dot)).T
        self.A, self.B = C[:, :I*J].reshape((K, I, J)), C[:, I*J:]
