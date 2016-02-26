import numpy as np

class BilinearFunction(object):
    """
        y_dot^k = sum_{i,j} q_{ij}^k y_i u_j + \sum_j r_j^k u_j + \sum_i s_i^k y_i + b^k
    """
    def __init__(self, y_dim, u_dim):
        self.I = self.K = y_dim
        self.J = u_dim
        self.Q = np.zeros((self.K, self.I, self.J))
        self.R = np.zeros((self.K, self.J))
        self.S = np.zeros((self.K, self.I))
        self.b = np.zeros(self.K)

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
        Y_dot = X.reshape((-1, I*J)).dot(self.Q.reshape((-1, I*J)).T) + U.dot(self.R.T) + Y.dot(self.S.T) + self.b[None, :]
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
                        Y_dot[n,k] += self.Q[k,i,j] * Y[n,i] * U[n,j]
                    Y_dot[n,k] += self.R[k,j] * U[n,j]
                for i in range(I):
                    Y_dot[n,k] += self.S[k,i] * Y[n,i]
                Y_dot[n,k] += self.b[k]
        if ndim == 1:
            Y_dot = Y_dot.reshape(-1)
        return Y_dot

    def eval3(self, Y, U):
        N, I, J = Y.shape[0], self.I, self.J
        assert Y.shape == (N, I)
        assert U.shape == (N, J)
        X = np.einsum("ni,nj->nij", Y, U)
        Z = np.c_[X.reshape((-1, I*J)), U, Y, np.ones(N)]
        C = np.c_[self.Q.reshape((-1, I*J)), self.R, self.S, self.b]
        return Z.dot(C.T)

    def jac_u(self, Y):
        ndim = Y.ndim
        if ndim == 1:
            jac = np.einsum("kij,i->kj", self.Q, Y) + self.R
            return jac
        else:
            return np.asarray([self.jac_u(y) for y in Y])

    def fit(self, Y, U, Y_dot):
        N, I, J, K = Y.shape[0], self.I, self.J, self.K
        assert Y.shape == (N, I)
        assert U.shape == (N, J)
        assert Y_dot.shape == (N, K)
        X = np.einsum("ni,nj->nij", Y, U)
        Z = np.c_[X.reshape((-1, I*J)), U, Y, np.ones(N)]
        C = np.linalg.solve(Z.T.dot(Z), Z.T.dot(Y_dot)).T
        self.Q, self.R, self.S, self.b = C[:, :I*J].reshape((K, I, J)), C[:, I*J:I*J+J], C[:, I*J+J:I*J+J+I], np.squeeze(C[:, I*J+J+I:])

    @staticmethod
    def compute_solver_terms(Y, U, Y_dot):
        N, I = Y.shape
        _, J = U.shape
        _, K = Y_dot.shape
        assert Y.shape == (N, I)
        assert U.shape == (N, J)
        assert Y_dot.shape == (N, K)
        X = np.einsum("ni,nj->nij", Y, U)
        Z = np.c_[X.reshape((-1, I*J)), U, Y, np.ones(N)]
        A, B = Z.T.dot(Z), Z.T.dot(Y_dot)
        post_fit = lambda C: ((C.T)[:, :I*J].reshape((K, I, J)), (C.T)[:, I*J:I*J+J], (C.T)[:, I*J+J:I*J+J+I], np.squeeze((C.T)[:, I*J+J+I:]))
        return A, B, post_fit
