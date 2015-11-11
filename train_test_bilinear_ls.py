from __future__ import division

import numpy as np
import argparse
import cv2
import h5py
import util

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_file', type=str)
    parser.add_argument('test_data_file', nargs='?', type=str, default=None)
    parser.add_argument('--rescale_factor', type=int, default=10)
    parser.add_argument('--draw_vel', type=int, default=0)

    args = parser.parse_args()
    
    train_file = h5py.File(args.train_data_file, 'r+')
    if args.test_data_file is None:
        test_file = h5py.File(args.train_data_file.replace('train', 'test'), 'r+')
    else:
        test_file = h5py.File(args.test_data_file, 'r+')
    
    # train
    Y = train_file['image_curr'][:].astype(np.float)
    N = Y.shape[0]
    Y = Y.reshape((N, -1))
    U = train_file['vel'][:].astype(np.float)
    Y_dot = train_file['image_diff'][:].astype(np.float)
    Y_dot = Y_dot.reshape((N, -1))
    y_dim = Y.shape[1]
    u_dim = U.shape[1]
    bifun = BilinearFunction(y_dim, u_dim)
    bifun.fit(Y, U, Y_dot)
    print "train error", (np.linalg.norm(Y_dot - bifun.eval(Y, U))**2) / (2*N)

    # test
    Y = test_file['image_curr'][:].astype(np.float)
    N, height, width = Y.shape
    Y = Y.reshape((N, -1))
    U = test_file['vel'][:].astype(np.float)
    Y_dot = test_file['image_diff'][:].astype(np.float)
    Y_dot = Y_dot.reshape((N, -1))
    Y_dot_pred = bifun.eval(Y, U)
    print "test error", (np.linalg.norm(Y_dot - Y_dot_pred)**2) / (2*N)
    
    # visualize test data
    for image_curr_data, vel_data, image_diff_data, image_diff_pred_data in zip(Y.reshape((N, height, width)), U, Y_dot.reshape((N, height, width)), Y_dot_pred.reshape((N, height, width))):
        try:
            image_diff_pred_data = np.clip(image_diff_pred_data, -2, 2)
            vis_image_gt = util.create_vis_image(image_curr_data, vel_data, image_diff_data, rescale_factor=args.rescale_factor, draw_vel=args.draw_vel)
            vis_image_pred = util.create_vis_image(image_curr_data, vel_data, image_diff_pred_data, rescale_factor=args.rescale_factor, draw_vel=args.draw_vel)
            vis_image = np.r_[vis_image_gt, vis_image_pred]
            
            cv2.imshow("Image window", vis_image)
            key = cv2.waitKey(0)
            key &= 255
            if key == 27 or key == ord('q'):
                break
        except KeyboardInterrupt:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
