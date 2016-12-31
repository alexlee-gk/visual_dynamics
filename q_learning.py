from __future__ import division, print_function
import numpy as np
import cvxpy
import utils


def as_list(l):
    if isinstance(l, list):
        return l
    else:
        return [l]


class QLearning(object):
    def __init__(self, servoing_policy, tr_reg=0.0, l1_reg=0.0, l2_reg=0.0, gamma=0.9, max_iters=5, theta_init=None, learn_lambda=True,
                 experience_replay=True, learn_bias=True):
        self.pol = servoing_policy
        self.predictor = servoing_policy.predictor
        self.action_transformer = self.predictor.transformers['u']
        self.tr_reg = tr_reg
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.gamma = gamma  # discount factor
        self.max_iters = max_iters
        self.learn_lambda = learn_lambda
        self.experience_replay = experience_replay
        self.learn_bias = learn_bias

        self._S = []
        self._A = []
        self._R = []
        self._S_p = []

        feature = self.predictor.feature(np.zeros(self.predictor.input_shapes[0]))
        features = feature if isinstance(feature, list) else [feature]
        self.repeats = []
        for feature in features:
            self.repeats.extend([np.prod(feature.shape[1:])] * feature.shape[0])

        self._theta = np.empty(len(self.repeats) + 2)
        if theta_init is None:
            self.theta = np.r_[np.ones(len(self.repeats)), 0.0, self.lambda_]
        else:
            self.theta = theta_init

        self.train_fn = None

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self.theta[...] = theta
        self.pol.w = np.repeat(self.theta[:-2], self.repeats)
        # theta[-2] is bias
        self.pol.lambda_ = self.theta[-1]

    @property
    def alpha(self):
        return self.pol.alpha

    @property
    def lambda_(self):
        return self.pol.lambda_

    def preprocess_action(self, a):
        _, action_shape = self.predictor.input_shapes
        if not isinstance(a, list) and a.shape == action_shape:
            a_prep = self.action_transformer.preprocess(a)
        else:
            A = a
            a_prep = np.asarray([self.action_transformer.preprocess(a) for a in A])
        return a_prep

    def phi(self, s, a):
        if not isinstance(s, list):
            obs, target_obs = s
            image = obs[0]
            target_image = target_obs[0]
            action = a

            feature = self.predictor.feature(image)
            next_feature = self.predictor.next_feature(image, action)
            target_feature = self.predictor.feature(target_image)

            features = feature if isinstance(feature, list) else [feature]
            next_features = next_feature if isinstance(next_feature, list) else [next_feature]
            target_features = target_feature if isinstance(target_feature, list) else [target_feature]

            phis = []
            for feature, next_feature, target_feature in zip(features, next_features, target_features):
                y_diff = self.alpha * target_feature + (1 - self.alpha) * feature - next_feature
                phi = (y_diff ** 2).sum(axis=(-2, -1))
                phis.append(phi)
            phis.append([1])
            phis.append([self.action_transformer.preprocess(action).T.dot(self.action_transformer.preprocess(action))])
            phi = np.concatenate(phis)
            # print(phi.dot(self.theta))
            return phi
        else:
            S, A = s, a
            return np.asarray([self.phi(s, a) for (s, a) in zip(S, A)])

    def batch_phi(self, S, A):
        assert len(S) == len(A)
        assert len(S) % 100 == 0
        if len(S) == 100:
            batch_image = np.asarray([obs[0] for (obs, target_obs) in S])
            batch_target_image = np.asarray([target_obs[0] for (obs, target_obs) in S])
            batch_action = A

            feature_names = as_list(self.predictor.feature_name)
            next_feature_names = as_list(self.predictor.next_feature_name)
            feature_and_next_feature_names = feature_names + next_feature_names
            batch_features_and_next_feature = \
                self.predictor.predict(feature_and_next_feature_names, batch_image, batch_action)
            batch_features = batch_features_and_next_feature[:len(feature_names)]
            batch_next_features = batch_features_and_next_feature[len(feature_names):]
            batch_target_features = self.predictor.feature(batch_target_image)

            batch_phis = []
            for batch_feature, batch_next_feature, batch_target_feature in zip(batch_features, batch_next_features,
                                                                               batch_target_features):
                y_diff = self.alpha * batch_target_feature + (1 - self.alpha) * batch_feature - batch_next_feature
                batch_phi = (y_diff ** 2).sum(axis=(-2, -1))
                batch_phis.append(batch_phi)

            batch_phis.append(np.ones((len(S), 1)))
            batch_preprocessed_action = self.preprocess_action(batch_action)
            batch_phis.append((batch_preprocessed_action ** 2).sum(axis=-1)[:, None])
            batch_phi = np.concatenate(batch_phis, axis=-1)
            return batch_phi
        else:
            batch_phis = []
            for i in range(len(S) // 100):
                batch_phi = self.batch_phi(S[i * 100:(i + 1) * 100], A[i * 100:(i + 1) * 100])
                batch_phis.append(batch_phi)
            batch_phi = np.concatenate(batch_phis, axis=0)
            return batch_phi

    def Q(self, s, a):
        return self.phi(s, a).dot(self.theta)

    def pi(self, s):
        # TODO: test all the methods, starting from here
        # argmin_a Q(s, a) = argmin_a theta @ phi(s, a) + lambda a @ a, where theta >= 0
        # argmin_u w @ phi(s, u) + lambda u @ u, where w >= 0
        if not isinstance(s, list):
            obs, target_obs = s
            # self.pol.set_target(target_obs)  # TODO
            self.pol.set_image_target(target_obs[0])
            # TODO: check objective is indeed less
            a = self.pol.act(obs)
            # a_random = np.random.random((a.shape))
            # assert self.phi(s, a_random).dot(self.theta) >= self.phi(s, a).dot(self.theta)
            return a
        else:
            S = s
            return np.asarray([self.pi(s) for s in S])

    def batch_pi(self, s):
        # argmin_a Q(s, a) = argmin_a theta @ phi(s, a) + lambda a @ a, where theta >= 0
        # argmin_u w @ phi(s, u) + lambda u @ u, where w >= 0
        assert len(s) % 100 == 0
        if len(s) == 100:
            batch_image = np.asarray([obs[0] for (obs, target_obs) in s])
            batch_target_image = np.asarray([target_obs[0] for (obs, target_obs) in s])
            batch_target_feature = self.predictor.feature(batch_target_image)
            _, action_shape = self.predictor.input_shapes
            batch_zero_action = np.zeros((len(batch_image),) + action_shape)

            # TODO: hard-coded way to pick mode
            if self.predictor.bilinear_type == 'channelwise':
                mode = 'linear'
            elif self.predictor.bilinear_type == 'group_convolution':
                mode = 'linear2'
            else:
                mode = None
            batch_jac, batch_next_feature = self.predictor.jacobian(self.predictor.next_feature_name,
                                                                            self.predictor.control_name,
                                                                            batch_image,
                                                                            batch_zero_action,
                                                                            ret_outputs=True,
                                                                            mode=mode)

            batch_actions = []
            for obs, target_feature, jac, next_feature in zip(s, zip(*batch_target_feature), zip(*batch_jac), zip(*batch_next_feature)):
                self.pol._y_target[...] = np.concatenate([f.flatten() for f in target_feature])
                a = self.pol.act(obs, jac=jac, next_feature=next_feature)
                batch_actions.append(a)
            batch_action = np.array(batch_actions)
            return batch_action
        else:
            batch_actions = []
            for i in range(len(s) // 100):
                batch_action = self.batch_pi(s[i * 100:(i + 1) * 100])
                batch_actions.append(batch_action)
            batch_action = np.concatenate(batch_actions, axis=0)
            return batch_action

    def V(self, s):
        # min_a Q(s, a)
        return self.Q(s, self.pi(s))

    def fit_imitation(self, S, A, A_target):
        batch_image = np.asarray([obs[0] for (obs, target_obs) in S])
        batch_target_image = np.asarray([target_obs[0] for (obs, target_obs) in S])
        batch_target_feature = self.predictor.feature(batch_target_image)
        _, action_shape = self.predictor.input_shapes
        batch_zero_action = np.zeros((len(batch_image),) + action_shape)

        # TODO: hard-coded way to pick mode
        if self.predictor.bilinear_type == 'channelwise' or \
                (hasattr(self.predictor, 'tf_nl_layer') and self.predictor.tf_nl_layer):
            mode = 'linear'
        elif self.predictor.bilinear_type in ('group_convolution', 'channelwise_local'):
            mode = 'linear2'
        else:
            mode = None
        batch_jac, batch_next_feature = self.predictor.jacobian(self.predictor.next_feature_name,
                                                                self.predictor.control_name,
                                                                batch_image,
                                                                batch_zero_action,
                                                                ret_outputs=True,
                                                                mode=mode)
        import theano
        import theano.tensor as T
        assert self.pol.w is not None
        # w_var = theano.shared(self.pol.w.astype(theano.config.floatX))
        theta_var = theano.shared(self.theta.astype(theano.config.floatX))

        y_targets = []
        Js = []
        y_next_preds = []
        for (obs, target_feature, jac, next_feature) in zip(S, zip(*batch_target_feature), zip(*batch_jac), zip(*batch_next_feature)):
            image = obs[0]
            y_target = np.concatenate([f.flatten() for f in target_feature])
            J = np.concatenate(jac)
            y_next_pred = np.concatenate([f.flatten() for f in next_feature])
            if self.alpha != 1.0:
                feature = self.predictor.feature(image)
                y = np.concatenate([f.flatten() for f in feature])
                y_target = self.alpha * y_target + (1 - self.alpha) * y
            y_targets.append(y_target)
            Js.append(J)
            y_next_preds.append(y_next_pred)
        y_targets = np.array(y_targets)
        Js = np.array(Js)
        y_next_preds = np.array(y_next_preds)

        # import IPython as ipy; ipy.embed()

        # w = np.concatenate([self.theta[i] * np.ones(repeat) for i, repeat in enumerate(self.repeats)])
        # lambda_ = self.theta[-1]

        # w_var = T.nnet.relu(T.concatenate([theta_var[i] * T.ones(repeat) for i, repeat in enumerate(self.repeats)]))
        # lambda_var = T.nnet.relu(theta_var[-1])
        # WJ_var = Js * w_var[None, :, None]

        ys = y_targets - y_next_preds + Js.dot(self.action_transformer.preprocess(np.zeros(action_shape)))

        cs_repeats = np.cumsum(np.r_[0, self.repeats])
        slices = [slice(start, stop) for (start, stop) in zip(cs_repeats[:-1], cs_repeats[1:])]
        A_split = np.array([np.einsum('nij,nik->njk', Js[:, s], Js[:, s]) for s in slices])
        B_split = np.array([np.einsum('nij,ni->nj', Js[:, s], ys[:, s]) for s in slices])

        w_var = T.nnet.relu(theta_var[:len(A_split)])
        lambda_var = T.nnet.relu(theta_var[-1])

        # A_var = T.batched_tensordot(WJ_var, Js, axes=(1, 1)) + lambda_var * T.eye(Js.shape[-1])
        # B_var = T.batched_tensordot(WJ_var, y_target - y_next_pred + Js.dot(self.action_transformer.preprocess(np.zeros(action_shape))), axes=(1, 1))
        A_var = T.tensordot(A_split, w_var, axes=(0, 0)) + lambda_var * T.eye(A_split.shape[-1])
        B_var = T.tensordot(B_split, w_var, axes=(0, 0))
        U_var = T.batched_dot(T.nlinalg.matrix_inverse(A_var), B_var)

        U_target = np.array([self.action_transformer.preprocess(a_target) for a_target in A_target])
        loss_var = ((U_var - U_target) ** 2).mean(axis=0).sum() / 2.

        # TODO: see why compiling imitation functiont akes so long
        # TODO: recursion limit; maybe not expand theta into w in the first place
        # TODO: verify matrix_inverse is correct

        # def act(y_target, J, y_next_pred, theta_var):
        #     # w_var = T.repeat(theta_var[:-2], self.repeats)
        #     # w_var = T.concatenate([theta_var[i] * T.ones(repeat) for i, repeat in enumerate(self.repeats)])
        #     # lambda_var = theta_var[-1]
        #     w_var = T.nnet.relu(T.concatenate([theta_var[i] * T.ones(repeat) for i, repeat in enumerate(self.repeats)]))
        #     lambda_var = T.nnet.relu(theta_var[-1])
        #     WJ = J * w_var[:, None]
        #     A = WJ.T.dot(J) + lambda_var * T.eye(J.shape[1])
        #     b = WJ.T.dot(y_target - y_next_pred + J.dot(self.action_transformer.preprocess(np.zeros(action_shape))))
        #     u_var = T.nlinalg.matrix_inverse(A).dot(b)
        #     return u_var
        #
        # def act_ind(i, y_targets, Js, y_next_preds, theta_var):
        #     return act(y_targets[i], Js[i], y_next_preds[i], theta_var)
        #
        # U_var, _ = theano.scan(fn=act_ind,
        #                         outputs_info=None,
        #                         sequences=T.arange(len(y_targets)),
        #                         non_sequences=[y_targets, Js, y_next_preds, theta_var])
        # U_target = np.array([self.action_transformer.preprocess(a_target) for a_target in A_target])
        # loss_var = ((U_var - U_target) ** 2).mean(axis=0).sum() / 2.

        # import sys
        # sys.setrecursionlimit(10000)

        import lasagne
        learning_rate_var = theano.tensor.scalar(name='learning_rate')
        # updates = lasagne.updates.momentum(loss_var, [w_var], learning_rate_var)
        updates = lasagne.updates.adam(loss_var, [theta_var], learning_rate=learning_rate_var)

        import time
        start_time = time.time()
        print("Compiling imitation training function...")
        train_fn = theano.function([learning_rate_var], loss_var, updates=updates)
        print("... finished in %.2f s" % (time.time() - start_time))
        max_iter = 1000
        train_losses = []
        for iter_ in range(max_iter):
            train_loss = float(train_fn(0.001))
            print("Iteration {} of {}".format(iter_, max_iter))
            print("    training loss = {:.6f}".format(train_loss))
            train_losses.append(train_loss)
        self.theta = np.maximum(theta_var.get_value(), 0)  # TODO?
        return train_losses

    def fit(self, S, A, R, S_p, feature_std=None):
        if self.experience_replay:
            capacity = 1000
            minibatch_size = 100
            self._S.extend(S)
            self._A.extend(A)
            self._R.extend(R)
            self._S_p.extend(S_p)
            if len(self._S) > capacity:
                self._S = self._S[-capacity:]
                self._A = self._A[-capacity:]
                self._R = self._R[-capacity:]
                self._S_p = self._S_p[-capacity:]
        else:
            minibatch_size = 100
            _S = S
            _A = A
            _R = R
            _S_p = S_p

        # # TODO: make sure S is not all of it when computing stats
        # batch_image = np.asarray([obs[0] for (obs, target_obs) in S])
        # batch_feature = self.predictor.feature(batch_image)
        # batch_features = batch_feature if isinstance(batch_feature, list) else [batch_feature]
        # feature_stds = []
        # for batch_feature in batch_features:
        #     feature_std = batch_feature.transpose([0, 2, 3, 1]).reshape((-1, batch_feature.shape[1])).std(axis=0)
        #     feature_stds.append(feature_std)
        # feature_std = np.concatenate(feature_stds)
        # zero_feature_std_ind = feature_std == 0
        # feature_std[feature_std == 0] = 1.0

        if feature_std is not None:
            theta_scale = np.r_[feature_std, 1.0, 1.0]
        else:
            theta_scale = np.ones(self.theta.shape[0])

        for iter_ in range(self.max_iters):
            if self.experience_replay:
                if len(self._S) > minibatch_size:
                    choice = np.random.choice(len(self._S), minibatch_size)
                    S = [self._S[i] for i in choice]
                    A = [self._A[i] for i in choice]
                    R = [self._R[i] for i in choice]
                    S_p = [self._S_p[i] for i in choice]
                else:
                    S = self._S
                    A = self._A
                    R = self._R
                    S_p = self._S_p
            else:
                if len(_S) > minibatch_size:
                    choice = np.random.choice(len(_S), minibatch_size)
                    S = [_S[i] for i in choice]
                    A = [_A[i] for i in choice]
                    R = [_R[i] for i in choice]
                    S_p = [_S_p[i] for i in choice]
            A = np.asarray(A)
            R = np.asarray(R)

            from utils import tic, toc
            tic()
            phi = self.batch_phi(S, A)
            # if feature_std is not None:
            #     phi[:, :-2] /= feature_std
            toc("phi")

            tic()
            if self.gamma == 0:
                # TODO: scale for stability
                Q_sample = R
            else:
                A_p = self.batch_pi(S_p)
                toc("pi")
                phi_p = self.batch_phi(S_p, A_p)
                # if feature_std is not None:
                #    phi_p[:, :-2] /= feature_std
                V_p = phi_p.dot(self.theta)
                Q_sample = R + self.gamma * V_p
                # Q_sample = R + self.gamma * self.V(S_p)
            toc("Q_sample")

            tic()
            lsq_A = phi
            lsq_b = Q_sample  # - self.lambda_ * (self.preprocess_action(S, A) ** 2).sum(axis=-1)

            scale = 0.01  # scale the objective for a more stable optimization
            x = theta_scale * self.theta
            prev_x = x.copy()

            unreg_loss = (((lsq_A / theta_scale).dot(x) - lsq_b) ** 2).mean(axis=0).sum() / 2
            loss = unreg_loss + self.tr_reg * ((x - prev_x) ** 2).sum() \
                              + self.l1_reg * np.abs(x[:-2]).sum() \
                              + self.l2_reg * (x[:-2] ** 2).sum()
            print('loss: %f\t%f' % (unreg_loss, loss))
            #                         loss +
            #                         self.tr_reg * ((self.theta - prev_theta) ** 2).sum() +
            #                         self.l1_reg * np.abs(self.theta[:-2]).sum() +
            #                         self.l2_reg * (self.theta[:-2] ** 2).sum()))

            x_var = cvxpy.Variable(x.shape[0])
            ### old
            # objective = cvxpy.Minimize((1 / (2 * len(lsq_A))) * cvxpy.sum_squares(lsq_A * x - lsq_b)
            #                            + self.l1_reg * cvxpy.norm(x[:-1], 1))  # no regularization on last weight
            # constraints = [0 <= x, x[:-1] <= 1]  # no upper constraint on last weight
            # constraints.append(cvxpy.sum_entries(x[:-1]) == 1)
            ###
            objective = cvxpy.Minimize((1 / (2 * len(lsq_A))) * cvxpy.sum_squares((np.sqrt(scale) * lsq_A / theta_scale) * x_var - np.sqrt(scale) * lsq_b) +
                                       scale * self.tr_reg * cvxpy.sum_squares(x_var - prev_x) +
                                       scale * self.l1_reg * cvxpy.norm(x_var[:-2], 1) +
                                       scale * self.l2_reg * cvxpy.norm(x_var[:-2], 2))
            constraints = [0 <= x_var[:-2], 0 <= x_var[-1]]  # no constraint on bias
            # constraints = [0 <= x, cvxpy.sum_entries(x[:-1]) == 1]  # no normalization constraint on last weight
            # constraints = [0 <= x, cvxpy.sum_entries(x[:-1]) == (self.theta.shape[0]-1)]  # no normalization constraint on last weight
            if not self.learn_bias:
                constraints.append(x_var[-2] == self.x[-2])
            if not self.learn_lambda:
                constraints.append(x_var[-1] == self.lambda_)
            # for ind in np.where(zero_feature_std_ind)[0]:
            #     constraints.append(x[ind] == 0)

            # TODO
            # assert self.theta.shape[0] == 512*3+1
            # for i in range(512 * 3):
            #     # if (i % 512) not in [36, 178, 307, 490]:
            #     if i not in [0, 1, 2, 3, 36, 178, 307, 490]:
            #         constraints.append(x[i] == 0)

            prob = cvxpy.Problem(objective, constraints)

            solved = False
            for solver in [None, cvxpy.GUROBI, cvxpy.CVXOPT]:
                try:
                    prob.solve(solver=solver)
                except cvxpy.error.SolverError:
                    continue
                if x_var.value is None:
                    continue
                solved = True
                break
            if not solved:
                import IPython as ipy;
                ipy.embed()
            x = np.squeeze(np.array(x_var.value), axis=1)
            toc("cvxpy")

            # theta[:-1] *= feature_std
            converged = np.allclose(x, self.theta, atol=1e-3)
            self.theta = x / theta_scale

            new_unreg_loss = (((lsq_A / theta_scale).dot(x) - lsq_b) ** 2).mean(axis=0).sum() / 2
            new_loss = new_unreg_loss + self.tr_reg * ((x - prev_x) ** 2).sum() \
                                      + self.l1_reg * np.abs(x[:-2]).sum() \
                                      + self.l2_reg * (x[:-2] ** 2).sum()
            print('new_loss: %f\t%f' % (new_unreg_loss, new_loss))
            # TODO: assert this?
            if new_loss > loss:
                print("new loss is higher")
                import IPython as ipy; ipy.embed()
            if converged:
                break
            print('iteration %d of %d' % (iter_, self.max_iters))
            print('\ttheta: %r' % self.theta)
            print('\ttheta != 0: %r' % self.theta[self.theta > 1e-9])
            print('\tsort theta: %r' % np.sort(self.theta)[::-1][:16])
            print('\targ sort theta: %r' % np.argsort(self.theta)[::-1][:16])
        # import IPython as ipy; ipy.embed()
        if converged:
            print('fitted Q-iteration converged in %d iterations' % (iter_ + 1))
        else:
            print('fitted Q-iteration reached maximum number of iterations %d' % self.max_iters)
        return prob.value / scale


class QLearningSlow(object):
    def __init__(self, servoing_policy, l1_reg=0.0, gamma=0.9, max_iters=5, theta_init=None, learn_lambda=True,
                 experience_replay=True):
        self.pol = servoing_policy
        self.predictor = servoing_policy.predictor
        self.action_transformer = self.predictor.transformers['u']
        self.l1_reg = l1_reg
        self.gamma = gamma  # discount factor
        self.max_iters = max_iters
        self.learn_lambda = learn_lambda
        self.experience_replay = experience_replay

        self._S = []
        self._A = []
        self._R = []
        self._S_p = []

        feature = self.predictor.feature(np.zeros(self.predictor.input_shapes[0]))
        features = feature if isinstance(feature, list) else [feature]
        self.repeats = []
        for feature in features:
            self.repeats.extend([np.prod(feature.shape[1:])] * feature.shape[0])

        self._theta = np.empty(len(self.repeats) + 1)
        if theta_init is None:
            self.theta = np.r_[np.ones(len(self.repeats)), self.lambda_]
        else:
            self.theta = theta_init

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self.theta[...] = theta
        self.pol.w = np.repeat(theta[:-1], self.repeats)
        self.pol.lambda_ = self.theta[-1]

    @property
    def alpha(self):
        return self.pol.alpha

    @property
    def lambda_(self):
        return self.pol.lambda_

    def preprocess_action(self, a):
        _, action_shape = self.predictor.input_shapes
        if not isinstance(a, list) and a.shape == action_shape:
            a_prep = self.action_transformer.preprocess(a)
        else:
            A = a
            a_prep = np.asarray([self.action_transformer.preprocess(a) for a in A])
        return a_prep

    def phi(self, s, a):
        if not isinstance(s, list):
            obs, target_obs = s
            image = obs[0]
            target_image = target_obs[0]
            action = a

            feature = self.predictor.feature(image)
            next_feature = self.predictor.next_feature(image, action)
            target_feature = self.predictor.feature(target_image)

            features = feature if isinstance(feature, list) else [feature]
            next_features = next_feature if isinstance(next_feature, list) else [next_feature]
            target_features = target_feature if isinstance(target_feature, list) else [target_feature]

            phis = []
            for feature, next_feature, target_feature in zip(features, next_features, target_features):
                y_diff = self.alpha * target_feature + (1 - self.alpha) * feature - next_feature
                phi = (y_diff ** 2).sum(axis=(-2, -1))
                phis.append(phi)
            phis.append([self.action_transformer.preprocess(action).T.dot(self.action_transformer.preprocess(action))])
            phi = np.concatenate(phis)
            # print(phi.dot(self.theta))
            return phi
        else:
            S, A = s, a
            return np.asarray([self.phi(s, a) for (s, a) in zip(S, A)])

    def batch_phi(self, S, A):
        assert len(S) == len(A)
        assert len(S) % 100 == 0
        if len(S) == 100:
            batch_image = np.asarray([obs[0] for (obs, target_obs) in S])
            batch_target_image = np.asarray([target_obs[0] for (obs, target_obs) in S])
            batch_action = A

            feature_names = as_list(self.predictor.feature_name)
            next_feature_names = as_list(self.predictor.next_feature_name)
            feature_and_next_feature_names = feature_names + next_feature_names
            batch_features_and_next_feature = \
                self.predictor.predict(feature_and_next_feature_names, batch_image, batch_action)
            batch_features = batch_features_and_next_feature[:len(feature_names)]
            batch_next_features = batch_features_and_next_feature[len(feature_names):]
            batch_target_features = self.predictor.feature(batch_target_image)

            batch_phis = []
            for batch_feature, batch_next_feature, batch_target_feature in zip(batch_features, batch_next_features,
                                                                               batch_target_features):
                y_diff = self.alpha * batch_target_feature + (1 - self.alpha) * batch_feature - batch_next_feature
                batch_phi = (y_diff ** 2).sum(axis=(-2, -1))
                batch_phis.append(batch_phi)

            batch_preprocessed_action = self.preprocess_action(batch_action)
            batch_phis.append((batch_preprocessed_action ** 2).sum(axis=-1)[:, None])
            batch_phi = np.concatenate(batch_phis, axis=-1)
            return batch_phi
        else:
            batch_phis = []
            for i in range(len(S) // 100):
                batch_phi = self.batch_phi(S[i * 100:(i + 1) * 100], A[i * 100:(i + 1) * 100])
                batch_phis.append(batch_phi)
            batch_phi = np.concatenate(batch_phis, axis=0)
            return batch_phi

    def Q(self, s, a):
        return self.phi(s, a).dot(self.theta)

    def pi(self, s):
        # TODO: test all the methods, starting from here
        # argmin_a Q(s, a) = argmin_a theta @ phi(s, a) + lambda a @ a, where theta >= 0
        # argmin_u w @ phi(s, u) + lambda u @ u, where w >= 0
        if not isinstance(s, list):
            obs, target_obs = s
            # self.pol.set_target(target_obs)  # TODO
            self.pol.set_image_target(target_obs[0])
            # TODO: check objective is indeed less
            a = self.pol.act(obs)
            # a_random = np.random.random((a.shape))
            # assert self.phi(s, a_random).dot(self.theta) >= self.phi(s, a).dot(self.theta)
            return a
        else:
            S = s
            return np.asarray([self.pi(s) for s in S])

    def batch_pi(self, s):
        # argmin_a Q(s, a) = argmin_a theta @ phi(s, a) + lambda a @ a, where theta >= 0
        # argmin_u w @ phi(s, u) + lambda u @ u, where w >= 0
        assert len(s) % 100 == 0
        if len(s) == 100:
            batch_image = np.asarray([obs[0] for (obs, target_obs) in s])
            batch_target_image = np.asarray([target_obs[0] for (obs, target_obs) in s])
            batch_target_feature = self.predictor.feature(batch_target_image)
            _, action_shape = self.predictor.input_shapes
            batch_zero_action = np.zeros((len(batch_image),) + action_shape)

            # TODO: hard-coded way to pick mode
            if self.predictor.bilinear_type == 'channelwise':
                mode = 'linear'
            elif self.predictor.bilinear_type == 'group_convolution':
                mode = 'linear2'
            else:
                mode = None
            batch_jac, batch_next_feature = self.predictor.jacobian(self.predictor.next_feature_name,
                                                                            self.predictor.control_name,
                                                                            batch_image,
                                                                            batch_zero_action,
                                                                            ret_outputs=True,
                                                                            mode=mode)

            batch_actions = []
            for obs, target_feature, jac, next_feature in zip(s, zip(*batch_target_feature), zip(*batch_jac), zip(*batch_next_feature)):
                self.pol._y_target[...] = np.concatenate([f.flatten() for f in target_feature])
                a = self.pol.act(obs, jac=jac, next_feature=next_feature)
                batch_actions.append(a)
            batch_action = np.array(batch_actions)
            return batch_action
        else:
            batch_actions = []
            for i in range(len(s) // 100):
                batch_action = self.batch_pi(s[i * 100:(i + 1) * 100])
                batch_actions.append(batch_action)
            batch_action = np.concatenate(batch_actions, axis=0)
            return batch_action

    def V(self, s):
        # min_a Q(s, a)
        return self.Q(s, self.pi(s))

    def fit(self, S, A, R, S_p, env_first_state_action, do_rollout):
        if self.experience_replay:
            self._S.extend(S)
            self._A.extend(A)
            self._R.extend(R)
            self._S_p.extend(S_p)
            if len(self._S) > 500:
                self._S = self._S[-500:]
                self._A = self._A[-500:]
                self._R = self._R[-500:]
                self._S_p = self._S_p[-500:]
            S = self._S
            A = self._A
            R = self._R
            S_p = self._S_p

        # TODO: make sure S is not all of it when computing stats
        batch_image = np.asarray([obs[0] for (obs, target_obs) in S])
        batch_feature = self.predictor.feature(batch_image)
        batch_features = batch_feature if isinstance(batch_feature, list) else [batch_feature]
        feature_stds = []
        for batch_feature in batch_features:
            feature_std = batch_feature.transpose([0, 2, 3, 1]).reshape((-1, batch_feature.shape[1])).std(axis=0)
            feature_stds.append(feature_std)
        feature_std = np.concatenate(feature_stds)
        zero_feature_std_ind = feature_std == 0
        feature_std[feature_std == 0] = 1.0

        # TODO:
        # batched pi
        # action regularization instead of clipping
        # constrained minimization?
        # random action noise should be scaled

        A = np.asarray(A)
        R = np.asarray(R)
        for iter_ in range(self.max_iters):
            from utils import tic, toc
            tic()
            phi = self.batch_phi(S, A)
            phi[:, :-1] /= feature_std
            toc("phi")

            tic()
            if self.gamma == 0.0:
                Q_sample = R
            else:
                A_p = self.batch_pi(S_p)
                toc("pi")
                phi_p = self.batch_phi(S_p, A_p)
                phi_p[:, :-1] /= feature_std
                V_p = phi_p.dot(self.theta)
                Q_sample = R + self.gamma * V_p
                # Q_sample = R + self.gamma * self.V(S_p)
            # Q_sample = []
            # for state, action in env_first_state_action:
            #     rewards = do_rollout(state, action, 25)
            #     Q_sample.append((np.array(rewards) * (self.gamma ** np.arange(len(rewards)))).sum())
            # Q_sample = np.asarray(Q_sample)
            toc("Q_sample")

            tic()
            lsq_A = phi
            lsq_b = Q_sample  # - self.lambda_ * (self.preprocess_action(S, A) ** 2).sum(axis=-1)

            # loss = ((lsq_A.dot(self.theta) - lsq_b) ** 2).mean(axis=0).sum() / 2
            # print('%.2f\t%.2f' % (loss, loss + self.l1_reg * np.abs(self.theta).sum()))

            x = cvxpy.Variable(self.theta.shape[0])
            ### old
            # objective = cvxpy.Minimize((1 / (2 * len(lsq_A))) * cvxpy.sum_squares(lsq_A * x - lsq_b)
            #                            + self.l1_reg * cvxpy.norm(x[:-1], 1))  # no regularization on last weight
            # constraints = [0 <= x, x[:-1] <= 1]  # no upper constraint on last weight
            # constraints.append(cvxpy.sum_entries(x[:-1]) == 1)
            ###
            objective = cvxpy.Minimize((1 / (2 * len(lsq_A))) * cvxpy.sum_squares(lsq_A * x - lsq_b))
            constraints = [0 <= x]
            # constraints = [0 <= x, cvxpy.sum_entries(x[:-1]) == 1]  # no normalization constraint on last weight
            # constraints = [0 <= x, cvxpy.sum_entries(x[:-1]) == (self.theta.shape[0]-1)]  # no normalization constraint on last weight
            if not self.learn_lambda:
                constraints.append(x[-1] == self.lambda_)
            for ind in np.where(zero_feature_std_ind)[0]:
                constraints.append(x[ind] == 0)
            prob = cvxpy.Problem(objective, constraints)

            solved = False
            for solver in [None, cvxpy.GUROBI, cvxpy.CVXOPT]:
                try:
                    prob.solve(solver=solver)
                except cvxpy.error.SolverError:
                    continue
                if x.value is None:
                    continue
                solved = True
                break
            if not solved:
                import IPython as ipy;
                ipy.embed()
            theta = np.squeeze(np.array(x.value), axis=1)
            toc("cvxpy")

            theta[:-1] *= feature_std
            converged = np.allclose(theta, self.theta, atol=1e-3)
            self.theta = theta
            if converged:
                break
            print('iteration %d of %d' % (iter_, self.max_iters))
            print('\ttheta: %r' % self.theta)
        if converged:
            print('fitted Q-iteration converged in %d iterations' % (iter_ + 1))
        else:
            print('fitted Q-iteration reached maximum number of iterations %d' % self.max_iters)
        return prob.value


class QLearning2(object):
    def __init__(self, servoing_policy, l1_reg=0.0, gamma=0.9, max_iters=5, theta_init=None, learn_lambda=True,
                 experience_replay=True):
        self.pol = servoing_policy
        self.predictor = servoing_policy.predictor
        self.action_transformer = self.predictor.transformers['u']
        self.l1_reg = l1_reg
        self.gamma = gamma  # discount factor
        self.max_iters = max_iters
        self.learn_lambda = learn_lambda
        self.experience_replay = experience_replay

        self._S = []
        self._A = []
        self._R = []
        self._S_p = []
        self._batch_phi = [[]]
        self._batch_target_feature = [[]]
        self._batch_jac = [[]]
        self._batch_next_feature = [[]]

        feature = self.predictor.feature(np.zeros(self.predictor.input_shapes[0]))
        features = feature if isinstance(feature, list) else [feature]
        self.repeats = []
        for feature in features:
            self.repeats.extend([np.prod(feature.shape[1:])] * feature.shape[0])
        self.feature_stats = [utils.OnlineStatistics() for _ in features]

        self._theta = np.empty(len(self.repeats) + 1)
        if theta_init is None:
            self.theta = np.r_[np.ones(len(self.repeats)), self.lambda_]
        else:
            self.theta = theta_init

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self.theta[...] = theta
        self.pol.w = np.repeat(theta[:-1], self.repeats)
        self.pol.lambda_ = self.theta[-1]

    @property
    def alpha(self):
        return self.pol.alpha

    @property
    def lambda_(self):
        return self.pol.lambda_

    def preprocess_action(self, a):
        _, action_shape = self.predictor.input_shapes
        if not isinstance(a, list) and a.shape == action_shape:
            a_prep = self.action_transformer.preprocess(a)
        else:
            A = a
            a_prep = np.asarray([self.action_transformer.preprocess(a) for a in A])
        return a_prep

    def phi(self, s, a):
        if not isinstance(s, list):
            obs, target_obs = s
            image = obs[0]
            target_image = target_obs[0]
            action = a

            feature = self.predictor.feature(image)
            next_feature = self.predictor.next_feature(image, action)
            target_feature = self.predictor.feature(target_image)

            features = feature if isinstance(feature, list) else [feature]
            next_features = next_feature if isinstance(next_feature, list) else [next_feature]
            target_features = target_feature if isinstance(target_feature, list) else [target_feature]

            phis = []
            for feature, next_feature, target_feature in zip(features, next_features, target_features):
                y_diff = self.alpha * target_feature + (1 - self.alpha) * feature - next_feature
                phi = (y_diff ** 2).sum(axis=(-2, -1))
                phis.append(phi)
            phis.append([self.action_transformer.preprocess(action).T.dot(self.action_transformer.preprocess(action))])
            phi = np.concatenate(phis)
            # print(phi.dot(self.theta))
            return phi
        else:
            S, A = s, a
            return np.asarray([self.phi(s, a) for (s, a) in zip(S, A)])

    def batch_phi(self, S, A):
        assert len(A) == len(A)
        assert len(S) % 100 == 0
        if len(S) == 100:
            batch_image = np.asarray([obs[0] for (obs, target_obs) in S])
            batch_target_image = np.asarray([target_obs[0] for (obs, target_obs) in S])
            batch_action = A

            feature_names = as_list(self.predictor.feature_name)
            next_feature_names = as_list(self.predictor.next_feature_name)
            feature_and_next_feature_names = feature_names + next_feature_names
            batch_features_and_next_feature = \
                self.predictor.predict(feature_and_next_feature_names, batch_image, batch_action)
            batch_features = batch_features_and_next_feature[:len(feature_names)]
            batch_next_features = batch_features_and_next_feature[len(feature_names):]
            batch_target_features = self.predictor.feature(batch_target_image)

            batch_phis = []
            for batch_feature, batch_next_feature, batch_target_feature in zip(batch_features, batch_next_features,
                                                                               batch_target_features):
                y_diff = self.alpha * batch_target_feature + (1 - self.alpha) * batch_feature - batch_next_feature
                batch_phi = (y_diff ** 2).sum(axis=(-2, -1))
                batch_phis.append(batch_phi)

            batch_preprocessed_action = self.preprocess_action(batch_action)
            batch_phis.append((batch_preprocessed_action ** 2).sum(axis=-1)[:, None])
            batch_phi = np.concatenate(batch_phis, axis=-1)
            return batch_phi
        else:
            batch_phis = []
            for i in range(len(S) // 100):
                batch_phi = self.batch_phi(S[i * 100:(i + 1) * 100], A[i * 100:(i + 1) * 100])
                batch_phis.append(batch_phi)
            batch_phi = np.concatenate(batch_phis, axis=0)
            return batch_phi

    def cached_batch_phi(self, S, A):
        try:
            assert len(A) == len(A)
            if len(S) > len(self._batch_phi[0]):
                batch_image = np.asarray([obs[0] for (obs, target_obs) in S[len(self._batch_phi[0]):]])
                batch_target_image = np.asarray([target_obs[0] for (obs, target_obs) in S[len(self._batch_phi[0]):]])
                batch_action = A[len(self._batch_phi[0]):]

                feature_names = as_list(self.predictor.feature_name)
                next_feature_names = as_list(self.predictor.next_feature_name)
                feature_and_next_feature_names = feature_names + next_feature_names
                batch_features_and_next_feature = \
                    self.predictor.predict(feature_and_next_feature_names, batch_image, batch_action)
                batch_features = batch_features_and_next_feature[:len(feature_names)]
                batch_next_features = batch_features_and_next_feature[len(feature_names):]
                batch_target_features = self.predictor.feature(batch_target_image)

                batch_phis = []
                for batch_feature, batch_next_feature, batch_target_feature in zip(batch_features, batch_next_features,
                                                                                   batch_target_features):
                    y_diff = self.alpha * batch_target_feature + (1 - self.alpha) * batch_feature - batch_next_feature
                    batch_phi = (y_diff ** 2).sum(axis=(-2, -1))
                    batch_phis.append(batch_phi)

                batch_preprocessed_action = self.preprocess_action(batch_action)
                batch_phis.append((batch_preprocessed_action ** 2).sum(axis=-1)[:, None])
                batch_phi = np.concatenate(batch_phis, axis=-1)
                if self._batch_phi == [[]]:
                    self._batch_phi = batch_phi
                else:
                    self._batch_phi = np.append(self._batch_phi, batch_phi, axis=0)

            batch_phi = self._batch_phi[:len(S)]
        except:
            import IPython as ipy; ipy.embed()
        return batch_phi

    def Q(self, s, a):
        return self.phi(s, a).dot(self.theta)

    def cached_batch_pi(self, s):
        # argmin_a Q(s, a) = argmin_a theta @ phi(s, a) + lambda a @ a, where theta >= 0
        # argmin_u w @ phi(s, u) + lambda u @ u, where w >= 0
        try:
            assert len(self._batch_jac) == len(self._batch_next_feature)
            assert len(self._batch_jac) == len(self._batch_target_feature)
            if len(s) > len(self._batch_jac[0]):
                batch_image = np.asarray([obs[0] for (obs, target_obs) in s[len(self._batch_jac[0]):]])
                batch_target_image = np.asarray([target_obs[0] for (obs, target_obs) in s[len(self._batch_jac[0]):]])
                batch_target_feature = self.predictor.feature(batch_target_image)
                _, action_shape = self.predictor.input_shapes
                batch_zero_action = np.zeros((len(batch_image),) + action_shape)

                # TODO: hard-coded way to pick mode
                if self.predictor.bilinear_type == 'channelwise':
                    mode = 'linear'
                elif self.predictor.bilinear_type == 'group_convolution':
                    mode = 'linear2'
                else:
                    mode = None
                batch_jac, batch_next_feature = self.predictor.batched_jacobian(self.predictor.next_feature_name,
                                                                                self.predictor.control_name,
                                                                                batch_image,
                                                                                batch_zero_action,
                                                                                ret_outputs=True,
                                                                                mode=mode)

                if self._batch_jac == [[]]:
                    self._batch_jac = batch_jac
                else:
                    for i in range(len(batch_jac)):
                        self._batch_jac[i] = np.append(self._batch_jac[i], batch_jac[i], axis=0)
                if self._batch_next_feature == [[]]:
                    self._batch_next_feature = batch_next_feature
                else:
                    for i in range(len(batch_next_feature)):
                        self._batch_next_feature[i] = np.append(self._batch_next_feature[i], batch_next_feature[i], axis=0)
                if self._batch_target_feature == [[]]:
                    self._batch_target_feature = batch_target_feature
                else:
                    for i in range(len(batch_target_feature)):
                        self._batch_target_feature[i] = np.append(self._batch_target_feature[i], batch_target_feature[i], axis=0)

            batch_jac = [batch_jac[:len(s)] for batch_jac in self._batch_jac]
            batch_next_feature = [batch_next_feature[:len(s)] for batch_next_feature in self._batch_next_feature]
            batch_target_feature = [batch_target_feature[:len(s)] for batch_target_feature in self._batch_target_feature]

            actions = []
            for obs, target_feature, jac, next_feature in zip(s, zip(*batch_target_feature), zip(*batch_jac), zip(*batch_next_feature)):
                self.pol._y_target[...] = np.concatenate([f.flatten() for f in target_feature])
                a = self.pol.act(obs, jac=jac, next_feature=next_feature)
                actions.append(a)
            actions = np.array(actions)
        except:
            import IPython as ipy; ipy.embed()
        return np.asarray(actions)

    def pi(self, s):
        # argmin_a Q(s, a) = argmin_a theta @ phi(s, a) + lambda a @ a, where theta >= 0
        # argmin_u w @ phi(s, u) + lambda u @ u, where w >= 0
        if not isinstance(s, list):
            obs, target_obs = s
            self.pol.set_target(target_obs)
            # self.pol.set_image_target(target_obs[0])
            # TODO: pass in cached jacobian and next_feature
            # TODO: check objective is indeed less
            a = self.pol.act(obs)
            # a_random = np.random.random((a.shape))
            # assert self.phi(s, a_random).dot(self.theta) >= self.phi(s, a).dot(self.theta)
            return a
        else:
            S = s
            return np.asarray([self.pi(s) for s in S])

    def V(self, s):
        # min_a Q(s, a)
        return self.Q(s, self.pi(s))

    def fit(self, S, A, R, S_p):
        if not self.experience_replay:
            for stat in self.feature_stats:
                stat.reset()

        # stats are updated with only the new states
        batch_image = np.asarray([obs[0] for (obs, target_obs) in S])
        batch_feature = self.predictor.feature(batch_image)
        batch_features = batch_feature if isinstance(batch_feature, list) else [batch_feature]
        feature_stds = []
        for batch_feature, stat in zip(batch_features, self.feature_stats):
            stat.add_data(batch_feature.transpose([0, 2, 3, 1]).reshape((-1, batch_feature.shape[1])))
            feature_stds.append(stat.std)
        feature_std = np.concatenate(feature_stds)
        feature_std[feature_std == 0] = 1.0

        if self.experience_replay:
            self._S.extend(S)
            self._A.extend(A)
            self._R.extend(R)
            self._S_p.extend(S_p)
            S = self._S
            A = self._A
            R = self._R
            S_p = self._S_p


        # TODO:
        # batched pi
        # esd before fitting
        # prevent memory error when computing mean; running mean?
        # check if rescaling in Q is the same as scaling in phi

        A = np.asarray(A)
        R = np.asarray(R)
        for iter_ in range(self.max_iters):
            try:
                from utils import tic, toc
                tic()
                phi = self.cached_batch_phi(S, A)
                phi[:, :-1] /= feature_std
                toc("phi")

                # phi 8.55722284317
                # pi 14.5149598122
                # Q_sample 23.5793268681

                # phi 3.54333901405
                # pi 9.41984891891
                # Q_sample 13.1131119728

                # ####
                # S2 = []
                # A2 = []
                # for i in range(2):
                #     S2.extend(S)
                #     A2.extend(A)
                # S = S2
                # A = np.asarray(A2)
                # S_p = S
                # #
                # # tic()
                # # batch_phi = self.batch_phi(S, A)
                # # toc()
                # # import IPython as ipy; ipy.embed()
                #
                # tic()
                # batch_image = np.asarray([obs[0] for (obs, target_obs) in S])
                # batch_target_image = np.asarray([target_obs[0] for (obs, target_obs) in S])
                # batch_action = A
                # toc()
                #
                # tic()
                # feature_names = as_list(self.predictor.feature_name)
                # next_feature_names = as_list(self.predictor.next_feature_name)
                # feature_and_next_feature_names = feature_names + next_feature_names
                # batch_features_and_next_feature = \
                #     self.predictor.predict(feature_and_next_feature_names, batch_image, batch_action)
                # batch_features = batch_features_and_next_feature[:len(feature_names)]
                # batch_next_features = batch_features_and_next_feature[len(feature_names):]
                # batch_target_features = self.predictor.feature(batch_target_image)
                # toc()
                #
                # tic()
                # batch_next_feature = \
                #     self.predictor.predict(next_feature_names, batch_image, batch_action)
                # toc()
                #
                # tic()
                # batch_phis = []
                # for batch_feature, batch_next_feature, batch_target_feature in zip(batch_features, batch_next_features,
                #                                                                    batch_target_features):
                #     y_diff = self.alpha * batch_target_feature + (1 - self.alpha) * batch_feature - batch_next_feature
                #     batch_phi = (y_diff ** 2).sum(axis=(-2, -1))
                #     batch_phis.append(batch_phi)
                # toc()
                #
                # tic()
                # batch_preprocessed_action = self.preprocess_action(batch_action)
                # batch_phis.append((batch_preprocessed_action ** 2).sum(axis=-1)[:, None])
                # batch_phi = np.concatenate(batch_phis, axis=-1)
                # toc()
                #
                # tic()
                # batch_image = np.asarray([obs[0] for (obs, target_obs) in s])
                # batch_target_image = np.asarray([obs[1] for (obs, target_obs) in s])
                # batch_target_feature = self.predictor.feature(batch_target_image)
                # _, action_shape = self.predictor.input_shapes
                # batch_zero_action = np.zeros((len(batch_image),) + action_shape)
                # toc()
                #
                # tic()
                # # TODO: hard-coded way to pick mode
                # if self.predictor.bilinear_type == 'channelwise':
                #     mode = 'linear'
                # elif self.predictor.bilinear_type == 'group_convolution':
                #     mode = 'linear2'
                # else:
                #     mode = None
                # batch_jac, batch_next_feature = self.predictor.batched_jacobian(self.predictor.next_feature_name,
                #                                                                 self.predictor.control_name,
                #                                                                 batch_image,
                #                                                                 batch_zero_action,
                #                                                                 ret_outputs=True,
                #                                                                 mode=mode)
                # toc()
                #
                # tic()
                # actions = []
                # for obs, target_feature, jac, next_feature in zip(s, zip(*batch_target_feature), zip(*batch_jac),
                #                                                   zip(*batch_next_feature)):
                #     self.pol._y_target[...] = np.concatenate([f.flatten() for f in target_feature])
                #     a = self.pol.act(obs, jac=jac, next_feature=next_feature)
                #     actions.append(a)
                # toc()
                #
                #
                # ####


                tic()
                A_p = self.cached_batch_pi(S_p)
                toc("pi")
                phi_p = self.batch_phi(S_p, A_p)
                phi_p[:, :-1] /= feature_std
                V_p = phi_p.dot(self.theta)
                Q_sample = R + self.gamma * V_p
                # Q_sample = R + self.gamma * self.V(S_p)
                toc("Q_sample")

                tic()
                lsq_A = phi
                lsq_b = Q_sample  # - self.lambda_ * (self.preprocess_action(S, A) ** 2).sum(axis=-1)

                # loss = ((lsq_A.dot(self.theta) - lsq_b) ** 2).mean(axis=0).sum() / 2
                # print('%.2f\t%.2f' % (loss, loss + self.l1_reg * np.abs(self.theta).sum()))

                x = cvxpy.Variable(self.theta.shape[0])
                ### old
                # objective = cvxpy.Minimize((1 / (2 * len(lsq_A))) * cvxpy.sum_squares(lsq_A * x - lsq_b)
                #                            + self.l1_reg * cvxpy.norm(x[:-1], 1))  # no regularization on last weight
                # constraints = [0 <= x, x[:-1] <= 1]  # no upper constraint on last weight
                # constraints.append(cvxpy.sum_entries(x[:-1]) == 1)
                ###
                objective = cvxpy.Minimize((1 / (2 * len(lsq_A))) * cvxpy.sum_squares(lsq_A * x - lsq_b))
                constraints = [0 <= x]
                # constraints = [0 <= x, cvxpy.sum_entries(x[:-1]) == 1]  # no normalization constraint on last weight
                if not self.learn_lambda:
                    constraints.append(x[-1] == self.lambda_)
                prob = cvxpy.Problem(objective, constraints)

                solved = False
                for solver in [None, cvxpy.GUROBI, cvxpy.CVXOPT]:
                    try:
                        prob.solve(solver=solver)
                    except cvxpy.error.SolverError:
                        continue
                    if x.value is None:
                        continue
                    solved = True
                    break
                if not solved:
                    import IPython as ipy;
                    ipy.embed()
                toc("cvxpy")

                theta = np.squeeze(np.array(x.value), axis=1)
                theta[:-1] *= feature_std
                converged = np.allclose(theta, self.theta, atol=1e-3)
                self.theta = theta
                if converged:
                    break
                print('iteration %d of %d' % (iter_, self.max_iters))
                print('\ttheta: %r' % self.theta)
            except:
                import IPython as ipy;
                ipy.embed()
            if converged:
                print('fitted Q-iteration converged in %d iterations' % (iter_ + 1))
            else:
                print('fitted Q-iteration reached maximum number of iterations %d' % self.max_iters)
        return prob.value
