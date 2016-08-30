import argparse
import numpy as np
import cv2
import theano
import theano.tensor as T
import policy
import utils
import utils.transformations as tf
import cvxpy as cvx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as manimation
from gui.grid_image_visualizer import GridImageVisualizer


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('env_fname', type=str, help='config file with environment arguments')
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('--output_dir', '-o', type=str, default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=str, default=None)
    args = parser.parse_args()

    predictor = utils.from_yaml(open(args.predictor_fname))
    env = utils.from_config(predictor.environment_config)

    # TODO: better way of handling policies. reset policies. control policies. mix of them at test time.
    try:
        policy_config = predictor.policy_config
        replace_config = {'env': env}
        try:
            replace_config['target_env'] = env.car_env
        except AttributeError:g
            pass
        # TODO: better way to populate config with existing instances
        pol = utils.from_config(policy_config, replace_config=replace_config)
        target_pol = pol.policies[0]
    except:
        offset = np.array([0., -4., 3.]) * 4
        # offset = np.array([4., 0., 1.]) * 4  # TODO
        target_pol = policy.OgreCameraTargetPolicy(env, env.car_env, 'quad_camera', 'quad', 'car', offset, tightness=1.0)
        # target_pol = policy.OgreCameraTargetPolicy(env, env.quad_camera_node, env.quad_node, env.car_node, env.car_env, offset, tightness=1.0)
        random_pol = policy.RandomPolicy(env.action_space, env.state_space)
        # pol = policy.MixedPolicy(target_pol, random_pol, act_probs=[0.25, 0.75], reset_probs=[1, 0])
        pol = policy.MixedPolicy([target_pol, random_pol], act_probs=[0.5, 0.5], reset_probs=[1, 0])

    class QLearning:
        def __init__(self, predictor, learning_rate=0.1, l1_reg=10.0, gamma=0.9, alpha=1.0, lambda_const=True, theta_init=None):
            self.predictor = predictor
            self.action_transformer = self.predictor.transformers['u']
            self.learning_rate = learning_rate
            self.l1_reg = l1_reg
            self.gamma = gamma  # discount factor
            self.pol = policy.ServoingPolicy(predictor, alpha=alpha)
            self.lambda_const = lambda_const

            self._theta = np.empty(512 + 1)
            self.theta = theta_init if theta_init is not None else np.zeros(512 + 1)

            # if beta_init is None:
            #     beta_init = np.zeros(512)
            # beta_init = beta_init.astype(theano.config.floatX)
            # self.beta_var = theano.shared(beta_init, 'beta')
            # self.theta_var = - T.nnet.nnet.softmax(self.beta_var[None, :])[0]
            # Q_sample_plus_reg_var = T.scalar('Q_sample_plus_reg')
            # phi_var = T.vector('phi')
            # f_var = (Q_sample_plus_reg_var - T.dot(self.theta_var, phi_var)) ** 2
            # grad_var = theano.gradient.grad(f_var, self.beta_var)
            # updates = {self.beta_var: self.beta_var - self.alpha * grad_var}
            # self.train_fn = theano.function([Q_sample_plus_reg_var, phi_var], f_var, updates=updates, allow_input_downcast=True)

        # @property
        # def theta(self):
        #     return self.theta_var.eval()

        @property
        def theta(self):
            return self._theta

        @theta.setter
        def theta(self, theta):
            self._theta[...] = theta
            self.pol.w = np.repeat(self.theta[:512], 32 * 32)
            self.pol.lambda_ = self.theta[-1]

        @property
        def alpha(self):
            return self.pol.alpha

        @alpha.setter
        def alpha(self, alpha):
            self.pol.alpha = alpha

        @property
        def lambda_(self):
            raise NotImplementedError
            return self.pol.lambda_

        @lambda_.setter
        def lambda_(self, lambda_):
            raise NotImplementedError
            self.pol.lambda_ = lambda_

        def preprocess_action(self, s, a):
            image, image_target, image_masked, image_masked_target = np.split(s, [3, 6, 9], axis=-1)
            action = a
            if self.predictor.batch_size(image, action) == 0:
                a_prep = self.action_transformer.preprocess(a)
            else:
                A = a
                a_prep = np.asarray([self.action_transformer.preprocess(a) for a in A])
            return a_prep

        def phi(self, s, a):
            image, image_target, image_masked, image_masked_target = np.split(s, [3, 6, 9], axis=-1)
            action = a

            shape = (512, 32, 32) if self.predictor.batch_size(image, action) == 0 else (-1, 512, 32, 32)
            feature_target = self.predictor.feature(image_target).reshape(shape)
            feature = self.predictor.feature(image).reshape(shape)
            next_feature = self.predictor.next_feature(image, action).reshape(shape)
            # if self.predictor.batch_size(image, action) == 0:
            #     mask = (self.predictor.transformers['x'].transformers[-1].deprocess(self.predictor.transformers['x'].preprocess(image_masked)) != 0).any(axis=2)[None, ...]
            #     target_mask = (self.predictor.transformers['x'].transformers[-1].deprocess(self.predictor.transformers['x'].preprocess(image_masked_target)) != 0).any(axis=2)[None, ...]
            # else:
            #     mask = np.array([(self.predictor.transformers['x'].transformers[-1].deprocess(self.predictor.transformers['x'].preprocess(image_masked)) != 0).any(axis=2) for image_masked in image_masked])[:, None, ...]
            #     target_mask = np.array([(self.predictor.transformers['x'].transformers[-1].deprocess(self.predictor.transformers['x'].preprocess(image_masked_target)) != 0).any(axis=2) for image_masked_target in image_masked_target])[:, None, ...]
            # feature *= mask
            # next_feature *= target_mask
            # feature_target *= target_mask

            y_diff = self.alpha * feature_target + (1 - self.alpha) * feature - next_feature

            # y_diff = self.alpha * self.predictor.feature(image_target) \
            #          + (1 - self.alpha) * self.predictor.feature(image) \
            #          - self.predictor.next_feature(image, action)

            phi = (y_diff ** 2).reshape(shape).sum(axis=(-2, -1))

            phi[np.logical_or((feature_target == 0).sum(axis=(-2, -1)) < 128,
                              (feature == 0).sum(axis=(-2, -1)) < 128)] *= 0.0

            phi = np.concatenate([phi,
                                  (self.preprocess_action(s, a) ** 2).sum(axis=-1).reshape(phi.shape[:-1] + (1,))],
                                  # np.ones(phi.shape[:-1] + (1,))],
                                 axis=-1)
            return phi

        def Q(self, s, a):
            return self.phi(s, a) @ self.theta

        def pi(self, s):
            # argmin_a Q(s, a) = argmin_a theta @ phi(s, a) + lambda a @ a, where theta >= 0
            # argmin_u w @ phi(s, u) + lambda u @ u, where w >= 0
            image, image_target, image_masked, image_masked_target = np.split(s, [3, 6, 9], axis=-1)
            if self.predictor.batch_size(image) == 0:
                self.pol.set_image_target(image_target)
                return self.pol.act(image)
            else:
                S = s
                return np.asarray([self.pi(s) for s in S])

        def V(self, s):
            # min_a Q(s, a)
            return self.Q(s, self.pi(s))

        # def update(self, s, a, r, s_p):
        #     # diff = (r + self.gamma * self.V(s_p)) - self.Q(s, a)
        #     # self.theta += self.alpha * diff * self.phi(s, a)
        #     Q_sample = r + self.gamma * self.V(s_p)
        #     Q_sample_plus_reg = Q_sample + self.lambda_ * (a ** 2).sum()
        #     phi = self.phi(s, a)
        #     f = float(self.train_fn(Q_sample_plus_reg, phi))
        #     print('r: %.2f, f: %.2f' % (r, f))
        #     print(np.sort(self.theta)[:5])
        #     print(np.argsort(self.theta)[:5])

        def update(self, s, a, r, s_p):
            phi = self.phi(s, a)
            Q_sample = r + self.gamma * self.V(s_p)
            diff = Q_sample - phi @ self.theta
            self.theta = self.theta + self.learning_rate * diff * phi
            # self.theta[:-1][self.theta[:-1] < 0] = 0.0  # no constraint on bias
            self.theta[self.theta < 0] = 0.0

        def fit(self, S, A, R, S_p):
            S = np.asarray(S)
            A = np.asarray(A)
            R = np.asarray(R)
            S_p = np.asarray(S_p)

            # from scipy.optimize import lsq_linear
            for i in range(1):
                # Q_sample = []
                # Phi = []
                # for (s, a, r, s_p, image_target) in zip(S, A, R, S_p, image_targets):
                #     self.set_target_obs(image_target)
                #     Q_sample.append(r + self.gamma * self.V(s_p))
                #     Phi.append(self.phi(s, a))
                # Q_sample = np.asarray(Q_sample)
                # Q_sample_plus_reg = Q_sample + self.lambda_ * np.square(A).sum(axis=1)
                # Phi = np.asarray(Phi)

                phi = self.phi(S, A)
                Q_sample = R + self.gamma * self.V(S_p)
                lsq_A = phi
                lsq_b = Q_sample  # - self.lambda_ * (self.preprocess_action(S, A) ** 2).sum(axis=-1)

                loss = ((lsq_A @ self.theta - lsq_b) ** 2).mean(axis=0).sum() / 2
                print('%.2f\t%.2f' % (loss, loss + self.l1_reg * np.abs(self.theta[:512]).sum()))

                # optimize_result = lsq_linear(lsq_A, lsq_b, bounds=(0, 1))
                # self.theta = optimize_result.x
                # print(self.theta)

                x = cvx.Variable(self.theta.shape[0])
                objective = cvx.Minimize((1 / (2 * len(lsq_A))) * cvx.sum_squares(lsq_A * x - lsq_b)
                                         + self.l1_reg * cvx.norm(x[:512], 1))  # no regularization on lambda nor bias
                # constraints = [0 <= x, x <= 1]
                constraints = [0 <= x]  # no constraint on bias
                if self.lambda_const:
                    constraints.append(x[-1] == self.theta[-1])
                prob = cvx.Problem(objective, constraints)
                prob.solve()
                print(prob.value)
                try:
                    self.theta = np.array(x.value)[:, 0]
                except IndexError:
                    import IPython as ipy; ipy.embed()

    import IPython as ipy; ipy.embed()
    q_learning = QLearning(predictor, l1_reg=0.1, alpha=1.0, theta_init=np.array([0.0] * 512 + [0.1]))

    for i in range(2):
        print("i: %d" % i)
        if i > 0:
            pol = policy.MixedPolicy(target_pol, q_learning.pol, act_probs=[0.5, 0.5], reset_probs=[1, 0])
        # pol = policy.MixedPolicy(target_pol, q_learning.pol, act_probs=[0, 1], reset_probs=[1, 0])

        S = []
        A = []
        R = []
        S_p = []

        features = []
        feature_next_preds = []
        masked_images = []
        next_masked_images = []

        done = False
        for traj_iter in range(args.num_trajs):
            print('traj_iter', traj_iter)
            try:
                state = pol.reset()
                env.reset(state)
                image_target, image_masked_target = env.observe()
                # image_target, = predictor.preprocess(image_target)
                # image_masked_target, = predictor.preprocess(image_masked_target)

                for step_iter in range(args.num_steps):
                    obs = env.observe()
                    image = obs[0]  # TODO: use all observations
                    image_masked = obs[1]
                    action = pol.act(image)
                    env.step(action)  # action is updated in-place if needed
                    next_image, next_image_masked = env.observe()

                    feature, feature_next_pred = predictor.predict(['x5', 'x5_next_pred'],
                                                                   image, action)
                    next_feature = predictor.predict('x5', next_image)
                    features.append(feature)
                    feature_next_preds.append(feature_next_pred)
                    masked_images.append(predictor.preprocess(image_masked)[0])
                    next_masked_images.append(predictor.preprocess(next_image_masked)[0])

                    # des_quad_T = target_pol.compute_desired_agent_transform(tightness=1.0)
                    # quad_T = env.quad_node.getTransform()
                    # pos_err = np.linalg.norm(des_quad_T[:3, 3] - quad_T[:3, 3])
                    # angle_err = np.linalg.norm(tf.axis_angle_from_matrix(des_quad_T @ tf.inverse_matrix(quad_T)))
                    # r = (pos_err + angle_err)

                    # r = np.linalg.norm(predictor.preprocess(image_masked_target)[0]
                    #                    - predictor.preprocess(next_image_masked)[0]) ** 2

                    target_T = target_pol.target_node.getTransform()
                    target_to_offset_T = tf.translation_matrix(target_pol.offset)
                    offset_T = target_T @ target_to_offset_T
                    agent_T = target_pol.agent_node.getTransform()
                    agent_to_camera_T = target_pol.camera_node.getTransform()
                    camera_T = agent_T @ agent_to_camera_T
                    pos_err = np.square(offset_T[:3, 3] - camera_T[:3, 3]).sum()
                    # x0 = target_T[:3, 3]
                    # x1 = camera_T[:3, 3]
                    # x2 = x1 + camera_T[:3, 2]
                    # d = np.linalg.norm(np.cross(x0 - x1, x0 - x2)) / np.linalg.norm(x2 - x1)
                    angle = tf.angle_between_vectors(camera_T[:3, 2], camera_T[:3, 3] - target_T[:3, 3])
                    r = 0.1 * pos_err + 1000.0 * angle ** 2
                    print(r, pos_err, angle ** 2)

                    s = np.concatenate([image, image_target, image_masked, image_masked_target], axis=-1)
                    a = action
                    s_p = np.concatenate([next_image, image_target, next_image_masked, image_masked_target], axis=-1)

                    S.append(s)
                    A.append(a)
                    R.append(r)
                    S_p.append(s_p)

                    # q_learning.update(s, a, r, s_p)

                    if args.visualize:
                        env.render()
                        images = [[*predictor.preprocess(image),
                                   *predictor.preprocess(next_image),
                                   *predictor.preprocess(image_target)],
                                  [*predictor.preprocess(image_masked),
                                   *predictor.preprocess(next_image_masked),
                                   *predictor.preprocess(image_masked_target)]]
                        fig, axarr = utils.draw_images_callback(images,
                                                                image_transformer=predictor.transformers['x'].transformers[-1],
                                                                num=8)

                    #     image_next_pred = predictor.predict('x0_next_pred', image, action)
                    #     done, key = utils.visualization.visualize_images_callback(*predictor.preprocess(image),
                    #                                                               image_next_pred,
                    #                                                               image_target,
                    #                                                               image_transformer=
                    #                                                               predictor.transformers['x'].transformers[-1],
                    #                                                               vis_scale=args.vis_scale,
                    #                                                               delay=100)
                    if done:
                        break
                    # if key == 32:  # space
                    #     break
                if done:
                    break
            except KeyboardInterrupt:
                break

        q_learning.fit(S, A, R, S_p)
        print(np.sort(q_learning.theta))


    ### masks
    # from sklearn import linear_model
    #
    # def compute_masks(images):
    #     masks = []
    #     for image in images:
    #         mask = (predictor.transformers['x'].transformers[-1].deprocess(image) != 0).any(axis=2)
    #         masks.append(mask.astype(dtype=int))
    #     return np.array(masks)
    #
    # feature_next_preds_train = np.array(feature_next_preds).reshape(len(feature_next_preds), 512, 32, 32)
    # next_masked_images_train = np.array(next_masked_images)
    # feature_next_preds_train_re = feature_next_preds_train.transpose((0, 2, 3, 1)).reshape((-1, 512))
    # next_masked_images_train_re = next_masked_images_train.transpose((0, 2, 3, 1)).reshape((-1, 3))
    # next_masks_train = compute_masks(next_masked_images_train)
    # next_masks_train_re = next_masks_train.flatten()
    #
    # regr = linear_model.LogisticRegression()
    # regr.fit(feature_next_preds_train_re, next_masks_train_re)
    #
    # pred_next_masks_train_re = regr.predict(feature_next_preds_train_re)
    #
    # pred_next_masks_train_re2 = (feature_next_preds_train_re @ regr.coef_.T > 0).astype(dtype=int)
    #
    #
    #
    #
    # pred_next_masks_train = pred_next_masks_train_re.reshape((100, 1, 32, 32))
    # w = np.squeeze(regr.coef_)

    ### masked images
    feature_next_preds_train = np.array(feature_next_preds).reshape(len(feature_next_preds), 512, 32, 32)
    next_masked_images_train = np.array(next_masked_images)
    feature_next_preds_train_re = feature_next_preds_train.transpose((0, 2, 3, 1)).reshape((-1, 512))
    next_masked_images_train_re = next_masked_images_train.transpose((0, 2, 3, 1)).reshape((-1, 3))
    # weights = np.linalg.lstsq(feature_next_preds_train_re, next_masked_images_train_re)[0]
    V_hat = np.linalg.solve(feature_next_preds_train_re.T @ feature_next_preds_train_re + 0.1 * np.eye(512),
                                  feature_next_preds_train_re.T @ next_masked_images_train_re)

    pred_next_masked_images_train_re = feature_next_preds_train_re @ V_hat
    pred_next_masked_images_train = pred_next_masked_images_train_re.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))

    images = [list(pred_next_masked_images_train[:25]), list(next_masked_images_train[:25]),
              list(pred_next_masked_images_train[25:50]), list(next_masked_images_train[25:50]),
              list(pred_next_masked_images_train[50:75]), list(next_masked_images_train[50:75]),
              list(pred_next_masked_images_train[75:100]), list(next_masked_images_train[75:100])]

    fig, axarr = utils.draw_images_callback(images,
                                            image_transformer=predictor.transformers['x'].transformers[-1],
                                            num=7)

    feature_next_preds_test = np.array(feature_next_preds).reshape(len(feature_next_preds), 512, 32, 32)
    next_masked_images_test = np.array(next_masked_images)
    feature_next_preds_test_re = feature_next_preds_test.transpose((0, 2, 3, 1)).reshape((-1, 512))
    next_masked_images_test_re = next_masked_images_test.transpose((0, 2, 3, 1)).reshape((-1, 3))

    pred_next_masked_images_test_re = feature_next_preds_test_re @ V_hat
    pred_next_masked_images_test = pred_next_masked_images_test_re.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))

    images = [list(pred_next_masked_images_test[:25]), list(next_masked_images_test[:25]),
              list(pred_next_masked_images_test[25:50]), list(next_masked_images_test[25:50]),
              list(pred_next_masked_images_test[50:75]), list(next_masked_images_test[50:75]),
              list(pred_next_masked_images_test[75:100]), list(next_masked_images_test[75:100])]

    fig, axarr = utils.draw_images_callback(images,
                                            image_transformer=predictor.transformers['x'].transformers[-1],
                                            num=7)




    features_train = np.array(features).reshape(len(features), 512, 32, 32)
    masked_images_train = np.array(masked_images)
    features_train_re = features_train.transpose((0, 2, 3, 1)).reshape((-1, 512))
    masked_images_train_re = masked_images_train.transpose((0, 2, 3, 1)).reshape((-1, 3))
    # weights = np.linalg.lstsq(features_train_re, masked_images_train_re)[0]
    V = np.linalg.solve(features_train_re.T @ features_train_re + 0.1 * np.eye(512), features_train_re.T @ masked_images_train_re)

    # np.square(features @ weights - masked_images).sum()
    pred_masked_images_train_re = features_train_re @ V
    pred_masked_images_train = pred_masked_images_train_re.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
    # pred_masked_images_train2 = np.array([(V.T @ feature.reshape((-1, 32 * 32))).reshape((-1, 32, 32)) for feature in features_train])

    images = [list(pred_masked_images_train[:25]), list(masked_images_train[:25]),
              list(pred_masked_images_train[25:50]), list(masked_images_train[25:50]),
              list(pred_masked_images_train[50:75]), list(masked_images_train[50:75]),
              list(pred_masked_images_train[75:100]), list(masked_images_train[75:100])]

    fig, axarr = utils.draw_images_callback(images,
                                            image_transformer=predictor.transformers['x'].transformers[-1],
                                            num=7)

    features_test = np.array(features).reshape(len(features), 512, 32, 32)
    masked_images_test = np.array(masked_images)
    features_test_re = features_test.transpose((0, 2, 3, 1)).reshape((-1, 512))
    masked_images_test_re = masked_images_test.transpose((0, 2, 3, 1)).reshape((-1, 3))

    pred_masked_images_test_re = features_test_re @ V
    pred_masked_images_test = pred_masked_images_test_re.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))

    images = [list(pred_masked_images_test[:25]), list(masked_images_test[:25]),
              list(pred_masked_images_test[25:50]), list(masked_images_test[25:50]),
              list(pred_masked_images_test[50:75]), list(masked_images_test[50:75]),
              list(pred_masked_images_test[75:100]), list(masked_images_test[75:100])]

    fig, axarr = utils.draw_images_callback(images,
                                            image_transformer=predictor.transformers['x'].transformers[-1],
                                            num=7)

    fig = plt.figure(num=5)
    fig.clear()
    for i, rgb in enumerate('rgb'):
       n, bins, patches = plt.hist(reg_weights[:, i], np.linspace(-2, 2, 100), facecolor=rgb, alpha=0.25)
    plt.draw()

    fig = plt.figure(num=6)
    fig.clear()
    for i, rgb in enumerate('rgb'):
       n, bins, patches = plt.hist(weights[:, i], np.linspace(-2, 2, 100), facecolor=rgb, alpha=0.25)
    plt.draw()

    import IPython as ipy; ipy.embed()

    args.num_trajs = 10
    args.num_steps = 100
    import matplotlib.pyplot as plt

    try:
        pol = q_learning.pol
        # w = pol.w.copy()
    except NameError:
        pol = policy.ServoingPolicy(predictor, alpha=1.0, lambda_=0.0)
        # pol = policy.MappedServoingPolicy(predictor, alpha=1.0, lambda_=0.0, V=V, V_hat=V_hat)
        # pol = MappedServoingPolicy(predictor, alpha=1.0, lambda_=0.0, V=V, V_hat=V_hat)

    # pol = policy.ServoingPolicy(predictor, alpha=0.8, lambda_=0.01)
    # theta = np.zeros(512)
    # theta[147] = 1.0
    # theta = q_learning.theta
    # pol.w = np.repeat(theta, 32 * 32)

    # pol.w = None
    # predictor.feature_name = ['x5']
    # predictor.feature_name = ['x']
    # predictor.feature_jacobian_name = ['x5_diff_pred']
    # predictor.next_feature_name = ['x5_next_pred']

    # predictor.feature_jacobian_name = ['x_next_pred']
    # predictor.next_feature_name = ['x_next_pred']

    mode = 'pixel_multiscale0'
    # mode = 'gt_weights'
    if 'pixel' in mode:
    # if mode in ['pixel_all', 'pixel_mask', 'pixel_multiscale']:
        if 'multiscale' in mode:
            predictor.feature_name = ['x0', 'x1', 'x2']
            predictor.feature_jacobian_name = ['x0_next_pred', 'x1_next_pred', 'x2_next_pred']
            predictor.next_feature_name = ['x0_next_pred', 'x1_next_pred', 'x2_next_pred']
        else:
            predictor.feature_name = ['x']
            predictor.feature_jacobian_name = ['x_next_pred']
            predictor.next_feature_name = ['x_next_pred']
        if mode == 'pixel_all':
            pol.w = None
        elif mode == 'pixel_mask':
            target_mask = (predictor.transformers['x'].transformers[-1].deprocess(image_masked_target) != 0).any(axis=2)
            w = np.ones((3, 32, 32))
            w *= target_mask
            pol.w = w.flatten()
        elif mode == 'pixel_multiscale012':
            pol.w = None
        elif mode == 'pixel_multiscale12':
            pol.w = np.repeat([0., 1., 1.], [3 * 32 ** 2, 3 * 16 ** 2, 3 * 8 ** 2])
        elif mode == 'pixel_multiscale2':
            pol.w = np.repeat([0., 0., 1.], [3 * 32 ** 2, 3 * 16 ** 2, 3 * 8 ** 2])
        elif mode == 'pixel_multiscale0':
            pol.w = np.repeat([1., 0., 0.], [3 * 32 ** 2, 3 * 16 ** 2, 3 * 8 ** 2])
        elif mode == 'pixel_multiscale012_weights':
            pol.w = np.repeat([1., 4., 16.], [3 * 32 ** 2, 3 * 16 ** 2, 3 * 8 ** 2])
        elif mode == 'pixel_multiscale012_r':
            pol.w = np.repeat([1., 0., 0., 1., 0., 0., 1., 0., 0.], [32 ** 2] * 3 + [16 ** 2] * 3 + [8 ** 2] * 3)
        elif mode == 'pixel_multiscale012_g':
            pol.w = np.repeat([0., 1., 0., 0., 1., 0., 0., 1., 0.], [32 ** 2] * 3 + [16 ** 2] * 3 + [8 ** 2] * 3)
        elif mode == 'pixel_multiscale012_b':
            pol.w = np.repeat([0., 0., 1., 0., 0., 1., 0., 0., 1.], [32 ** 2] * 3 + [16 ** 2] * 3 + [8 ** 2] * 3)
        else:
            raise ValueError
    else:
        predictor.feature_name = ['x5']
        predictor.feature_jacobian_name = ['x5_diff_pred']
        predictor.next_feature_name = ['x5_next_pred']
        if mode == 'all':
            pol.w = None
        elif mode == 'rl_weights':
            pol.w = np.repeat(q_learning.theta[:512], 32 * 32)
        elif mode == 'top':
            theta = np.zeros(512)
            theta[147] = 1.0
            pol.w = np.repeat(theta, 32 * 32)
        elif mode == 'mask':
            theta = np.zeros(512)
            theta[147] = 1.0
            w = np.repeat(theta, 32 * 32).reshape((512, 32, 32))
            target_mask = (predictor.transformers['x'].transformers[-1].deprocess(image_masked_target) != 0).any(axis=2)
            w[147] *= target_mask
            pol.w = w.flatten()
        elif mode == 'gt_top':
            theta = np.zeros(512)
            theta[226] = 1.0
            pol.w = np.repeat(theta, 32 * 32)
        elif mode == 'gt_weights':
            # pol.w = np.repeat(reg_weights, 32 * 32, axis=0)
            pass
        else:
            raise ValueError

    record = args.visualize and args.visualize.endswith('.mp4')
    if args.visualize:
        fig = plt.figure(figsize=(16, 12), frameon=False, tight_layout=True)
        gs = gridspec.GridSpec(1, 1)
        image_visualizer = GridImageVisualizer(fig, gs[0], 10, rows=4, cols=3)  # hardcoded grid dimensions
        plt.show(block=False)
        import ogre
        visualization_camera_sensor = ogre.PyCameraSensor(env.app.camera, 640, 480)
        if record:
            FFMpegWriter = manimation.writers['ffmpeg']
            writer = FFMpegWriter(fps=1.0 / env.dt)
            writer.setup(fig, args.visualize, fig.dpi)

    np.random.seed(seed=7)
    errors = []
    error_names = ['masked image', 'position', 'rotation']
    error_header_format = "{:>15}" * (1 + len(error_names))
    error_row_format = "{:>15}" + "{:>15.2f}" * len(error_names)
    print('=' * 15 * (1 + len(error_names)))
    print(error_header_format.format("", *error_names))
    done = False
    for traj_iter in range(args.num_trajs):
        print('traj_iter', traj_iter)
        try:
            state = target_pol.reset()
            env.reset(state)
            obs_target = env.observe()
            image_target = obs_target[0]
            # q_learning.set_target_obs(image_target)
            pol.set_image_target(image_target)
            image_target, = predictor.preprocess(image_target)

            for step_iter in range(args.num_steps):
                obs = env.observe()
                image = obs[0]  # TODO: use all observations
                action = pol.act(image)
                # action = pol.gt_act(image, predictor.preprocess(obs[0])[0], predictor.preprocess(obs_target[0])[0])
                # action = pol.gt_act(image, predictor.preprocess(obs[1])[0], predictor.preprocess(obs_target[1])[0])
                env.step(action)  # action is updated in-place if needed
                if args.visualize:
                    next_image = env.observe()[0]
                    env.render()
                    image_next_pred = predictor.predict('x0_next_pred', image, action)
                    # done, key = utils.visualization.visualize_images_callback(*predictor.preprocess(image),
                    #                                                           image_next_pred,
                    #                                                           image_target,
                    #                                                           image_transformer=
                    #                                                           predictor.transformers['x'].transformers[-1],
                    #                                                           vis_scale=args.vis_scale,
                    #                                                           delay=100)

                    image_masked = obs[1]
                    # image_masked_next_pred = predictor.predict('x0_next_pred', image_masked, action)
                    image_masked_target = obs_target[1]
                    image_masked_target, = predictor.preprocess(image_masked_target)
                    next_image_masked, = predictor.preprocess(env.observe()[1])


                    if 'pixel' in mode:
                        features = predictor.predict(predictor.feature_name + predictor.next_feature_name, image, action)
                        next_pred_features = features[3:]
                        features = features[:3]
                        next_features = predictor.predict(predictor.feature_name, next_image)
                        target_features = predictor.predict(predictor.feature_name, image_target, preprocessed=True)
                        images = list(zip(*[features, next_pred_features, next_features, target_features]))
                    else:
                        feature, feature_next_pred = predictor.predict(['x5', 'x5_next_pred'],
                                                                       image, action)
                        next_feature = predictor.predict('x5', next_image)
                        feature_target = predictor.predict('x5', image_target, preprocessed=True)
                        # pol.w *= np.repeat((feature_target.sum(axis=(-2, -1)) != 0.0).astype(np.float), 32 * 32)

                        try:
                            order_inds = np.argsort(q_learning.theta[:512])[::-1]
                        except NameError:
                            # order_inds = np.argsort(pol.w.reshape((512, 32 * 32, 3)).mean(axis=1).sum(axis=1))[::-1]
                            # order_inds = np.argsort(pol.w.reshape((512, 32 * 32, 3)).mean(axis=1)[:, 0])[::-1]  # red
                            order_inds = np.argsort(V[:, 0])[::-1]  # red

                        globals().update(locals())
                        images = [[predictor.preprocess(image)[0], image_next_pred, image_target],
                                  # *[[feature[ind], feature_next_pred[ind], feature_target[ind]] for ind in order_inds[:4]],
                                  [predictor.preprocess(image_masked)[0], next_image_masked, image_masked_target],
                                  [np.tensordot(V.T, feature, axes=1),
                                   np.tensordot(V_hat.T, feature_next_pred, axes=1),
                                   np.tensordot(V.T, feature_target, axes=1)]]

                    import time
                    start_time = time.time()
                    globals().update(locals())
                    images = [image for row_images in images for image in row_images]
                    images = [predictor.transformers['x'].transformers[-1].deprocess(image) for image in images]
                    images.append(visualization_camera_sensor.observe())
                    try:
                        image_visualizer.update(images)
                        if record:
                            writer.grab_frame()
                    except:
                        done = True
                    # fig, axarr = utils.draw_images_callback(images,
                    #                                         image_transformer=predictor.transformers['x'].transformers[-1],
                    #                                         num=10)
                    # print(time.time() - start_time)
                    image_masked_error = np.linalg.norm(image_masked_target - next_image_masked) ** 2
                    target_T = target_pol.target_node.getTransform()
                    target_to_offset_T = tf.translation_matrix(target_pol.offset)
                    offset_T = target_T @ target_to_offset_T
                    agent_T = target_pol.agent_node.getTransform()
                    agent_to_camera_T = target_pol.camera_node.getTransform()
                    camera_T = agent_T @ agent_to_camera_T
                    pos_error = np.square(offset_T[:3, 3] - camera_T[:3, 3]).sum()
                    angle_error = tf.angle_between_vectors(camera_T[:3, 2], camera_T[:3, 3] - target_T[:3, 3])
                    errors.append([image_masked_error, pos_error, angle_error])
                    print(error_row_format.format(str((traj_iter, step_iter)), *np.sqrt([image_masked_error, pos_error, angle_error])))

                    # target_T = target_pol.target_node.getTransform()
                    # target_to_offset_T = tf.translation_matrix(target_pol.offset)
                    # offset_T = target_T @ target_to_offset_T
                    # agent_T = target_pol.agent_node.getTransform()
                    # agent_to_camera_T = target_pol.camera_node.getTransform()
                    # camera_T = agent_T @ agent_to_camera_T
                    # pos_err = np.square(offset_T[:3, 3] - camera_T[:3, 3]).sum()
                    # angle = tf.angle_between_vectors(camera_T[:3, 2], camera_T[:3, 3] - target_T[:3, 3])
                    # r = 0.1 * pos_err + 1000.0 * angle ** 2
                    # print(r, pos_err, angle ** 2)

                    # feature_next = predictor.next_feature(image, action).reshape((512, 32, 32))
                    # feature_target = predictor.feature(image_target, preprocessed=True).reshape((512, 32, 32))
                    # order_inds = np.argsort(q_learning.theta)[::-1]
                    # feature_next = feature_next[order_inds, ...]
                    # feature_target = feature_target[order_inds, ...]
                    # output_pair_arr = np.array([feature_next, feature_target])

                    # plt.ion()
                    # data_min = min(feature_next.min(), feature_target.min())
                    # data_max = max(feature_next.max(), feature_target.max())
                    # plt.subplot(221)
                    # plt.imshow(utils.vis_square(feature_next, data_min=data_min, data_max=data_max))
                    # plt.subplot(222)
                    # plt.imshow(utils.vis_square(feature_target, data_min=data_min, data_max=data_max))
                    # data_min = output_pair_arr.min(axis=(0, 2, 3))[:, None, None]
                    # data_max = output_pair_arr.max(axis=(0, 2, 3))[:, None, None]
                    # plt.subplot(223)
                    # plt.imshow(utils.vis_square(feature_next, data_min=data_min, data_max=data_max))
                    # plt.subplot(224)
                    # plt.imshow(utils.vis_square(feature_target, data_min=data_min, data_max=data_max))
                    # plt.draw()


                    # feature_next = (feature_next * 255.0).astype(np.uint8)
                    # feature_target = (feature_target * 255.0).astype(np.uint8)
                    # done, key = utils.visualization.visualize_images_callback(feature_next,
                    #                                                           feature_target,
                    #                                                           window_name='features7',
                    #                                                           vis_scale=1,
                    #                                                           delay=100)
                if done:
                    break
                # if key == 32:  # space
                #     break
            if done:
                break
        except KeyboardInterrupt:
            break
    print('-' * 15 * (1 + len(error_names)))
    print(error_row_format.format("RMS", *np.sqrt(np.mean(np.array(errors), axis=0))))
    print('%.2f\t%.2f\t%.2f' % tuple([*np.sqrt(np.mean(np.array(errors), axis=0))]))
    if record:
        writer.finish()

    return

    if args.output_dir:
        raise NotImplementedError
        container = utils.container.ImageDataContainer(args.output_dir, 'x')
        container.reserve(env.sensor_names + ['state'], (args.num_trajs, args.num_steps + 1))
        container.reserve('action', (args.num_trajs, args.num_steps))
        container.add_info(environment_config=env.get_config())
    else:
        container = None

    # def down(image, power=2):
    #     if power > 0:
    #         image = (image[:, ::2, ::2] + image[:, 1::2, ::2] + image[:, ::2, 1::2] + image[:, 1::2, 1::2]) / 4.0
    #         image = down(image, power=power - 1)
    #     return image
    #
    # def up(image, power=2):
    #     if power > 0:
    #         image_up = np.zeros((image.shape[0], 2 * image.shape[1], 2 * image.shape[2]))
    #         image_up[:, ::2, ::2] = image
    #         image_up[:, 1::2, ::2] = image
    #         image_up[:, ::2, 1::2] = image
    #         image_up[:, 1::2, 1::2] = image
    #         image = up(image_up, power=power - 1)
    #     return image
    #
    # # TODO: remove the following
    # u_prev = np.random.random((6,))
    # image = (np.random.random((480, 640, 3)) * 255).astype(np.uint8)
    # # import IPython as ipy; ipy.embed()
    # # import theano
    # # import time
    # #
    # # self = predictor
    # # name = 'x_next_pred'
    # # wrt_name = self.control_name
    # # inputs = [image, u_prev]
    # # inputs = self.preprocess(*inputs)
    # # inputs = [input_.astype(theano.config.floatX, copy=False) for input_ in inputs]
    # # output_var, wrt_var = lasagne.layers.get_output([self.pred_layers[name], self.pred_layers[wrt_name]],
    # #                                                 deterministic=True)
    # # output_shape, wrt_shape = lasagne.layers.get_output_shape([self.pred_layers[name], self.pred_layers[wrt_name]])
    # # if len(wrt_shape) != 2 or wrt_shape[0] not in (1, None):
    # #     raise ValueError("the shape of the wrt variable is %r but the"
    # #                      "variable should be two-dimensional with the"
    # #                      "leading axis being a singleton or None")
    # # output_dim = np.prod(output_shape[1:])
    # # _, wrt_dim = wrt_shape
    # #
    # # ### ORIGINAL
    # # jac_var, _ = theano.scan(
    # #     lambda eval_points, output_var, wrt_var: theano.gradient.Rop(output_var, wrt_var, eval_points),
    # #     sequences=np.eye(wrt_dim)[:, None, :],
    # #     non_sequences=[output_var.flatten(), wrt_var])
    # #
    # # input_vars = [input_var for input_var in self.input_vars if input_var in theano.gof.graph.inputs([jac_var])]
    # # start_time = time.time()
    # # print("Compiling jacobian function...")
    # # jac_fn = theano.function(input_vars, jac_var)
    # # print("... finished in %.2f s" % (time.time() - start_time))
    # # start_time = time.time()
    # # jac = jac_fn(*[input_[None, ...] for input_ in inputs])
    # # lop_rop2, lop2 = jac.dot(jac.T), jac.dot(y)
    # # print(time.time() - start_time)
    # #
    # # print("Compiling jacobian function...")
    # # jac_output_fn = theano.function(input_vars, [jac_var, output_var])
    # # print("... finished in %.2f s" % (time.time() - start_time))
    # # start_time = time.time()
    # # for i in range(10):
    # #     jac, output = jac_output_fn(*[input_[None, ...] for input_ in inputs])
    # # print((time.time() - start_time) / 10)
    # #
    # # # output_fn = theano.function(input_vars, output_var)
    # # start_time = time.time()
    # # for i in range(10):
    # #     jac = jac_fn(*[input_[None, ...] for input_ in inputs])
    # #     output = output_fn(*[input_[None, ...] for input_ in inputs])
    # # print((time.time() - start_time) / 10)
    # #
    # #
    # # ### SINGLETON
    # # x_var = T.tensor3('x')
    # # u_var = T.vector('u')
    # # X_var, U_var = self.input_vars
    # # o_var = theano.clone(output_var, replace={X_var: x_var[None, :, :, :], U_var: u_var[None, :]})
    # # o_var = theano.clone(output_var, replace={X_var: X_var[0:1, :, :, :], U_var: U_var[0:1, :]})
    # #
    # # rop_var = theano.gradient.Rop(o_var.flatten(), U_var, np.eye(wrt_dim)[0:1])
    # # start_time = time.time()
    # # print("Compiling rop function...")
    # # rop_fn = theano.function([X_var, U_var], rop_var)
    # # # rop_fn = theano.function([x_var, u_var], rop_var)
    # # print("... finished in %.2f s" % (time.time() - start_time))
    # # start_time = time.time()
    # # rop = rop_fn(*[input_[None, ...] for input_ in inputs])
    # # # rop = rop_fn(*inputs)
    # # print(time.time() - start_time)
    # #
    # # ### SERVOING POLICY
    # #
    # # mv_var = T.matrix('mv')
    # # rop_var = theano.gradient.Rop(output_var.flatten(), wrt_var, mv_var)
    # # start_time = time.time()
    # # print("Compiling rop function...")
    # # rop_fn = theano.function([X_var, U_var, mv_var], rop_var)
    # # print("... finished in %.2f s" % (time.time() - start_time))
    # # start_time = time.time()
    # # rop = rop_fn(*[input_[None, ...] for input_ in inputs], np.eye(6).astype(theano.config.floatX)[0:1, :])
    # # print(time.time() - start_time)
    # #
    # #
    # #
    # # y = np.random.random((inputs[0].shape)).flatten().astype(theano.config.floatX)
    # # y_var = T.vector('v')
    # # lop_var = theano.gradient.Lop(output_var.flatten(), wrt_var, y_var)
    # # start_time = time.time()
    # # print("Compiling lop function...")
    # # lop_fn = theano.function([X_var, U_var, y_var], lop_var)
    # # print("... finished in %.2f s" % (time.time() - start_time))
    # # start_time = time.time()
    # # lop = lop_fn(*[input_[None, ...] for input_ in inputs], y)
    # # print(time.time() - start_time)
    # #
    # # lop_rop_var, _ = theano.scan(
    # #     lambda eval_points, output_var, wrt_var: theano.gradient.Lop(output_var,
    # #                                                                  wrt_var,
    # #                                                                  theano.gradient.Rop(output_var, wrt_var, eval_points)),
    # #     sequences=np.eye(wrt_dim)[:, None, :],
    # #     non_sequences=[output_var.flatten(), wrt_var])
    # # start_time = time.time()
    # # print("Compiling lop_rop function...")
    # # lop_rop_fn = theano.function([X_var, U_var], lop_rop_var)
    # # print("... finished in %.2f s" % (time.time() - start_time))
    # #
    # # start_time = time.time()
    # # print("Compiling lop_rop function...")
    # # lop_rop_fn = theano.function([X_var, U_var, y_var], [lop_rop_var, lop_var])
    # # print("... finished in %.2f s" % (time.time() - start_time))
    # # start_time = time.time()
    # # lop_rop, lop = lop_rop_fn(*[input_[None, ...] for input_ in inputs], y)
    # # print(time.time() - start_time)
    # #
    # # u = self.alpha * np.linalg.solve(JW.T.dot(J) + self.lambda_ * np.eye(J.shape[1]), JW.T.dot(self.y_target - y))
    #
    #
    #
    #
    #
    # # J = predictor.feature_jacobian(image, u_prev)
    # import time
    # # start_time = time.time()
    # # J = predictor.jacobian('x_next_pred', predictor.control_name, image, u_prev)
    # # print(time.time() - start_time)
    # # # 0.2
    #
    # start_time = time.time()
    # predictor.feature_jacobian_name = 'x_next_pred'
    # J_feat, y = predictor.feature_jacobian(image, u_prev)
    # print(time.time() - start_time)
    # print(J_feat)
    # print(y)
    # print(J_feat.shape)
    # print(y.shape)
    #
    # start_time = time.time()
    # J_fwd = predictor.jacobian('x_next_pred', predictor.control_name, image, u_prev, mode='fwd')
    # print(time.time() - start_time)
    # print(J_fwd)
    # print(np.allclose(J_fwd, J_feat))
    # print(np.abs(J_fwd - J_feat).max())
    #
    # start_time = time.time()
    # J_batched = predictor.jacobian('x_next_pred', predictor.control_name, image, u_prev, mode='batched')
    # print(time.time() - start_time)
    # print(J_batched)
    # print(np.allclose(J_fwd, J_batched))
    # print(np.abs(J_fwd - J_batched).max())
    #
    # start_time = time.time()
    # J_rev = predictor.jacobian('x_next_pred', predictor.control_name, image, u_prev, mode='reverse')
    # print(time.time() - start_time)
    # print(J_rev)
    # print(np.allclose(J_fwd, J_rev))
    # print(np.abs(J_fwd - J_rev).max())
    #
    # # start_time = time.time()
    # # J0 = predictor.jacobian('x_next_pred', predictor.control_name, image, u_prev)
    # # print(time.time() - start_time)
    # # # 0.2
    #
    # start_time = time.time()
    # for i in range(10):
    #     J0, y = predictor.feature_jacobian(image, u_prev)
    # print((time.time() - start_time) / 10.0)
    #
    # start_time = time.time()
    # for i in range(10):
    #     J1 = predictor.jacobian('x_next_pred', predictor.control_name, image, u_prev, mode='fwd')
    # print((time.time() - start_time) / 10.0)
    #
    # start_time = time.time()
    # for i in range(10):
    #     J2 = predictor.jacobian('x_next_pred', predictor.control_name, image, u_prev, mode='batched')
    # print((time.time() - start_time) / 10.0)
    #
    # start_time = time.time()
    # for i in range(10):
    #     J3 = predictor.jacobian('x_next_pred', predictor.control_name, image, u_prev, mode='reverse')
    # print((time.time() - start_time) / 10.0)
    # return
    #
    # import IPython as ipy; ipy.embed()

    # TODO: option for policy
    offset = np.array([0., -4., 3.]) * 4
    target_pol = policy.OgreCameraTargetPolicy(env, env.quad_camera_node, env.quad_node, env.car_node, offset, tightness=1.0)
    random_pol = policy.RandomPolicy(env.action_space, env.state_space)
    pol = policy.MixedPolicy([target_pol, random_pol], act_probs=[0.25, 0.75], reset_probs=[1, 0])
    # pol = policy.MixedPolicy([target_pol, random_pol], act_probs=[0, 1], reset_probs=[1, 0])
    # pol = policy.MixedPolicy([target_pol, random_pol], act_probs=[1, 0], reset_probs=[1, 0])
    done = False
    for traj_iter in range(args.num_trajs):
        print('traj_iter', traj_iter)
        try:
            image_preds = []
            env.reset(pol)
            for step_iter in range(args.num_steps):
                state = env.state
                obs = env.observe()
                action = pol.act(obs)
                env.step(action)  # action is updated in-place if needed
                if container:
                    container.add_datum(traj_iter, step_iter, state=state, action=action,
                                        **dict(zip(env.sensor_names, obs)))
                    if step_iter == (args.num_steps-1):
                        obs_next = env.observe()
                        container.add_datum(traj_iter, step_iter + 1, state=env.state,
                                            **dict(zip(env.sensor_names, obs_next)))
                if args.visualize:
                    env.render()

                    image = obs[0]
                    image, action = predictor.preprocess(image, action)
                    if not image_preds:
                        image_preds.append(image)
                    image_one_pred, image_pred = predictor.predict('x_next_pred',
                                                                   np.array([image, image_preds[-1]]),
                                                                   np.array([action]*2),
                                                                   preprocessed=True)
                    image_preds.append(image_pred)
                    image_prev = image
                    image = env.observe()[0]
                    image, = predictor.preprocess(image)
                    vis_images = image_prev, image, image_one_pred, image_pred
                    done, key = utils.visualization.visualize_images_callback(*vis_images,
                                                                              image_transformer=predictor.transformers['x'].transformers[-1],
                                                                              vis_scale=args.vis_scale,
                                                                              delay=100)

                    # image = obs[0]
                    # image, action = predictor.preprocess(image, action)
                    # image = down(image)
                    # if not image_preds:
                    #     image_preds.append(image)
                    # image_one_pred = predictor.predict('x_next_pred', up(image), action, preprocessed=True)
                    # image_pred = predictor.predict('x_next_pred', up(image_preds[-1]), action, preprocessed=True)
                    # image_preds.append(image_pred)
                    # image_prev = image
                    # image = env.observe()[0]
                    # image, = predictor.preprocess(image)
                    # image = down(image)
                    # vis_images = image_prev, image, image_one_pred, image_pred
                    # done, key = utils.visualization.visualize_images_callback(*vis_images,
                    #                                                           image_transformer=
                    #                                                           predictor.transformers[
                    #                                                               'x'].transformers[-1],
                    #                                                           vis_scale=args.vis_scale,
                    #                                                           delay=100)
                    # if step_iter == 0 and traj_iter == 0:
                    #     import IPython as ipy; ipy.embed()



                    # import IPython as ipy; ipy.embed()

                    # image, = predictor.preprocess(obs[0])
                    # image = down(image)
                    # num_steps = 10
                    # images = [image]
                    # image_one_preds = [image]
                    # image_preds = [image]
                    # for t in range(num_steps):
                    #     image = env.observe()[0]
                    #     env.step(action)
                    #     image, action = predictor.preprocess(image, action)
                    #     image = down(image)
                    #     images.append(image)
                    #     image_one_preds.append(predictor.predict('x_next_pred', up(images[t]), action, preprocessed=True))
                    #     image_preds.append(predictor.predict('x_next_pred', up(image_preds[t]), action, preprocessed=True))
                    #     vis_images = images[-1], image_one_preds[-1], image_preds[-1]
                    #     done, key = utils.visualization.visualize_images_callback(*vis_images,
                    #                                                               image_transformer=
                    #                                                               predictor.transformers[
                    #                                                                   'x'].transformers[-1],
                    #                                                               vis_scale=args.vis_scale,
                    #                                                               delay=100)

                    # vis_images = np.concatenate([images, image_preds, image_one_preds], axis=2)
                    # done, key = utils.visualization.visualize_images_callback(*vis_images,
                    #                                                           image_transformer=predictor.transformers['x'].transformers[-1],
                    #                                                           vis_scale=args.vis_scale,
                    #                                                           delay=100)

                    # image = obs[0]
                    # image_next = env.observe()[0]
                    # feature_names = ['x']
                    # feature_next_pred_names = ['x_next_pred']
                    # if 'x5' in predictor.feature_name:  # TODO: hack
                    #     feature_names += ['x5']
                    #     feature_next_pred_names += ['x5_next_pred']
                    # # TODO: predict: handle case when action is provided but it isn't necessary
                    # image_next, *feature_nexts = predictor.predict(feature_names, image_next)
                    # image_next_pred, *feature_next_preds = predictor.predict(feature_next_pred_names, image, action)
                    #
                    # image_next = predictor.transformers['x'].deprocess(image_next)
                    # image_next_pred = predictor.transformers['x'].deprocess(image_next_pred)
                    #
                    # feature_next, feature_next_pred = feature_nexts[0], feature_next_preds[0]
                    # output_pair_arr = np.array([feature_next, feature_next_pred])
                    # data_min = output_pair_arr.min(axis=(0, 2, 3))[:, None, None]
                    # data_max = output_pair_arr.max(axis=(0, 2, 3))[:, None, None]
                    # feature_next = utils.vis_square(feature_next, data_min=data_min, data_max=data_max)
                    # feature_next_pred = utils.vis_square(feature_next_pred, data_min=data_min, data_max=data_max)
                    # feature_next = (feature_next * 255.0).astype(np.uint8)
                    # feature_next_pred =  (feature_next_pred * 255.0).astype(np.uint8)
                    # feature_next = np.tile(feature_next[:, :, None], (1, 1, 3))
                    # feature_next_pred = np.tile(feature_next_pred[:, :, None], (1, 1, 3))
                    # vis_image = np.zeros((2 * 759, 32 + 759, 3), dtype=np.uint8)
                    # vis_image[:32, :32] = image_next_pred
                    # vis_image[759:759+32, :32] = image_next
                    # vis_image[:759, 32:32+759] = feature_next_pred
                    # vis_image[759:759+759, 32:32+759] = feature_next
                    # done, key = utils.visualization.visualize_images_callback(vis_image,
                    #                                                           vis_scale=args.vis_scale,
                    #                                                           delay=100)
                    if done:
                        break
                    if key == 32:  # space
                        import IPython as ipy; ipy.embed()
                        continue
                        if container is None:
                            break
                        else:
                            print("Can't skip to next trajectory when a container is being used. Ignoring key press.")
            if done:
                break
        except KeyboardInterrupt:
            break
    env.close()
    if args.visualize:
        cv2.destroyAllWindows()
    if container:
        container.close()


if __name__ == "__main__":
    main()
