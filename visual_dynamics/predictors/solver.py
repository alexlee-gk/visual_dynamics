from __future__ import division, print_function

import time

import lasagne
import lasagne.layers as L
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import yaml

from visual_dynamics.gui.grid_image_visualizer import GridImageVisualizer
from visual_dynamics.gui.loss_plotter import LossPlotter
from visual_dynamics.utils.config import ConfigObject
from visual_dynamics.utils.generator import DataGenerator, ParallelGenerator
from visual_dynamics.utils.math_utils import OnlineStatistics
from . import layers_theano as LT


class TheanoNetSolver(ConfigObject):
    def __init__(self, train_data_fnames, val_data_fnames=None, data_names=None, data_name_offset_pairs=None,
                 input_names=None, output_names=None,
                 trainable_tags_list=None, batch_size=32, test_iter=10, solver_type='ADAM', test_interval=1000,
                 base_lr=0.001, gamma=1.0, stepsize=1000, display=20, max_iter=10000, momentum=0.9, momentum2=0.999,
                 weight_decay=0.0005, snapshot_interval=1000, snapshot_prefix='', average_loss=10, loss_interval=100,
                 plot_interval=100, iter_=0, losses=None, train_losses=None, val_losses=None, loss_iters=None):
        """
        Args:
            data_names: Iterable of names for the image and velocity inputs in the data files.
            input_names: Iterable of names of the input variables for the image, velocity and next image
            output_names: Iterable of tuples, each being a tuple of prediction and target names of the variables to be
                used for the loss.
            trainable_tags_list: Iterable of tags, each being a dict of tags to filter out the parameters that should
                be trained.
        """
        self.train_data_fnames = train_data_fnames or []
        self.val_data_fnames = val_data_fnames or []
        self.data_names = data_names or ['image', 'action']
        self.data_name_offset_pairs = data_name_offset_pairs or [('image', 0), ('action', 0), ('image', 1)]
        self.input_names = input_names or ['x', 'u', 'x_next']
        self.output_names = output_names or [('x_next_pred', 'x_next')]
        self.trainable_tags_list = trainable_tags_list if trainable_tags_list is not None else [dict(trainable=True)]

        self.batch_size = batch_size
        self.test_iter = test_iter
        self.solver_type = solver_type
        self.test_interval = test_interval
        self.base_lr = base_lr
        self.gamma = gamma
        self.stepsize = stepsize
        self.display = display
        self.max_iter = max_iter
        self.momentum = momentum
        self.momentum2 = momentum2
        self.weight_decay = weight_decay
        self.snapshot_interval = snapshot_interval
        self.snapshot_prefix = snapshot_prefix
        self.average_loss = average_loss
        self.loss_interval = loss_interval
        self.plot_interval = plot_interval

        self.iter_ = iter_
        self.losses = losses or []
        self.train_losses = train_losses or []
        self.val_losses = val_losses or []
        self.loss_iters = loss_iters or []

        self._last_snapshot_iter = None
        self._visualize_loss_num = None

    def get_outputs(self, net, X, U, X_next, preprocessed=False):
        for output_name in self.output_names:
            if not (isinstance(output_name, tuple) or isinstance(output_name, list)) \
                    or len(output_name) != 2:
                raise ValueError("output_names should be iterable of pair tuples")
        names = [name for pair in self.output_names for name in pair]  # flatten output_names
        time_inames_dict = dict()
        for i, name in enumerate(names):
            if isinstance(name, tuple) or isinstance(name, list):
                name, t = name
            else:
                t = 0
            if t not in time_inames_dict:
                time_inames_dict[t] = []
            time_inames_dict[t].append((i, name))
        iouts = []
        for t, inames in time_inames_dict.items():
            inds, names = zip(*inames)
            if t == 0:
                outs = net.predict(names, [X, U], preprocessed=preprocessed)
            elif t == 1:
                outs = net.predict(names, [X_next], preprocessed=preprocessed)
            else:
                raise NotImplementedError("output name with time %d" % t)
            iouts.extend(zip(inds, outs))
        _, outputs = zip(*sorted(iouts))
        return list(zip(outputs[0::2], outputs[1::2]))

    def get_output_vars(self, net, deterministic=False):
        for output_name in self.output_names:
            if not (isinstance(output_name, tuple) or isinstance(output_name, list))\
                    or len(output_name) != 2:
                raise ValueError("output_names should be iterable of pair tuples")
        names = [name for pair in self.output_names for name in pair]  # flatten output_names
        time_inames_dict = dict()
        for i, name in enumerate(names):
            if isinstance(name, tuple) or isinstance(name, list):
                name, t = name
            else:
                t = 0
            if t not in time_inames_dict:
                time_inames_dict[t] = []
            time_inames_dict[t].append((i, name))
        ivars = []
        for t, inames in time_inames_dict.items():
            inds, names = zip(*inames)
            layers = [net.pred_layers[name] for name in names]
            if t == 0:
                vars_ = lasagne.layers.get_output(layers, deterministic=deterministic)
            elif t == 1:
                input_vars = [net.pred_layers[name].input_var for name in self.input_names]
                vars_ = lasagne.layers.get_output(layers, inputs=input_vars[-1], deterministic=deterministic)
            else:
                raise NotImplementedError("output name with time %d" % t)
            ivars.extend(zip(inds, vars_))
        _, output_vars = zip(*sorted(ivars))
        return list(zip(output_vars[0::2], output_vars[1::2]))

    def get_loss_var(self, net, deterministic=False):
        # import IPython as ipy; ipy.embed()
        # output_names = [('x0_next_pred', ('x0', 1)),
        #                 (('x1', 1), 'x1_next_pred')]
        pred_vars, target_vars = zip(*self.get_output_vars(net, deterministic=deterministic))
        loss = 0
        for pred_var, target_var in zip(pred_vars, target_vars):
            loss += ((target_var - pred_var) ** 2).mean(axis=0).sum() / 2.
        params_regularizable = [param for param in net.get_all_params(regularizable=True).values()
                                if param in theano.gof.graph.inputs([loss])]  # exclude params not in computation graph
        param_l2_penalty = lasagne.regularization.apply_penalty(params_regularizable, lasagne.regularization.l2)
        loss += self.weight_decay * param_l2_penalty / 2.
        return loss

    def compile_train_fn(self, net):
        input_vars = [net.pred_layers[name].input_var for name in self.input_names]
        # training loss
        loss = self.get_loss_var(net, deterministic=False)
        # training function
        # params = list(net.get_all_params(trainable=True).values())
        trainable_params_dict = dict()
        for trainable_tags in self.trainable_tags_list:
            # assume trainable=True unless there is an explicit tag of trainable=False
            trainable_tags = dict(trainable_tags)
            trainable_tags.setdefault('trainable', True)
            trainable_params_dict.update(net.get_all_params(**trainable_tags))
        params = list(trainable_params_dict.values())
        nontrainable_params = list(set(net.get_all_params().values()) - set(params))
        print('parameters that are not trainable:\n%r' % nontrainable_params)

        print('training with output names:\n%r' % self.output_names)
        unused_params = [param for param in params if param not in theano.gof.graph.inputs([loss])]
        print('parameters that are trainable but that will not be updated during training:\n%r' % unused_params)
        params = [param for param in params if param in theano.gof.graph.inputs([loss])]
        print('parameters that are trainable and that will be updated during training:\n%r' % params)
        learning_rate_var = theano.tensor.scalar(name='learning_rate')
        if self.solver_type == 'SGD':
            if self.momentum:
                updates = lasagne.updates.momentum(loss, params, learning_rate_var, momentum=self.momentum)
            else:
                updates = lasagne.updates.sgd(loss, params, learning_rate_var)
        elif self.solver_type == 'ADAM':
            updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate_var,
                                           beta1=self.momentum, beta2=self.momentum2)
        else:
            raise ValueError('Unknown solver type %s' % self.solver_type)
        start_time = time.time()
        print("Compiling training function...")
        train_fn = theano.function(input_vars + [learning_rate_var], loss, updates=updates)
        print("... finished in %.2f s" % (time.time() - start_time))
        return train_fn

    def compile_val_fn(self, net):
        input_vars = [net.pred_layers[name].input_var for name in self.input_names]
        # validation loss
        val_loss = self.get_loss_var(net, deterministic=True)
        # validation function
        start_time = time.time()
        print("Compiling validation function...")
        val_fn = theano.function(input_vars, val_loss)
        print("... finished in %.2f s" % (time.time() - start_time))
        return val_fn

    def solve(self, net):
        losses = self.step(self.max_iter - self.iter_, net)
        # save snapshot after the optimization is done if it hasn't already been saved
        if self._last_snapshot_iter != self.iter_:
            self.snapshot(net)

        # display losses after optimization is done
        train_loss, val_loss = losses
        print("Iteration {} of {}".format(self.iter_, self.max_iter))
        print("    training loss = {:.6f}".format(train_loss))
        if val_loss is not None:
            print("    validation loss = {:.6f}".format(val_loss))

    def step(self, iters, net):
        self.standarize(net)

        # training data
        train_data_gen = DataGenerator(self.train_data_fnames,
                                       data_name_offset_pairs=self.data_name_offset_pairs,
                                       transformers=net.transformers,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       dtype=theano.config.floatX)
        train_data_gen = ParallelGenerator(train_data_gen, nb_worker=4)
        validate = self.test_interval and self.val_data_fnames
        if validate:
            # validation data
            val_data_gen = DataGenerator(self.val_data_fnames,
                                         data_name_offset_pairs =self.data_name_offset_pairs ,
                                         transformers=net.transformers,
                                         batch_size=self.batch_size,
                                         shuffle=True,
                                         dtype=theano.config.floatX)
            val_data_gen = ParallelGenerator(val_data_gen,
                                             max_q_size=self.test_iter,
                                             nb_worker=1)

        print("Size of training data is %d" % train_data_gen.size)
        train_fn = self.compile_train_fn(net)
        if validate:
            print("Size of validation data is %d" % val_data_gen.size)
            val_fn = self.compile_val_fn(net)

        if self.plot_interval:
            fig, loss_plotter, image_visualizer = self.loss_visualization_init(validate=validate)

        print("Starting training...")
        stop_iter = self.iter_ + iters
        while self.iter_ < stop_iter:
            if validate and self.iter_ % self.test_interval == 0:
                val_loss = float(sum([val_fn(*next(val_data_gen)) for _ in range(self.test_iter)]) / self.test_iter)
                print("    validation loss = {:.6f}".format(val_loss))

            current_step = self.iter_ // self.stepsize
            learning_rate = self.base_lr * self.gamma ** current_step
            train_batch_data = tuple(next(train_data_gen))
            loss = float(train_fn(*(train_batch_data + (learning_rate,))))
            self.losses.append(loss)

            if self.display and self.iter_ % self.display == 0:
                print("Iteration {} of {}, lr = {}".format(self.iter_, self.max_iter, learning_rate))
                print("    training loss = {:.6f}".format(loss))

            if self.loss_interval and (self.iter_ % self.loss_interval == 0 or
                                       self.snapshot_interval and self.iter_ % self.snapshot_interval == 0):  # update loss plot for snapshot
                average_loss = min(self.average_loss, len(self.losses))
                train_loss = float(sum(self.losses[-average_loss:]) / average_loss)
                self.train_losses.append(train_loss)
                if validate:
                    val_loss = float(sum([val_fn(*next(val_data_gen)) for _ in range(self.test_iter)]) / self.test_iter)
                    self.val_losses.append(val_loss)
                self.loss_iters.append(self.iter_)

            # plot visualization using first datum in batch
            if self.plot_interval and self.iter_ % self.plot_interval == 0:
                batch_data = next(val_data_gen) if validate else train_batch_data
                outputs = self.get_outputs(net, *[datum[0] for datum in batch_data], preprocessed=True)
                self.loss_visualization_update(loss_plotter, image_visualizer, outputs, validate=validate)
                loss_fig_fname = self.get_snapshot_fname('_loss.pdf')
                fig.savefig(loss_fig_fname)

            self.iter_ += 1

            if self.snapshot_interval and self.iter_ % self.snapshot_interval == 0:
                self.snapshot(net)

        average_loss = min(self.average_loss, len(self.losses))
        train_loss = float(sum(self.losses[-average_loss:]) / average_loss)
        if validate:
            val_loss = float(sum([val_fn(*next(val_data_gen)) for _ in range(self.test_iter)]) / self.test_iter)
        else:
            val_loss = None
        return train_loss, val_loss

    def standarize(self, net, aggregating_batch_size=100, check=False):
        start_time = time.time()
        print("Standarizing outputs...")

        # training data (one pass)
        train_data_once_gen = DataGenerator(self.train_data_fnames,
                                            data_name_offset_pairs=self.data_name_offset_pairs,
                                            transformers=net.transformers,
                                            once=True,
                                            batch_size=aggregating_batch_size,
                                            shuffle=False,
                                            dtype=theano.config.floatX)
        train_data_once_gen = ParallelGenerator(train_data_once_gen, nb_worker=1)

        standarize_layers = set()
        for pred_layer in net.pred_layers.values():
            for layer in L.get_all_layers(pred_layer):
                if isinstance(layer, LT.StandarizeLayer):
                    standarize_layers.add(layer)
        standarize_layers = list(standarize_layers)
        standarize_layer_names = [standarize_layer.name for standarize_layer in standarize_layers]
        online_stats = [OnlineStatistics(axis=standarize_layer.shared_axes) for standarize_layer in standarize_layers]

        for batch_data in train_data_once_gen:
            X = batch_data[0]
            outputs = net.predict(standarize_layer_names, [X], preprocessed=True)  # assume the only input is X
            for output, online_stat in zip(outputs, online_stats):
                online_stat.add_data(output)

        for online_stat, standarize_layer in zip(online_stats, standarize_layers):
            standarize_layer.offset.set_value(online_stat.mean.astype(theano.config.floatX))
            standarize_layer.scale.set_value(online_stat.std.astype(theano.config.floatX))

        print("... finished in %.2f s" % (time.time() - start_time))

        if check:
            train_data_once_gen = DataGenerator(self.train_data_fnames,
                                                data_name_offset_pairs=self.data_name_offset_pairs,
                                                transformers=net.transformers,
                                                once=True,
                                                batch_size=aggregating_batch_size,
                                                shuffle=False,
                                                dtype=theano.config.floatX)
            train_data_once_gen = ParallelGenerator(train_data_once_gen, nb_worker=1)

            check_online_stats = [OnlineStatistics(axis=standarize_layer.shared_axes) for standarize_layer in standarize_layers]

            for batch_data in train_data_once_gen:
                X = batch_data[0]
                outputs = net.predict(standarize_layer_names, [X], preprocessed=True)  # assume the only input is X
                for output, online_stat in zip(outputs, check_online_stats):
                    online_stat.add_data(output)

            # check that the standard deviation after standarization is actually 1
            for online_stat in check_online_stats:
                assert np.allclose(online_stat.mean, 0, atol=1e-5)
                assert np.allclose(online_stat.std, 1)

    def loss_visualization_init(self, validate=True):
        fig = plt.figure(figsize=(18, 18), frameon=False, tight_layout=True)
        fig.canvas.set_window_title(self.snapshot_prefix)
        gs = gridspec.GridSpec(1, 3)
        plt.show(block=False)

        loss_labels = ['train', 'train']
        if validate:
            loss_labels += ['val']
        loss_plotter = LossPlotter(fig, gs[0], labels=loss_labels)

        output_labels = [str(output_name) for output_name_pair in self.output_names for output_name in output_name_pair]
        image_visualizer = GridImageVisualizer(fig, gs[1:], rows=len(self.output_names), cols=2, labels=output_labels)
        return fig, loss_plotter, image_visualizer

    def loss_visualization_update(self, loss_plotter, image_visualizer, outputs, validate=True):
        # outputs of single non-batched datum
        all_loss_iters = [None, self.loss_iters]
        all_losses = [self.losses, self.train_losses]
        if validate:
            all_loss_iters += [self.loss_iters]
            all_losses += [self.val_losses]
        loss_plotter.update(all_losses, all_loss_iters)

        vis_outputs = []
        for output_pair in outputs:
            for output in output_pair:
                vis_outputs.append(output)
        image_visualizer.update(vis_outputs)

    def get_snapshot_fname(self, ext):
        return self.snapshot_prefix + '_iter_%s' % str(self.iter_) + ext

    def snapshot(self, net):
        model_fname = self.get_snapshot_fname('_model.yaml')
        print("Saving predictor to file", model_fname)
        with open(model_fname, 'w') as model_file:
            config = net.get_config()
            config['pretrained_fname'] = net.save_model(model_fname)
            yaml.dump(config, model_file, width=float('inf'))
        solver_fname = self.get_snapshot_fname('_solver.yaml')
        print("Saving solver to file", solver_fname)
        with open(solver_fname, 'w') as solver_file:
            self.to_yaml(solver_file)
        self._last_snapshot_iter = self.iter_

    def _get_config(self):
        config = super(TheanoNetSolver, self)._get_config()
        config.update({'train_data_fnames': self.train_data_fnames,
                       'val_data_fnames': self.val_data_fnames,
                       'data_names': self.data_names,
                       'data_name_offset_pairs': self.data_name_offset_pairs,
                       'input_names': self.input_names,
                       'output_names': self.output_names,
                       'trainable_tags_list': self.trainable_tags_list,
                       'batch_size': self.batch_size,
                       'test_iter': self.test_iter,
                       'solver_type': self.solver_type,
                       'test_interval': self.test_interval,
                       'base_lr': self.base_lr,
                       'gamma': self.gamma,
                       'stepsize': self.stepsize,
                       'display': self.display,
                       'max_iter': self.max_iter,
                       'momentum': self.momentum,
                       'momentum2': self.momentum2,
                       'weight_decay': self.weight_decay,
                       'snapshot_interval': self.snapshot_interval,
                       'snapshot_prefix': self.snapshot_prefix,
                       'average_loss': self.average_loss,
                       'loss_interval': self.loss_interval,
                       'plot_interval': self.plot_interval,
                       'iter_': self.iter_,
                       'losses': self.losses,
                       'train_losses': self.train_losses,
                       'val_losses': self.val_losses,
                       'loss_iters': self.loss_iters})
        return config

    def __repr__(self):
        config = self._get_config()
        config.pop('loss_iters', None)
        config.pop('losses', None)
        config.pop('train_losses', None)
        config.pop('val_losses', None)
        return super(TheanoNetSolver, self).__repr__(config)


class BilinearSolver(TheanoNetSolver):
    def __init__(self, train_data_fnames, val_data_fnames=None, data_names=None, data_name_offset_pairs=None,
                 input_names=None, output_names=None,
                 loss_batch_size=32, aggregating_batch_size=1000, num_channel_groups=1,
                 test_iter=10, max_iter=1, weight_decay=0.0005,
                 snapshot_prefix='', average_loss=10, iter_=0):
        self.train_data_fnames = train_data_fnames or []
        self.val_data_fnames = val_data_fnames or []
        self.data_names = data_names or ['image', 'action']
        self.data_name_offset_pairs = data_name_offset_pairs or [('image', 0), ('action', 0), ('image', 1)]
        self.input_names = input_names or ['x', 'u', 'x_next']
        self.output_names = output_names or [('x_next_pred', 'x_next')]

        self.loss_batch_size = loss_batch_size
        self.aggregating_batch_size = aggregating_batch_size
        self.num_channel_groups = num_channel_groups
        self.test_iter = test_iter
        self.max_iter = max_iter
        self.weight_decay = weight_decay
        self.snapshot_prefix = snapshot_prefix
        self.average_loss = average_loss

        self.iter_ = iter_

        self._last_snapshot_iter = None

    def solve(self, net):
        losses = self.step(self.max_iter - self.iter_, net)
        # save snapshot after the optimization is done if it hasn't already been saved
        if self._last_snapshot_iter != self.iter_:
            self.snapshot(net)

        # display losses after optimization is done
        train_loss, val_loss = losses
        print("Iteration {} of {}".format(self.iter_, self.max_iter))
        print("    training loss = {:.6f}".format(train_loss))
        if val_loss is not None:
            print("    validation loss = {:.6f}".format(val_loss))

    def step(self, iters, net):
        self.standarize(net)

        # training data
        train_data_gen = DataGenerator(self.train_data_fnames,
                                       data_name_offset_pairs=self.data_name_offset_pairs,
                                       transformers=net.transformers,
                                       batch_size=self.loss_batch_size,
                                       shuffle=True,
                                       dtype=theano.config.floatX)
        train_data_gen = ParallelGenerator(train_data_gen,
                                           max_q_size=self.average_loss,
                                           nb_worker=4)
        validate = bool(self.val_data_fnames)
        if validate:
            # validation data
            val_data_gen = DataGenerator(self.val_data_fnames,
                                         data_name_offset_pairs=self.data_name_offset_pairs,
                                         transformers=net.transformers,
                                         batch_size=self.loss_batch_size,
                                         shuffle=True,
                                         dtype=theano.config.floatX)
            val_data_gen = ParallelGenerator(val_data_gen,
                                             max_q_size=self.test_iter,
                                             nb_worker=1)

        print("Size of training data is %d" % train_data_gen.size)
        if validate:
            print("Size of validation data is %d" % val_data_gen.size)
        val_fn = self.compile_val_fn(net)  # need val_fn to 'validate' on training data

        print("Starting training...")
        stop_iter = self.iter_ + iters
        while self.iter_ < stop_iter:
            # print losses
            train_loss = float(sum([val_fn(*next(train_data_gen)) for _ in range(self.average_loss)]) / self.average_loss)
            print("Iteration {} of {}".format(self.iter_, self.max_iter))
            print("    training loss = {:.6f}".format(train_loss))
            if validate:
                val_loss = float(sum([val_fn(*next(val_data_gen)) for _ in range(self.test_iter)]) / self.test_iter)
                print("    validation loss = {:.6f}".format(val_loss))

            # ensure outputs_names follow the expected format
            curr_names = []
            bilinear_layers = []
            for output_pair in self.output_names:
                try:
                    try:
                        next_pred_name, (curr_name, offset) = output_pair
                    except ValueError:
                        (curr_name, offset), next_pred_name = output_pair
                        output_pair[:] = output_pair[::-1]
                    if offset != 1:
                        raise Exception
                    next_pred_layer = net.pred_layers[next_pred_name]
                    curr_layer, bilinear_layer = next_pred_layer.input_layers
                    if curr_layer != net.pred_layers[curr_name] or not isinstance(bilinear_layer,
                                                                                  (LT.BilinearLayer,
                                                                                   LT.BilinearChannelwiseLayer)):
                        raise Exception
                    curr_names.append(curr_name)
                    bilinear_layers.append(bilinear_layer)
                except Exception:
                    raise NotImplementedError('bilinear solver for output pair %r' % output_pair)

            for channel_group in range(self.num_channel_groups):
                print("channel group %d" % channel_group)
                # training data (one pass)
                train_data_once_gen = DataGenerator(self.train_data_fnames,
                                                    data_name_offset_pairs=self.data_name_offset_pairs,
                                                    transformers=net.transformers,
                                                    once=True,
                                                    batch_size=self.aggregating_batch_size,
                                                    shuffle=False,
                                                    dtype=theano.config.floatX)
                train_data_once_gen = ParallelGenerator(train_data_once_gen, nb_worker=4)

                start_time = time.time()
                print("Aggregating matrices...")
                Ns = {i: 0 for i in range(len(bilinear_layers))}
                As = {}
                Bs = {}
                post_fit_all = {}
                diff_outputs = {}
                batch_iter = 0
                for X, U, X_next in train_data_once_gen:
                    print("batch %d" % batch_iter)
                    curr_outputs = net.predict(curr_names, [X], preprocessed=True)
                    next_outputs = net.predict(curr_names, [X_next], preprocessed=True)
                    for i, (bilinear_layer, curr_output, next_output) in \
                            enumerate(zip(bilinear_layers, curr_outputs, next_outputs)):
                        c_dim = curr_output.shape[1]
                        assert c_dim % self.num_channel_groups == 0
                        if self.num_channel_groups != 1 and net.bilinear_type != 'channelwise':
                            raise ValueError("num_channel_groups != 1 is only valid for channelwise bilinear layers")
                        if net.bilinear_type == 'share':
                            batch_size = np.prod(curr_output.shape[:2])
                            vel = np.repeat(U[:, None, :], c_dim, axis=1).reshape((batch_size, -1))
                        else:
                            batch_size = curr_output.shape[0]
                            vel = U
                        Ns[i] += batch_size
                        if net.bilinear_type == 'share' or net.bilinear_type == 'full':
                            if i not in As:
                                As[i] = 0
                                Bs[i] = 0
                                diff_outputs[i] = 0
                            curr_output = curr_output.reshape((batch_size, -1))
                            next_output = next_output.reshape((batch_size, -1))
                            diff_output = next_output - curr_output
                            diff_outputs[i] += diff_output.sum(axis=0)
                            A, B, post_fit = bilinear.BilinearFunction.compute_solver_terms(curr_output, vel,
                                                                                            diff_output)
                            As[i] += A
                            Bs[i] += B
                        elif net.bilinear_type == 'channelwise':
                            cg_dim = c_dim // self.num_channel_groups
                            cg_slice = slice(channel_group * cg_dim, (channel_group + 1) * cg_dim)
                            if i not in As:
                                y_dim = bilinear_layer.y_dim
                                u_dim = bilinear_layer.u_dim
                                As[i] = np.zeros((cg_dim, (y_dim+1)*(u_dim+1), (y_dim+1)*(u_dim+1)), dtype=theano.config.floatX)
                                Bs[i] = np.zeros((cg_dim, (y_dim+1)*(u_dim+1), y_dim), dtype=theano.config.floatX)
                                diff_outputs[i] = np.zeros((cg_dim, y_dim), dtype=theano.config.floatX)
                            curr_output = curr_output[:, cg_slice].reshape((batch_size, cg_dim, -1))
                            next_output = next_output[:, cg_slice].reshape((batch_size, cg_dim, -1))
                            diff_output = next_output - curr_output
                            diff_outputs[i] += diff_output.sum(axis=0)
                            for channel in range(cg_slice.start, cg_slice.stop):
                                A, B, post_fit = bilinear.BilinearFunction.compute_solver_terms(curr_output[:, channel % cg_dim, :], vel,
                                                                                                diff_output[:, channel % cg_dim, :])
                                As[i][channel % cg_dim] += A.astype(theano.config.floatX)
                                Bs[i][channel % cg_dim] += B.astype(theano.config.floatX)
                                del A, B
                        else:
                            raise NotImplementedError("aggregating matrices for bilinear_type %s" % net.bilinear_type)
                        if i not in post_fit_all:
                            post_fit_all[i] = post_fit
                    batch_iter += 1
                    # if batch_iter == 2:  # TODO: remove me
                    #     break
                print("... finished in %2.f s" % (time.time() - start_time))

                # mean_diff_output = {}
                # scale_offset_transformer = net.transformers[0].transformers[-1]
                # for i in range(len(diff_outputs)):
                #     mean_diff_output[i] = ((diff_outputs[i] / Ns[i] - scale_offset_transformer.offset) \
                #                            * (1.0 / scale_offset_transformer.scale)).astype(scale_offset_transformer._data_dtype)
                # mean_diff_output_flat = np.concatenate([mean.flatten() for mean in mean_diff_output.values()])
                # for i in range(256):
                #     num = (mean_diff_output_flat == i).sum()
                #     if num > 0:
                #         print('%d: %d' % (i, num))

                start_time = time.time()
                print("Solving linear systems...")
                for i, bilinear_layer in enumerate(bilinear_layers):
                    start_time_single = time.time()
                    post_fit = post_fit_all[i]
                    Q_param, R_param, S_param, b_param = bilinear_layer.get_params()
                    if net.bilinear_type == 'share' or net.bilinear_type == 'full':
                        A = As[i] / (2. * Ns[i])
                        A += self.weight_decay * np.diag([1.] * (len(A) - 1) + [0.])  # don't regularize bias, which is the last one
                        B = Bs[i] / (2. * Ns[i])
                        Q, R, S, b = post_fit(np.linalg.solve(A, B))
                        Q = np.asarray(Q, dtype=theano.config.floatX)
                        R = np.asarray(R, dtype=theano.config.floatX)
                        S = np.asarray(S, dtype=theano.config.floatX)
                        b = np.asarray(b, dtype=theano.config.floatX)
                        Q_param.set_value(Q)
                        R_param.set_value(R)
                        S_param.set_value(S)
                        b_param.set_value(b)
                    elif net.bilinear_type == 'channelwise':
                        c_dim = L.get_output_shape(bilinear_layer)[1]
                        cg_dim = c_dim // self.num_channel_groups
                        cg_slice = slice(channel_group * cg_dim, (channel_group + 1) * cg_dim)
                        # use borrow=True only if the param is shared variable in cpu (i.e. TensorSharedVariable)
                        Q = Q_param.get_value(borrow=isinstance(Q_param, T.sharedvar.TensorSharedVariable))
                        R = R_param.get_value(borrow=isinstance(R_param, T.sharedvar.TensorSharedVariable))
                        S = S_param.get_value(borrow=isinstance(S_param, T.sharedvar.TensorSharedVariable))
                        b = b_param.get_value(borrow=isinstance(b_param, T.sharedvar.TensorSharedVariable))
                        for channel, A, B in (zip(range(cg_slice.start, cg_slice.stop), As[i], Bs[i])):
                            A = A / (2. * Ns[i])
                            A += self.weight_decay * np.diag([1.] * (len(A) - 1) + [0.])  # don't regularize bias, which is the last one
                            B = B / (2. * Ns[i])
                            Q[channel], R[channel], S[channel], b[channel] = post_fit(np.linalg.solve(A, B))
                        Q_param.set_value(Q, borrow=isinstance(Q_param, T.sharedvar.TensorSharedVariable))
                        R_param.set_value(R, borrow=isinstance(R_param, T.sharedvar.TensorSharedVariable))
                        S_param.set_value(S, borrow=isinstance(S_param, T.sharedvar.TensorSharedVariable))
                        b_param.set_value(b, borrow=isinstance(b_param, T.sharedvar.TensorSharedVariable))
                    else:
                        raise NotImplementedError("exact solve for bilinear_type %s" % net.bilinear_type)
                    del Q, R, S, b
                    print("%2.f s" % (time.time() - start_time_single))
                del Ns, As, Bs, post_fit_all, diff_outputs
                print("... finished in %2.f s" % (time.time() - start_time))

            self.iter_ += 1

        # X_next_pred = net.predict('x_next_pred', [X, U], preprocessed=True)
        # fig, axarr = utils.draw_images_callback(list(zip(*[[*X[:10]], [*X_next_pred[:10]], [*X_next[:10]]])),
        #                                         image_transformer=net.transformers['x'].transformers[-1],
        #                                         num=11)

        train_loss = float(sum([val_fn(*next(train_data_gen)) for _ in range(self.average_loss)]) / self.average_loss)
        if validate:
            val_loss = float(sum([val_fn(*next(val_data_gen)) for _ in range(self.test_iter)]) / self.test_iter)
        else:
            val_loss = None
        return train_loss, val_loss

    def _get_config(self):
        config = ConfigObject._get_config(self)
        config.update({'train_data_fnames': self.train_data_fnames,
                       'val_data_fnames': self.val_data_fnames,
                       'data_names': self.data_names,
                       'data_name_offset_pairs': self.data_name_offset_pairs,
                       'input_names': self.input_names,
                       'output_names': self.output_names,
                       'loss_batch_size': self.loss_batch_size,
                       'aggregating_batch_size': self.aggregating_batch_size,
                       'num_channel_groups': self.num_channel_groups,
                       'test_iter': self.test_iter,
                       'max_iter': self.max_iter,
                       'weight_decay': self.weight_decay,
                       'snapshot_prefix': self.snapshot_prefix,
                       'average_loss': self.average_loss,
                       'iter_': self.iter_})
        return config
