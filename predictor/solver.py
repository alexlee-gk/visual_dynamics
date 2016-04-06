import time
import numpy as np
import yaml
import matplotlib.pyplot as plt
import theano
import lasagne
import utils
import bilinear
from . import layers_theano


class TheanoNetSolver(utils.config.ConfigObject):
    def __init__(self, train_data_fnames, val_data_fname=None, data_names=None, input_names=None, output_names=None,
                 batch_size=32, test_iter=10, solver_type='ADAM', test_interval=1000, base_lr=0.001, gamma=1.0,
                 stepsize=1000, display=20, max_iter=10000, momentum=0.9, momentum2=0.999, weight_decay=0.0005,
                 snapshot_interval=1000, snapshot_prefix='', average_loss=10, loss_interval=100, plot_interval=100,
                 iter_=0, losses=None, train_losses=None, val_losses=None, loss_iters=None):
        """
        Args:
            data_names: Iterable of names for the image and velocity inputs in the data files.
            input_names: Iterable of names of the input variables for the image, velocity and next image
            output_names: Iterable of tuples, each being a tuple of prediction and target names of the variables to be
                used for the loss.
        """
        self.train_data_fnames = train_data_fnames or []
        self.val_data_fname = val_data_fname
        self.data_names = data_names or ['image', 'vel']
        self.input_names = input_names or ['x', 'u', 'x_next']
        self.output_names = output_names or [('x_next_pred', 'x_next')]

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
                outs = net.predict(names, X, U, preprocessed=preprocessed)
            elif t == 1:
                outs = net.predict(names, X_next, preprocessed=preprocessed)
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
        params = list(net.get_all_params(trainable=True).values())
        unused_params = [param for param in params if param not in theano.gof.graph.inputs([loss])]
        if unused_params:
            print('parameters %r are unused for training with output names %r' % (unused_params, self.output_names))
        params = [param for param in params if param in theano.gof.graph.inputs([loss])]
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
        train_fn = theano.function([*input_vars, learning_rate_var], loss, updates=updates)
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
        # training data
        train_data_gen = utils.generator.ImageVelDataGenerator(*self.train_data_fnames,
                                                               data_names=self.data_names,
                                                               transformers=net.transformers,
                                                               batch_size=self.batch_size,
                                                               shuffle=True,
                                                               dtype=theano.config.floatX)
        train_data_gen = utils.generator.ParallelGenerator(train_data_gen, nb_worker=4)
        validate = self.test_interval and self.val_data_fname is not None
        if validate:
            # validation data
            val_data_gen = utils.generator.ImageVelDataGenerator(self.val_data_fname,
                                                                 data_names=self.data_names,
                                                                 transformers=net.transformers,
                                                                 batch_size=self.batch_size,
                                                                 shuffle=True,
                                                                 dtype=theano.config.floatX)
            val_data_gen = utils.generator.ParallelGenerator(val_data_gen,
                                                             max_q_size=self.test_iter,
                                                             nb_worker=1)

        print("Size of training data is %d" % train_data_gen.size())
        train_fn = self.compile_train_fn(net)
        if validate:
            print("Size of validation data is %d" % val_data_gen.size())
            val_fn = self.compile_val_fn(net)

        print("Starting training...")
        stop_iter = self.iter_ + iters
        while self.iter_ < stop_iter:
            if validate and self.iter_ % self.test_interval == 0:
                val_loss = float(sum([val_fn(*next(val_data_gen)) for _ in range(self.test_iter)]) / self.test_iter)
                print("    validation loss = {:.6f}".format(val_loss))

            current_step = self.iter_ // self.stepsize
            learning_rate = self.base_lr * self.gamma ** current_step
            loss = float(train_fn(*next(train_data_gen), learning_rate))
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
                self.visualize_loss(net.name)

            # plot visualization using first datum in batch
            if self.plot_interval and self.iter_ % self.plot_interval == 0:
                net.plot(*[datum[0] for datum in next(val_data_gen)], preprocessed=True)

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

    def visualize_loss(self, window_title=None):
        plt.ion()
        fig = plt.figure(num=self._visualize_loss_num)
        self._visualize_loss_num = fig.number
        plt.cla()
        if window_title is not None:
            fig.canvas.set_window_title(window_title)
        plt.plot(self.loss_iters, self.train_losses, label='train')
        if self.val_losses:
            plt.plot(self.loss_iters, self.val_losses, label='val')
        plt.ylabel('iteration')
        plt.ylabel('loss')
        plt.legend()
        axes = plt.gca()
        ylim = axes.get_ylim()
        ylim = (min(0, ylim[0]), min(2 * np.median([*self.train_losses, *self.val_losses]), ylim[1]))
        axes.set_ylim(ylim)
        plt.draw()

    def get_snapshot_fname(self, ext):
        return self.snapshot_prefix + '_iter_%s' % str(self.iter_) + ext

    def snapshot(self, net):
        model_fname = self.get_snapshot_fname('_model.yaml')
        print("Saving predictor to file", model_fname)
        with open(model_fname, 'w') as model_file:
            config = net.get_config(model_fname)
            yaml.dump(config, model_file)
        solver_fname = self.get_snapshot_fname('_solver.yaml')
        print("Saving solver to file", solver_fname)
        with open(solver_fname, 'w') as solver_file:
            self.to_yaml(solver_file)
        try:
            if self.loss_interval:
                loss_fig_fname = self.get_snapshot_fname('_loss.pdf')
                plt.savefig(loss_fig_fname)
        except AttributeError:
            pass
        self._last_snapshot_iter = self.iter_

    def get_config(self):
        config = {'class': self.__class__,
                  'train_data_fnames': self.train_data_fnames,
                  'val_data_fname': self.val_data_fname,
                  'data_names': self.data_names,
                  'input_names': self.input_names,
                  'output_names': self.output_names,
                  'batch_size': self.batch_size,
                  'test_iter': self.test_iter,
                  'solver_type': self.solver_type,
                  'test_interval': self.test_interval,
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
                  'loss_iters': self.loss_iters}
        return config
