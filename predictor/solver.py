import time
import numpy as np
import yaml
import matplotlib.pyplot as plt
import theano
import lasagne
import utils


class TheanoNetSolver(utils.config.ConfigObject):
    def __init__(self, batch_size=32, test_iter=10, solver_type='ADAM', test_interval=1000, base_lr=0.001, gamma=1.0,
                 stepsize=1000, display=20, max_iter=10000, momentum=0.9, momentum2=0.999, weight_decay=0.0005,
                 snapshot_interval=1000, snapshot_prefix='', loss_interval=100, plot_interval=100,
                 iter_=0, losses=None, val_losses=None, loss_iters=None):
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
        self.loss_interval = loss_interval
        self.plot_interval = plot_interval

        self.iter_ = iter_
        self.losses = losses or []
        self.val_losses = val_losses or []
        self.loss_iters = loss_iters or []

    def get_loss_var(self, net, output_names, deterministic=False):
        pred_names, target_names = zip(*output_names)
        output_layers = [net.pred_layers[name] for name in pred_names + target_names]
        output_vars = lasagne.layers.get_output(output_layers, deterministic=deterministic)
        pred_vars, target_vars = output_vars[:len(output_names)], output_vars[len(output_names):]
        loss = 0
        for pred_var, target_var in zip(pred_vars, target_vars):
            loss += ((target_var - pred_var) ** 2).mean(axis=0).sum() / 2.
        params_regularizable = [param for param in net.get_all_params(regularizable=True).values()
                                if param in theano.gof.graph.inputs([loss])]  # exclude params not in computation graph
        param_l2_penalty = lasagne.regularization.apply_penalty(params_regularizable, lasagne.regularization.l2)
        loss += self.weight_decay * param_l2_penalty / 2.
        return loss

    def compile_train_fn(self, net, input_names, output_names):
        input_vars = [net.pred_layers[name].input_var for name in input_names]
        # training loss
        loss = self.get_loss_var(net, output_names, deterministic=False)
        # training function
        params = list(net.get_all_params(trainable=True).values())
        unused_params = [param for param in params if param not in theano.gof.graph.inputs([loss])]
        if unused_params:
            print('parameters %r are unused for training with output names %r' % (unused_params, output_names))
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

    def compile_val_fn(self, net, input_names, output_names):
        input_vars = [net.pred_layers[name].input_var for name in input_names]
        # validation loss
        val_loss = self.get_loss_var(net, output_names, deterministic=True)
        # validation function
        start_time = time.time()
        print("Compiling validation function...")
        val_fn = theano.function(input_vars, val_loss)
        print("... finished in %.2f s" % (time.time() - start_time))
        return val_fn

    def solve(self, net, input_names, output_names, train_data_gen, val_data_gen=None):
        losses = self.step(self.max_iter - self.iter_, net, input_names, output_names, train_data_gen,
                           val_data_gen=val_data_gen)

        # save snapshot after the optimization is done if it hasn't already been saved
        if self.snapshot_interval and self.iter_ % self.snapshot_interval != 0:
            self.snapshot(net)

        # display losses after optimization is done
        try:
            loss, val_loss = losses
        except ValueError:
            loss = losses
            val_loss = None
        print("Iteration {} of {}".format(self.iter_, self.max_iter))
        print("    training loss = {:.6f}".format(loss))
        if val_loss:
            print("    validation loss = {:.6f}".format(val_loss))

    def step(self, iters, net, input_names, output_names, train_data_gen, val_data_gen=None, callback=None):
        train_fn = self.compile_train_fn(net, input_names, output_names)
        validate = self.test_interval and val_data_gen is not None
        if validate:
            val_fn = self.compile_val_fn(net, input_names, output_names)

        print("Starting training...")
        stop_iter = self.iter_ + iters
        while self.iter_ < stop_iter:
            if validate and self.iter_ % self.test_interval == 0:
                val_loss = float(sum([val_fn(*next(val_data_gen)) for _ in range(self.test_iter)]) / self.test_iter)
                print("    validation loss = {:.6f}".format(val_loss))

            current_step = self.iter_ // self.stepsize
            learning_rate = self.base_lr * self.gamma ** current_step
            loss = float(train_fn(*next(train_data_gen), learning_rate))

            if self.display and self.iter_ % self.display == 0:
                print("Iteration {} of {}, lr = {}".format(self.iter_, self.max_iter, learning_rate))
                print("    training loss = {:.6f}".format(loss))

            if self.loss_interval and (self.iter_ % self.loss_interval == 0 or
                                       self.snapshot_interval and self.iter_ % self.snapshot_interval == 0):  # update loss plot for snapshot
                self.losses.append(loss)
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

        if validate:
            val_loss = sum([val_fn(*next(val_data_gen)) for _ in range(self.test_iter)]) / self.test_iter
            losses = (loss, val_loss)
        else:
            losses = loss
        return losses

    def visualize_loss(self, window_title=None):
        plt.ion()
        fig = plt.figure(2)
        plt.cla()
        if window_title is not None:
            fig.canvas.set_window_title(window_title)
        plt.plot(self.loss_iters, self.losses, label='train')
        plt.plot(self.loss_iters, self.val_losses, label='val')
        plt.ylabel('iteration')
        plt.ylabel('loss')
        plt.legend()
        axes = plt.gca()
        ylim = axes.get_ylim()
        ylim = (min(0, ylim[0]), min(2 * np.median([*self.losses, *self.val_losses]), ylim[1]))
        axes.set_ylim(ylim)
        plt.draw()

    def get_snapshot_fname(self, ext):
        return self.snapshot_prefix + '_iter_%s' % str(self.iter_) + ext

    def snapshot(self, net):
        model_fname = self.get_snapshot_fname('_model.yaml')
        with open(model_fname, 'w') as model_file:
            config = net.get_config(model_fname)
            yaml.dump(config, model_file)
        solver_fname = self.get_snapshot_fname('_solver.yaml')
        with open(solver_fname, 'w') as solver_file:
            self.to_yaml(solver_file)
        if self.loss_interval:
            loss_fig_fname = self.get_snapshot_fname('_loss.pdf')
            plt.savefig(loss_fig_fname)

    def get_config(self):
        config = {'class': self.__class__,
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
                  'loss_interval': self.loss_interval,
                  'plot_interval': self.plot_interval}
        return config
