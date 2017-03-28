import os
import time
from collections import OrderedDict

import h5py
import lasagne
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

from visual_dynamics.utils import iter_util
try:
    from visual_dynamics.utils import visualization_theano
except ImportError:
    visualization_theano = None
from visual_dynamics.utils.config import ConfigObject, from_yaml
from visual_dynamics.utils.container import MultiDataContainer
from visual_dynamics.utils.transformer import Transformer
from . import predictor


class TheanoNetPredictor(predictor.NetPredictor, ConfigObject):
    def __init__(self, build_net, input_names, input_shapes, transformers=None, name=None, pretrained_fname=None,
                 solvers=None, environment_config=None, policy_config=None, **kwargs):
        """
        Args:
            build_net: Function that builds the net and returns a dict the network layers, which should contain at least
                all the root layers of the network. Different keys of this dict can map to the same layer.
            input_names: Iterable of names of the input variables (e.g. the image and velocity)
            input_shapes: Iterable of shapes for the image and velocity inputs.
            transformers: Iterable of transformers for the image and velocity inputs.
            name: Name of this net predictor. Defaults to class name.
            pretrained_fname: File name of h5 file with parameters to initialize the parameters of this net. The
                file name could also be an iteration number, in which case the file with the corresponding number
                in the default snapshot directory is used.
            kwargs: Optional arguments that are passed to build_net.
        """
        self.build_net = build_net
        self._kwargs = kwargs
        self.__dict__.update(kwargs)
        predictor.NetPredictor.__init__(self, input_names, input_shapes, transformers=transformers, name=name, backend='theano')
        self.pred_layers = self.build_net(self.preprocessed_input_shapes, **kwargs)
        for pred_layer in list(self.pred_layers.values()):
            self.pred_layers.update((layer.name, layer) for layer in lasagne.layers.get_all_layers(pred_layer) if layer.name is not None)
        self.input_vars = [self.pred_layers[input_name].input_var for input_name in self.input_names if input_name in self.pred_layers]
        # layer_name_aliases = [('x0', 'x'), ('x', 'x0'), ('x0_next', 'x_next'), ('x_next', 'x0_next'), ('x0', 'image_curr'), ('x0_next', 'image_next'), ('x0_next_pred', 'image_next_pred')]
        # for name, name_alias  in layer_name_aliases:
        #     if name in self.pred_layers and name_alias not in self.pred_layers:
        #         self.pred_layers[name_alias] = self.pred_layers[name]
        # self.pred_vars = OrderedDict(zip(self.pred_layers.keys(),
        #                              lasagne.layers.get_output(self.pred_layers.values(), deterministic=True)))
        print("Network %s has %d parameters" % (self.name, lasagne.layers.count_params(self.pred_layers.values())))
        self.transformers = transformers or [Transformer() for _ in self.preprocessed_input_shapes]
        self.pred_fns = {}
        self.jac_fns = {}
        if pretrained_fname is not None:
            try:
                iter_ = int(pretrained_fname)
                pretrained_fname = '%s_iter_%d_model.h5' % (self.get_snapshot_prefix(), iter_)
            except ValueError:
                pretrained_fname = pretrained_fname.replace('.yaml', '.h5')
            self.copy_from(pretrained_fname)
        # draw net and save to file
        net_graph_fname = os.path.join(self.get_model_dir(), 'net_graph.png')
        if visualization_theano is not None:
            visualization_theano.draw_to_file(self.get_all_layers(), net_graph_fname, output_shape=True, verbose=True)
        self._draw_fig_num = None
        # self.draw()
        self.solvers = solvers or []
        self.environment_config = environment_config
        self.policy_config = policy_config

    def train(self, solver_or_fname):
        if isinstance(solver_or_fname, str):
            with open(solver_or_fname) as solver_file:
                solver = from_yaml(solver_file)
        else:
            solver = solver_or_fname
        self.solvers.append(solver)
        data_fnames = solver.train_data_fnames + solver.val_data_fnames
        with MultiDataContainer(data_fnames) as data_container:
            environment_config = data_container.get_info('environment_config')
            policy_config = data_container.get_info('policy_config')
        if self.environment_config:
            if self.environment_config != environment_config:
                raise ValueError('environment config mismatch across trainings:\n%r\n%r'
                                 % (self.environment_config, environment_config))
        else:
            self.environment_config = environment_config
        if self.policy_config:
            if self.policy_config != policy_config:
                raise ValueError('policies config mismatch across trainings:\n%r\n%r'
                                 % (self.policy_config, policy_config))
        else:
            self.policy_config = policy_config
        solver.solve(self)

    def _compile_pred_fn(self, names):
        output_layers = [self.pred_layers[name] for name in names]
        pred_vars = lasagne.layers.get_output(output_layers, deterministic=True)
        input_vars = [input_var for input_var in self.input_vars if input_var in
                      theano.gof.graph.inputs(pred_vars)]
        start_time = time.time()
        print("Compiling prediction function...")
        pred_fn = theano.function(input_vars, pred_vars)
        print("... finished in %.2f s" % (time.time() - start_time))
        return pred_fn

    def predict(self, name_or_names, inputs, preprocessed=False):
        names = tuple(iter_util.flatten_tree(name_or_names))
        batch_size = self.batch_size(inputs, preprocessed=preprocessed)
        if not preprocessed:
            inputs = self.preprocess(inputs)
        inputs = [input_.astype(theano.config.floatX, copy=False) for input_ in inputs]
        if batch_size == 0:
            inputs = [input_[None, ...] for input_ in inputs]
        pred_fn = self.pred_fns.get(names) or self.pred_fns.setdefault(names, self._compile_pred_fn(names))
        preds = pred_fn(*inputs)
        if batch_size == 0:
            preds = [np.squeeze(pred, 0) for pred in preds]
        return iter_util.unflatten_tree(name_or_names, preds)

    def _get_jacobian_var(self, names, wrt_name, mode=None):
        """
        Returns the jacobian expressions and the respective outputs for a single
        data point. Assumes that the inputs being passed have a single leading
        dimension (i.e. batch_size=1).
        """
        output_wrt_vars = lasagne.layers.get_output([self.pred_layers[name] for name in (names + (wrt_name,))], deterministic=True)
        output_vars, wrt_var = output_wrt_vars[:-1], output_wrt_vars[-1]
        output_wrt_shapes = lasagne.layers.get_output_shape([self.pred_layers[name] for name in (names + (wrt_name,))])
        output_shapes, wrt_shape = output_wrt_shapes[:-1], output_wrt_shapes[-1]
        output_dim = sum([np.prod(output_shape[1:]) for output_shape in output_shapes])
        if len(wrt_shape) != 2 or wrt_shape[0] not in (1, None):
            raise ValueError("the shape of the wrt variable is %r but the"
                             "variable should be two-dimensional with the"
                             "leading axis being a singleton or None")
        _, wrt_dim = wrt_shape
        if mode is None:
            if wrt_dim < output_dim:
                mode = 'forward'
            else:
                mode = 'reverse'
        if mode in ('fwd', 'forward'):
            # compute jacobian as multiple Rop jacobian-vector product gradients for each index of wrt variable
            output_var = T.concatenate([output_var.flatten() for output_var in output_vars])
            jac_var, _ = theano.scan(lambda eval_points, output_var, wrt_var: theano.gradient.Rop(output_var, wrt_var, eval_points),
                                     sequences=np.eye(wrt_dim)[:, None, :],
                                     non_sequences=[output_var, wrt_var])
            jac_var = jac_var.T
        elif mode == 'batched':
            # same as forward mode but using batch computations as opposed to scan
            # see https://github.com/Theano/Theano/issues/4087
            output_var = T.concatenate([T.flatten(output_var, outdim=2) for output_var in output_vars], axis=1)
            jac_var = theano.gradient.Rop(output_var, wrt_var, np.eye(wrt_dim))
            input_vars = [input_var for input_var in self.input_vars if input_var in theano.gof.graph.inputs([jac_var])]
            rep_dict = {input_var: T.repeat(input_var, wrt_dim, axis=0) for input_var in input_vars}
            jac_var = theano.clone(jac_var, replace=rep_dict)
            jac_var = jac_var.T
        elif mode in ('rev', 'reverse'):
            # compute jacobian as multiple Lop vector-jacobian product gradients for each index of the output variable
            output_var = T.concatenate([output_var.flatten() for output_var in output_vars])
            jac_var = theano.gradient.jacobian(output_var, wrt_var)
            jac_var = jac_var[:, 0, :]
        elif mode == 'linear':
            jac_vars = []
            for output_var in output_vars:
                input_vars = [input_var for input_var in self.input_vars if
                              input_var in theano.gof.graph.inputs([output_var])]
                # using tensordot to multiply ones with the input_var seems to be faster than using repeat
                # rep_dict = {input_var: T.repeat(input_var, wrt_dim + 1, axis=0)
                #             for input_var in input_vars if input_var != wrt_var}
                rep_dict = {input_var: T.tensordot(T.ones((wrt_dim + 1, 1)), input_var, axes=1)
                            for input_var in input_vars if input_var != wrt_var}
                rep_dict[wrt_var] = np.r_[np.zeros((1, wrt_dim)), np.eye(wrt_dim)].astype(theano.config.floatX)
                rep_output_var = theano.clone(output_var, replace=rep_dict)
                jac_var = rep_output_var[1:] - rep_output_var[0]
                jac_var = jac_var.reshape((wrt_dim, -1, 1)).T
                jac_vars.append(jac_var)
        else:
            raise ValueError('mode can only be fwd, forward, rev, reverse, batched or linear, but %r was given' % mode)
        if mode != 'linear':
            split_inds = np.r_[0, np.cumsum([np.prod(output_shape[1:]) for output_shape in output_shapes])]
            jac_vars = [jac_var[start_ind:end_ind].reshape((1, -1, wrt_dim)) for (start_ind, end_ind) in zip(split_inds[:-1], split_inds[1:])]
        return jac_vars, output_vars

    def _compile_jacobian_fn(self, names, wrt_name, ret_outputs=False, mode=None):
        jac_vars, output_vars = self._get_jacobian_var(names, wrt_name, mode=mode)
        if ret_outputs:
            all_vars = jac_vars + output_vars
        else:
            all_vars = jac_vars
        input_vars = [input_var for input_var in self.input_vars if input_var in theano.gof.graph.inputs(all_vars)]
        start_time = time.time()
        print("Compiling jacobian function...")
        jac_fn = theano.function(input_vars, all_vars, on_unused_input='warn')
        print("... finished in %.2f s" % (time.time() - start_time))
        return jac_fn

    def jacobian(self, name_or_names, wrt_name, inputs, preprocessed=False, ret_outputs=False, mode=None):
        names = tuple(iter_util.flatten_tree(name_or_names))
        batch_size = self.batch_size(inputs, preprocessed=preprocessed)
        if not preprocessed:
            inputs = self.preprocess(inputs)
        inputs = [input_.astype(theano.config.floatX, copy=False) for input_ in inputs]
        if batch_size in (0, 1):
            if batch_size == 0:
                inputs = [input_[None, :] for input_ in inputs]
            jac_fn_args = (names, wrt_name, ret_outputs, mode)
            jac_fn = self.jac_fns.get(jac_fn_args) or \
                self.jac_fns.setdefault(jac_fn_args, self._compile_jacobian_fn(*jac_fn_args))
            preds = jac_fn(*inputs)
            if batch_size == 0:
                preds = [np.squeeze(pred, 0) for pred in preds]
        else:
            batched_preds = [self.jacobian(names, wrt_name, single_inputs,
                                           preprocessed=True, ret_outputs=ret_outputs,
                                           mode=mode) for single_inputs in zip(*inputs)]
            if ret_outputs:
                batched_preds = [sum(preds, []) for preds in batched_preds]
            preds = [np.array(pred) for pred in zip(*batched_preds)]
        if ret_outputs:
            return iter_util.unflatten_tree([name_or_names, name_or_names], preds)
        else:
            return iter_util.unflatten_tree(name_or_names, preds)

    def get_all_layers(self):
        layers = []
        for pred_layer in list(self.pred_layers.values()):
            layers.extend(lasagne.layers.get_all_layers(pred_layer))
        return lasagne.utils.unique(layers)

    def get_all_params(self, **tags):
        params = lasagne.layers.get_all_params(self.pred_layers.values(), **tags)
        params_dict = OrderedDict([(param.name, param) for param in params])
        if len(params_dict) != len(params):
            raise ValueError('parameters do not have unique names')
        return params_dict

    def get_all_param_values(self, **tags):
        params_dict = self.get_all_params(**tags)
        param_values_dict = OrderedDict([(name, param.get_value()) for (name, param) in params_dict.items()])
        return param_values_dict

    def set_all_param_values(self, param_values_dict, **tags):
        params_dict = self.get_all_params(**tags)
        set_param_names = []
        skipped_param_names = []

        for name, value in param_values_dict.items():
            try:
                param = params_dict[name]
            except KeyError:
                skipped_param_names.append(name)
                continue
            if param.get_value().shape != value.shape:
                raise ValueError('mismatch: parameter has shape %r but value to set has shape %r' %
                                 (param.get_value().shape, value.shape))
            param.set_value(value)
            set_param_names.append(name)
        if skipped_param_names:
            print('skipped parameters with names: %r' % skipped_param_names)
            print('set parameters with names: %r' % set_param_names)

    def save_model(self, model_fname):
        model_fname = model_fname.replace('.yaml', '.h5')
        all_param_values = self.get_all_param_values()
        print("Saving model parameters to file", model_fname)
        with h5py.File(model_fname, 'w') as h5_file:
            for name, value in all_param_values.items():
                h5_file.create_dataset(name, data=value)
        return model_fname

    def copy_from(self, model_fname):
        print("Copying model parameters from file", model_fname)
        with h5py.File(model_fname, 'r') as h5_file:
            param_values = dict()
            for name in h5_file.keys():
                param_values[name] = h5_file[name][:]
        param_values = OrderedDict([(name, value.astype(theano.config.floatX, copy=False)) for (name, value) in param_values.items()])
        self.set_all_param_values(param_values)

    def draw(self):
        net_graph_fname = os.path.join(self.get_model_dir(), 'net_graph.png')
        with open(net_graph_fname, 'rb') as net_graph_file:
            image = plt.imread(net_graph_file)
        plt.ion()
        fig = plt.figure(num=self._draw_fig_num, figsize=(10.*image.shape[1]/image.shape[0], 10.), tight_layout=True)
        self._draw_fig_num = fig.number
        plt.axis('off')
        fig.canvas.set_window_title('Net graph for %s' % self.name)
        plt.imshow(image)
        plt.draw()

    def _get_config(self):
        config = ConfigObject._get_config(self)
        config.update({'build_net': self.build_net,
                       'input_names': self.input_names,
                       'input_shapes': self.input_shapes,
                       'transformers': self.transformers,
                       'name': self.name,
                       'solvers': self.solvers,
                       'environment_config': self.environment_config,
                       'policy_config': self.policy_config})
        config.update(self._kwargs)
        return config


class TheanoNetFeaturePredictor(TheanoNetPredictor, predictor.FeaturePredictor):
    def __init__(self, build_net, input_names, input_shapes, feature_name,
                 next_feature_name, control_name, feature_jacobian_name=None,
                 transformers=None, name=None, pretrained_fname=None, **kwargs):
        TheanoNetPredictor.__init__(
            self, build_net, input_names, input_shapes,
            transformers=transformers, name=name,
            pretrained_fname=pretrained_fname, **kwargs)
        predictor.FeaturePredictor.__init__(
            self, input_names, input_shapes, feature_name, next_feature_name,
            control_name, feature_jacobian_name=feature_jacobian_name,
            transformers=transformers, name=name)

    def feature_jacobian(self, inputs, preprocessed=False, mode=None):
        assert len(inputs) == 2
        if self.feature_jacobian_name:
            jac, next_feature = \
                self.predict([self.feature_jacobian_name, self.next_feature_name],
                             inputs, preprocessed=preprocessed)
        else:
            jac, next_feature = self.jacobian(self.next_feature_name, self.control_name,
                                              inputs, preprocessed=preprocessed,
                                              ret_outputs=True, mode=mode)
        return jac, next_feature

    def _get_config(self):
        config = dict(TheanoNetPredictor._get_config(self))
        config.update(predictor.FeaturePredictor._get_config(self))
        return config
