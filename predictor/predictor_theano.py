import os
import numpy as np
import pickle
from collections import OrderedDict
import time
import theano
import theano.tensor as T
import lasagne
import matplotlib.pyplot as plt
from . import predictor
import utils


class TheanoNetPredictor(predictor.NetPredictor, utils.config.ConfigObject):
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
            pretrained_fname: File name of pickle file with parameters to initialize the parameters of this net. The
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
        self.input_vars = [self.pred_layers[name].input_var for name in self.input_names]
        # layer_name_aliases = [('x0', 'x'), ('x', 'x0'), ('x0_next', 'x_next'), ('x_next', 'x0_next'), ('x0', 'image_curr'), ('x0_next', 'image_next'), ('x0_next_pred', 'image_next_pred')]
        # for name, name_alias  in layer_name_aliases:
        #     if name in self.pred_layers and name_alias not in self.pred_layers:
        #         self.pred_layers[name_alias] = self.pred_layers[name]
        # self.pred_vars = OrderedDict(zip(self.pred_layers.keys(),
        #                              lasagne.layers.get_output(self.pred_layers.values(), deterministic=True)))
        print("Network %s has %d parameters" % (self.name, lasagne.layers.count_params(self.pred_layers.values())))
        self.transformers = transformers or [utils.Transformer() for _ in self.preprocessed_input_shapes]
        self.pred_fns = {}
        self.jac_fns = {}
        if pretrained_fname is not None:
            try:
                iter_ = int(pretrained_fname)
                pretrained_fname = '%s_iter_%d_model.pkl' % (self.get_snapshot_prefix(), iter_)
            except ValueError:
                pretrained_fname = pretrained_fname.replace('.yaml', '.pkl')
            self.copy_from(pretrained_fname)
        # draw net and save to file
        net_graph_fname = os.path.join(self.get_model_dir(), 'net_graph.png')
        utils.visualization_theano.draw_to_file(self.get_all_layers(), net_graph_fname, output_shape=True, verbose=True)
        self._draw_fig_num = None
        # self.draw()
        self.solvers = solvers or []
        self.environment_config = environment_config
        self.policy_config = policy_config

    def train(self, solver_fname):
        with open(solver_fname) as solver_file:
            solver = utils.config.from_yaml(solver_file)
        if not solver.snapshot_prefix:
            solver_file_base_name = os.path.splitext(os.path.split(solver_fname)[1])[0]
            solver.snapshot_prefix = self.get_snapshot_prefix(solver_file_base_name)
        self.solvers.append(solver)
        data_fnames = solver.train_data_fnames + ([solver.val_data_fname] or [])
        with utils.container.MultiDataContainer(data_fnames) as data_container:
            environment_config = data_container.get_info('environment_config')
            policy_config = data_container.get_info('policy_config')
        if self.environment_config:
            if self.environment_config != environment_config:
                raise ValueError('environment config mismatch across trainings:\n%r\n%r'
                                 % (self.environment_config, environment_config))
        else:
            self.environment_config = environment_config
        if self.policy_config:
            if self.policy_config!= policy_config:
                raise ValueError('policy config mismatch across trainings:\n%r\n%r'
                                 % (self.policy_config, policy_config))
        else:
            self.policy_config = policy_config
        solver.solve(self)

    def _compile_pred_fn(self, name_or_names):
        if isinstance(name_or_names, str):
            output_layer_or_layers = self.pred_layers[name_or_names]
        else:
            output_layer_or_layers = [self.pred_layers[name] for name in name_or_names]
        pred_var_or_vars = lasagne.layers.get_output(output_layer_or_layers, deterministic=True)
        input_vars = [input_var for input_var in self.input_vars if input_var in
                      theano.gof.graph.inputs(pred_var_or_vars if isinstance(pred_var_or_vars, list) else [pred_var_or_vars])]
        start_time = time.time()
        print("Compiling prediction function...")
        pred_fn = theano.function(input_vars, pred_var_or_vars)
        print("... finished in %.2f s" % (time.time() - start_time))
        return pred_fn

    def predict(self, name_or_names, *inputs, **kwargs):
        kwargs = utils.python3.get_kwargs(dict(preprocessed=False), kwargs)
        preprocessed = kwargs['preprocessed']
        if not isinstance(name_or_names, str):
            name_or_names = tuple(name_or_names)
        batch_size = self.batch_size(*inputs, preprocessed=preprocessed)
        if not preprocessed:
            inputs = self.preprocess(*inputs)
        inputs = [input_.astype(theano.config.floatX, copy=False) for input_ in inputs]
        if batch_size == 0:
            inputs = [input_[None, ...] for input_ in inputs]
        pred_fn = self.pred_fns.get(name_or_names) or self.pred_fns.setdefault(name_or_names, self._compile_pred_fn(name_or_names))
        pred_or_preds = pred_fn(*inputs)
        if batch_size == 0:
            if isinstance(name_or_names, str):
                pred_or_preds = np.squeeze(pred_or_preds, 0)
            else:
                pred_or_preds = [np.squeeze(pred, 0) for pred in pred_or_preds]
        return pred_or_preds

    def _get_jacobian_var(self, name, wrt_name, mode=None):
        """
        Returns a matrix of a single jacobian. Assumes that the inputs being
        passed have a single leading dimension (i.e. batch_size=1).
        """
        output_var, wrt_var = lasagne.layers.get_output([self.pred_layers[name], self.pred_layers[wrt_name]], deterministic=True)
        output_shape, wrt_shape = lasagne.layers.get_output_shape([self.pred_layers[name], self.pred_layers[wrt_name]])
        if len(wrt_shape) != 2 or wrt_shape[0] not in (1, None):
            raise ValueError("the shape of the wrt variable is %r but the"
                             "variable should be two-dimensional with the"
                             "leading axis being a singleton or None")
        output_dim = np.prod(output_shape[1:])
        _, wrt_dim = wrt_shape
        if mode is None:
            if wrt_dim < output_dim:
                mode = 'forward'
            else:
                mode = 'reverse'
        if mode in ('fwd', 'forward'):
            # compute jacobian as multiple Rop jacobian-vector product gradients for each index of wrt variable
            jac_var, _ = theano.scan(lambda eval_points, output_var, wrt_var: theano.gradient.Rop(output_var, wrt_var, eval_points),
                                     sequences=np.eye(wrt_dim)[:, None, :],
                                     non_sequences=[output_var.flatten(), wrt_var])
            jac_var = jac_var.T
        elif mode == 'batched':
            # same as forward mode but using batch computations as opposed to scan
            # see https://github.com/Theano/Theano/issues/4087
            jac_var = theano.gradient.Rop(T.flatten(output_var, outdim=2), wrt_var, np.eye(wrt_dim))
            input_vars = [input_var for input_var in self.input_vars if input_var in theano.gof.graph.inputs([jac_var])]
            rep_dict = {input_var: T.repeat(input_var, wrt_dim, axis=0) for input_var in input_vars}
            jac_var = theano.clone(jac_var, replace=rep_dict)
            jac_var = jac_var.T
        elif mode in ('rev', 'reverse'):
            # compute jacobian as multiple Lop vector-jacobian product gradients for each index of the output variable
            jac_var = theano.gradient.jacobian(output_var.flatten(), wrt_var)
            jac_var = jac_var[:, 0, :]
        else:
            raise ValueError('mode can only be fwd, forward, rev, reverse or batched, but %r was given' % mode)
        return jac_var

    def _compile_jacobian_fn(self, name, wrt_name, other_names=(), mode=None):
        jac_var = self._get_jacobian_var(name, wrt_name, mode=mode)
        other_vars = lasagne.layers.get_output([self.pred_layers[name] for name in other_names], deterministic=True)
        input_vars = [input_var for input_var in self.input_vars if input_var in theano.gof.graph.inputs([jac_var] + other_vars)]
        start_time = time.time()
        print("Compiling jacobian function...")
        jac_fn = theano.function(input_vars, [jac_var] + other_vars if other_vars else jac_var, on_unused_input='warn')
        print("... finished in %.2f s" % (time.time() - start_time))
        return jac_fn

    def jacobian(self, name, wrt_name, *inputs, **kwargs):
        kwargs = utils.python3.get_kwargs(dict(preprocessed=False,
                                               other_names=(),
                                               mode=None),
                                          kwargs)
        preprocessed, other_names, mode = [kwargs[key] for key in ['preprocessed', 'other_names', 'mode']]
        batch_size = self.batch_size(*inputs, preprocessed=preprocessed)
        if not preprocessed:
            inputs = self.preprocess(*inputs)
        inputs = [input_.astype(theano.config.floatX, copy=False) for input_ in inputs]
        if batch_size in (0, 1):
            if batch_size == 0:
                inputs = [input_[None, :] for input_ in inputs]
            jac_fn_key = (name, wrt_name) + tuple(other_names) + (mode,)
            jac_fn = self.jac_fns.get(jac_fn_key) or \
                     self.jac_fns.setdefault(jac_fn_key,
                                             self._compile_jacobian_fn(name, wrt_name, other_names=other_names, mode=mode))
            jac_and_others = jac_fn(*inputs) if other_names else [jac_fn(*inputs)]
            jac, others = jac_and_others[0], jac_and_others[1:]
            output_shape, wrt_shape = lasagne.layers.get_output_shape([self.pred_layers[name], self.pred_layers[wrt_name]])
            output_dim = np.prod(output_shape[1:])
            wrt_dim = np.prod(wrt_shape[1:])
            assert jac.shape == (output_dim, wrt_dim)
            if batch_size != 0:
                jac = jac.reshape((1, output_dim, wrt_dim))
            if batch_size == 0:
                others = [np.squeeze(other, 0) for other in others]
            if other_names:
                return (jac,) + tuple(others)
            else:
                return jac
        else:
            if other_names:
                preds = zip(*[self.jacobian(name, wrt_name, *single_inputs, other_names=other_names, mode=mode) for single_inputs in zip(inputs)])
                return (np.asarray(pred) for pred in preds)
            else:
                return np.asarray([self.jacobian(name, wrt_name, *single_inputs, other_names=other_names, mode=mode) for single_inputs in zip(inputs)])

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
        model_fname = model_fname.replace('.yaml', '.pkl')
        print("Saving model parameters to file", model_fname)
        with open(model_fname, 'wb') as model_file:
            all_param_values = self.get_all_param_values()
            pickle.dump(all_param_values, model_file, protocol=pickle.HIGHEST_PROTOCOL)
        return model_fname

    def copy_from(self, model_fname):
        print("Copying model parameters from file", model_fname)
        with open(model_fname, 'rb') as model_file:
            param_values = pickle.load(model_file)
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
        config = utils.ConfigObject._get_config(self)
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
    def __init__(self, build_net, input_names, input_shapes, transformers=None, name=None, pretrained_fname=None,
                 feature_name=None, next_feature_name=None, control_name=None, **kwargs):
        TheanoNetPredictor.__init__(
            self, build_net, input_names, input_shapes, transformers=transformers, name=name,
            pretrained_fname=pretrained_fname, **kwargs)
        predictor.FeaturePredictor.__init__(
            self, input_names, input_shapes, transformers=transformers, name=name,
            feature_name=feature_name, next_feature_name=next_feature_name, control_name=control_name)

    def feature_jacobian(self, X, U, preprocessed=False, mode=None):
        if isinstance(self.next_feature_name, (list, tuple)):
            jac, next_feature = zip(*[self.jacobian(next_feature_name, self.control_name,
                                                    X, U, preprocessed=preprocessed,
                                                    other_names=(next_feature_name,),
                                                    mode=mode) for next_feature_name in self.next_feature_name])
        else:
            jac, next_feature = self.jacobian(self.next_feature_name, self.control_name,
                                              X, U, preprocessed=preprocessed,
                                              other_names=(self.next_feature_name,),
                                              mode=mode)
        return jac, next_feature

    def _get_config(self):
        config = dict(TheanoNetPredictor._get_config(self))
        config.update(predictor.FeaturePredictor._get_config(self))
        return config
