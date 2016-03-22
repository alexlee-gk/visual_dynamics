import os
import numpy as np
import pickle
from collections import OrderedDict
import time
import theano
import lasagne
from . import predictor
from .solver import TheanoNetSolver
import bilinear
import utils
from .layers_theano import set_layer_param_tags


class TheanoNetPredictor(predictor.NetPredictor, utils.config.ConfigObject):
    def __init__(self, build_net, input_shapes, input_names=None, transformers=None, name=None, pretrained_fname=None, **kwargs):
        """
        Args:
            build_net: Function that builds the net and returns a dict the network layers, which should contain at least
                all the root layers of the network. Different keys of this dict can map to the same layer.
            input_shapes: Iterable of shapes for the image and velocity inputs.
            input_names: Iterable of names of the input variables for the image and velocity
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
        self.input_names = input_names or ['x', 'u']
        predictor.NetPredictor.__init__(self, input_shapes, transformers=transformers, name=name, backend='theano')
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
        if pretrained_fname is not None and not pretrained_fname.endswith('.pkl'):
            pretrained_fname = self.get_snapshot_prefix() + '_iter_%s' % str(pretrained_fname) + '_model.pkl'
        if pretrained_fname is not None:
            self.copy_from(pretrained_fname)

    def train(self, *train_data_fnames, val_data_fname=None, data_names=None, input_names=None, output_names=None,
              solver_fname=None, train_nb_worker=4, val_nb_worker=1):
        """
        Args:
            data_names: Iterable of names for the image and velocity inputs in the data files.
            input_names: Iterable of names of the input variables for the image, velocity and next image
            output_names: Iterable of tuples, each being a tuple of prediction and target names of the variables to be
                used for the loss.
        """
        if solver_fname is not None:
            with open(solver_fname) as yaml_string:
                solver = utils.config.from_yaml(yaml_string)
        else:
            solver = TheanoNetSolver()
        solver.snapshot_prefix = self.get_snapshot_prefix(solver.snapshot_prefix)

        # training data
        data_names = data_names or ['image', 'vel']
        train_data_gen = utils.generator.ImageVelDataGenerator(*train_data_fnames,
                                                               data_names=data_names,
                                                               transformers=self.transformers,
                                                               batch_size=solver.batch_size,
                                                               shuffle=True,
                                                               dtype=theano.config.floatX)
        train_data_gen = utils.generator.ParallelGenerator(train_data_gen,
                                                           nb_worker=train_nb_worker)
        if solver.test_interval and val_data_fname is not None:
            # validation data
            val_data_gen = utils.generator.ImageVelDataGenerator(val_data_fname,
                                                                 data_names=data_names,
                                                                 transformers=self.transformers,
                                                                 batch_size=solver.batch_size,
                                                                 shuffle=True,
                                                                 dtype=theano.config.floatX)
            val_data_gen = utils.generator.ParallelGenerator(val_data_gen,
                                                             max_q_size=solver.test_iter,
                                                             nb_worker=val_nb_worker)
        else:
            val_data_gen = None

        input_names = input_names or ['x', 'u', 'x_next']
        output_names = output_names or [('x_next_pred', 'x_next')]
        solver.solve(self, input_names, output_names, train_data_gen, val_data_gen=val_data_gen)

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

    def predict(self, name_or_names, *inputs, preprocessed=False):
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

    def _compile_jacobian_fn(self, name, wrt_name):
        output_var, wrt_var = lasagne.layers.get_output([self.pred_layers[name], self.pred_layers[wrt_name]], deterministic=True)
        jac_var = theano.gradient.jacobian(output_var.flatten(), wrt_var)
        input_vars = [input_var for input_var in self.input_vars if input_var in theano.gof.graph.inputs([jac_var])]
        start_time = time.time()
        print("Compiling jacobian function...")
        jac_fn = theano.function(input_vars, jac_var)
        print("... finished in %.2f s" % (time.time() - start_time))
        return jac_fn

    def jacobian(self, name, wrt_name, *inputs, preprocessed=False):
        batch_size = self.batch_size(*inputs, preprocessed=preprocessed)
        if not preprocessed:
            inputs = self.preprocess(*inputs)
        inputs = [input_.astype(theano.config.floatX, copy=False) for input_ in inputs]
        if batch_size in (0, 1):
            if batch_size == 0:
                inputs = [input_[None, :] for input_ in inputs]
            jac_fn = self.jac_fns.get((name, wrt_name)) or self.jac_fns.setdefault((name, wrt_name), self._compile_jacobian_fn(name, wrt_name))
            jac = jac_fn(*inputs)
            jac = jac.swapaxes(0, 1)
            jac = jac.reshape((*jac.shape[:2], -1))
            if batch_size == 0:
                jac = np.squeeze(jac, axis=0)
            return jac
        else:
            return np.asarray([self.jacobian(name, wrt_name, *single_inputs) for single_inputs in zip(inputs)])

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
        for name, value in param_values_dict.items():
            try:
                param = params_dict[name]
            except KeyError:
                raise ValueError('there is no parameter with name %s')
            if param.get_value().shape != value.shape:
                raise ValueError('mismatch: parameter has shape %r but value to set has shape %r' %
                                 (param.get_value().shape, value.shape))
            param.set_value(value)

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

    def get_config(self, model_fname=None):
        model_fname = model_fname or os.path.join(self.get_model_dir(), time.strftime('%Y%m%d_%H%M%S_model.pkl'))
        model_fname = self.save_model(model_fname)
        config = {'class': self.__class__,
                  'build_net': self.build_net,
                  'input_shapes': self.input_shapes,
                  'input_names': self.input_names,
                  'transformers': [transformer.get_config() for transformer in self.transformers],
                  'name': self.name,
                  'pretrained_fname': model_fname}
        config.update(self._kwargs)
        return config

    @classmethod
    def from_config(cls, config):
        config['transformers'] = [utils.config.from_config(transformer_config) for transformer_config in config.get('transformers', [])]
        return cls(**config)


class TheanoNetFeaturePredictor(TheanoNetPredictor, predictor.FeaturePredictor):
    def __init__(self, build_net, input_shapes, input_names=None, transformers=None, name=None, pretrained_fname=None,
                 feature_name=None, next_feature_name=None, feature_jacobian_name=None, control_name=None, **kwargs):
        TheanoNetPredictor.__init__(
            self, build_net, input_shapes, input_names=input_names, transformers=transformers, name=name,
            pretrained_fname=pretrained_fname, **kwargs)
        predictor.FeaturePredictor.__init__(
            self, input_shapes, transformers=transformers, name=name,
            feature_name=feature_name, next_feature_name=next_feature_name,
            feature_jacobian_name=feature_jacobian_name, control_name=control_name)

    def get_config(self, model_fname=None):
        config = {**TheanoNetPredictor.get_config(self, model_fname=model_fname),
                  **predictor.FeaturePredictor.get_config(self)}
        return config


class TheanoNetHierarchicalFeaturePredictor(TheanoNetPredictor, predictor.HierarchicalFeaturePredictor):
    def __init__(self, build_net, input_shapes, input_names=None, transformers=None, name=None, pretrained_fname=None,
                 feature_name=None, next_feature_name=None, feature_jacobian_name=None, control_name=None,
                 levels=None, loss_levels=None, map_names=None, next_map_names=None, **kwargs):
        TheanoNetPredictor.__init__(
            self, build_net, input_shapes, input_names=input_names, transformers=transformers, name=name,
            pretrained_fname=pretrained_fname, levels=levels, **kwargs)
        predictor.HierarchicalFeaturePredictor.__init__(
            self, input_shapes, transformers=transformers, name=name,
            feature_name=feature_name, next_feature_name=next_feature_name,
            feature_jacobian_name=feature_jacobian_name, control_name=control_name,
            levels=levels, loss_levels=loss_levels, map_names=map_names, next_map_names=next_map_names)

    def train(self, *args, **kwargs):
        output_names = kwargs.get('output_names') or [('x%d_next_pred' % loss_level, 'x%d_next' % loss_level) for loss_level in self.loss_levels]
        TheanoNetPredictor.train(self, *args, **dict(**kwargs, output_names=output_names))

    def get_config(self, model_fname=None):
        config = {**TheanoNetPredictor.get_config(self, model_fname=model_fname),
                  **predictor.HierarchicalFeaturePredictor.get_config(self)}
        return config


class FcnActionCondEncoderOnlyTheanoNetFeaturePredictor(TheanoNetFeaturePredictor):
    def __init__(self, *args, **kwargs):
        super(FcnActionCondEncoderOnlyTheanoNetFeaturePredictor, self).__init__(*args, **kwargs)
        pretrained_file = kwargs.get('pretrained_file')
        if pretrained_file is None:
            fname = 'models/theano/fcn-32s-pascalcontext.pkl'
            print("Loading weights from pickle file", fname)
            param_dict = pickle.load(open(fname, 'rb'), encoding='latin1')
            all_params = OrderedDict((param.name, param) for param in self.get_all_params() if param.name is not None)
            for level in self.get_levels():
                if level == 0:
                    continue
                for i_conv in [1, 2]:
                    W, b = param_dict['conv%d_%d'%(level, i_conv)]
                    if level == 1 and i_conv == 1:
                        W = W[:, ::-1, :, :] * 255.0 / 2.0
                    param_W = all_params['x%d_conv%d.W'%(level, i_conv)]
                    param_b = all_params['x%d_conv%d.b'%(level, i_conv)]
                    param_W.set_value(W)
                    param_b.set_value(b)
                    for layer in self.get_all_layers():
                        set_layer_param_tags(layer, params=(param_W, param_b), trainable=False)

    def train(self, *train_hdf5_fnames, val_hdf5_fname, solverstate_fname=None,
              batch_size=32,
              test_iter = 10,
              solver_type = 'SGD',
              test_interval = 1000,
              base_lr = 0.05,
              gamma = 0.9,
              stepsize = 1000,
              display = 20,
              max_iter=10000,
              momentum = 0.9,
              momentum2 = 0.999,
              weight_decay=0.0005,
              snapshot=10000, #TODO: 1000
              snapshot_prefix=None,
              visualize_response_maps=False):
        # training loss
        param_l2_penalty = lasagne.regularization.regularize_network_params(self.pred_layers['x0_next_pred'], lasagne.regularization.l2)
        loss = self.loss + weight_decay * param_l2_penalty / 2.

        # training function
        start_time = time.time()
        print("Compiling training function...")
        train_fn = theano.function([self.X_var, self.U_var, self.X_diff_var], loss)
        print("... finished in %.2f s"%(time.time() - start_time))

        validate = test_interval and val_hdf5_fname is not None
        if validate:
            # validation data
            val_minibatches = iterate_minibatches_indefinitely(val_hdf5_fname, data_names=['image_curr', 'vel', 'image_diff'],
                                                               batch_size=batch_size, shuffle=True)
            # validation loss
            test_loss = self.loss_deterministic + weight_decay * param_l2_penalty / 2.
            # validation function
            start_time = time.time()
            print("Compiling validation function...")
            val_fn = theano.function([self.X_var, self.U_var, self.X_diff_var], test_loss)
            print("... finished in %.2f s"%(time.time() - start_time))

        if solverstate_fname is not None and not solverstate_fname.endswith('.pkl'):
            solverstate_fname = self.get_snapshot_prefix() + '_iter_' + solverstate_fname + '.pkl'
        self.restore_solver(solverstate_fname)

        start_time = time.time()
        print("Aggregating matrices...")
        N = 0
        Alevels = {}
        Blevels = {}
        post_fit_all = {}
        batch_iter = 0
        for X, U, X_diff in iterate_minibatches_once(*train_hdf5_fnames, data_names=['image_curr', 'vel', 'image_diff'], batch_size=10000):
            print("batch %d"%batch_iter)
            N += X.shape[0]
            Xlevels = self.response_maps_from_input(X)
            Xlevels_next = self.response_maps_from_input(X + X_diff)
            for level in self.get_levels():
                Xlevel = Xlevels['x%d'%level]
                Xlevel_next = Xlevels_next['x%d'%level]
                c_dim = Xlevel.shape[1]
                if level not in Alevels:
                    Alevels[level] = [0] * c_dim
                    Blevels[level] = [0] * c_dim
                for channel in range(c_dim):
                    Xlevelchannel = Xlevel[:, channel, ...].reshape((Xlevel.shape[0], -1))
                    Xlevelchannel_next = Xlevel_next[:, channel, ...].reshape((Xlevel_next.shape[0], -1))
                    Xlevelchannel_diff = Xlevelchannel_next - Xlevelchannel
                    A, B, post_fit = bilinear.BilinearFunction.compute_solver_terms(Xlevelchannel, U, Xlevelchannel_diff)
                    Alevels[level][channel] += A
                    Blevels[level][channel] += B
                    if level not in post_fit_all:
                        post_fit_all[level] = post_fit
            batch_iter += 1
            if batch_iter == 10:
                break
        print("... finished in %2.f s"%(time.time() - start_time))

        start_time = time.time()
        print("Solving linear systems...")
        all_params = OrderedDict((param.name, param) for param in self.get_all_params() if param.name is not None)
        for level in self.get_levels():
            start_time_level = time.time()
            post_fit = post_fit_all[level]
            Qchannels = []
            Rchannels = []
            Schannels = []
            bchannels = []
            for A, B in zip(Alevels[level], Blevels[level]):
                A = A / (2. * N) + weight_decay * np.diag([1.]*(len(A)-1) + [0.]) # don't regularize bias, which is the last one
                B = B / (2. * N)
                Q, R, S, b = post_fit(np.linalg.solve(A, B))
                Qchannels.append(Q)
                Rchannels.append(R)
                Schannels.append(S)
                bchannels.append(b)
            all_params['x%d_diff_pred.Q'%level].set_value(np.asarray(Qchannels, dtype=theano.config.floatX))
            all_params['x%d_diff_pred.R'%level].set_value(np.asarray(Rchannels, dtype=theano.config.floatX))
            all_params['x%d_diff_pred.S'%level].set_value(np.asarray(Schannels, dtype=theano.config.floatX))
            all_params['x%d_diff_pred.b'%level].set_value(np.asarray(bchannels, dtype=theano.config.floatX))
            print("%2.f s"%(time.time() - start_time_level))
        print("... finished in %2.f s"%(time.time() - start_time))
        import IPython as ipy; ipy.embed()

        X, U, X_diff = next(iterate_minibatches_once(val_hdf5_fname, data_names=['image_curr', 'vel', 'image_diff'], batch_size=100))

#         self.snapshot('exact', snapshot_prefix) # TODO
        loss = train_fn(X, U, X_diff)
        print("    training loss = {:.6f}".format(float(loss)))
        test_loss = sum([val_fn(*next(val_minibatches)) for _ in range(test_iter)]) / test_iter
        print("    validation loss = {:.6f}".format(float(test_loss)))
        # visualize response maps of first image in batch
        if visualize_response_maps:
            self.visualize_response_maps(X[0], U[0], x_next=X[0]+X_diff[0])
        import IPython as ipy; ipy.embed();

