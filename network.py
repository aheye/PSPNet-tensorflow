import numpy as np
import tensorflow as tf

DEFAULT_PADDING = 'VALID'
DEFAULT_DATAFORMAT = 'NHWC'

BN_param_map = {'scale':    'gamma',
                'offset':   'beta',
                'variance': 'moving_variance',
                'mean':     'moving_mean'}

def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True, is_training=False, num_classes=21,
                 scale_channels=1):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        self.is_training = is_training
        self.setup(is_training, num_classes, scale_channels=scale_channels)

    def setup(self, is_training):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path, encoding='latin1').item()

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].items():
                    try:
                        if 'bn' in op_name:
                            param_name = BN_param_map[param_name]
                            data = np.squeeze(data)

                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=self.trainable)

    def get_layer_name(self):
        return layer_name
    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')
    @layer
    def zero_padding(self, input, paddings, name):
        pad_mat = np.array([[0,0], [paddings, paddings], [paddings, paddings], [0, 0]])
        return tf.pad(input, paddings=pad_mat, name=name)

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]

        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding,data_format=DEFAULT_DATAFORMAT)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i, c_o])
            output = convolve(input, kernel)

            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def atrous_conv(self,
                    input,
                    k_h,
                    k_w,
                    c_o,
                    dilation,
                    name,
                    relu=True,
                    padding=DEFAULT_PADDING,
                    group=1,
                    biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]

        convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation, padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i, c_o])
            output = convolve(input, kernel)

            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name,
                              data_format=DEFAULT_DATAFORMAT)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        output = tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name,
                              data_format=DEFAULT_DATAFORMAT)
        return output

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:        return tf.nn.softmax(input, name)

    @layer
    def batch_normalization(self, input, name, scale_offset=True, relu=False):
        output = tf.layers.batch_normalization(
            input,
            momentum=0.95,
            epsilon=1e-5,
            training=self.is_training,
            name=name
        )

        if relu:
            output = tf.nn.relu(output)

        return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)

    @layer
    def resize_bilinear(self, input, size, name):
        return tf.image.resize_bilinear(input, size=size, align_corners=True, name=name)

    @layer
    def fst(self, inputs, name, prev_fused=None, out_chan=256, _lambda=0.1):
        """Feature Space Transformation function
        V_feats: Visual space features, 3d tensor
        L_feats: Lidar space features, 3d tensor
        prev_fused: if None, this is the first layer, return fst only
                    if a Tensor, this will be used to create a residual fuse op
        """
        with tf.variable_scope(name) as scope:
            # Concatenate over feature map dimension
            C = tf.concat([inputs[0], inputs[1]], axis=3)
 
            # Compute scalar and offset for FST linear function
            stride = [1, 1, 1, 1]
            c_i = C.get_shape()[-1]
            v_i = inputs[0].get_shape()[-1]
            l_i = inputs[1].get_shape()[-1]

            # Upscale lidar features
            lidar_kernel = self.make_var('lidar_weights', shape=[1, 1, l_i, v_i])
            lidar_scale = tf.nn.conv2d(inputs[1],
                                       filter=lidar_kernel,
                                       strides=stride,
                                       padding="SAME")

            scalar_kernel = self.make_var('scalar_weights', shape=[1, 1, c_i, v_i])
            scalar = tf.nn.conv2d(C, filter=scalar_kernel, strides=stride, padding='SAME', data_format=DEFAULT_DATAFORMAT)

            offset_kernel = self.make_var('offset_weights', shape=[1, 1, c_i, v_i])
            offset = tf.nn.conv2d(C, filter=scalar_kernel, strides=stride, padding='SAME', data_format=DEFAULT_DATAFORMAT) 

            # Compute fst tensor
            fst_kernel = self.make_var('fst_weights', shape=[1, 1, v_i, out_chan])
            fst = tf.nn.conv2d(((lidar_scale * scalar) + offset) + inputs[0],
                               filter=fst_kernel, 
                               strides=stride,
                               padding='SAME')
            # Kill early if prev_fused is not set
            if prev_fused is None:
                return fst

            # Residucal based fuse function
            c_i = prev_fused.get_shape()[-1]
            # Shrink size of prev_fuse if needed
            if prev_fused.get_shape()[-2] > fst.get_shape()[-2]:
                fuse_stride = 2
                fuse_padding = "SAME"
            else:
                fuse_stride = 1
                fuse_padding = "SAME"
            fused_kernel = self.make_var('fused_weights', shape=[fuse_stride, fuse_stride, c_i, out_chan])
            fused = tf.nn.conv2d(prev_fused,
                                 filter=fused_kernel,
                                 strides=[1, fuse_stride, fuse_stride, 1],
                                 padding=fuse_padding)
            fused = fused + (_lambda * fst)
            return fused


