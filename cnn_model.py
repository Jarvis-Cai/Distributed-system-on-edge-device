import tensorflow as tf


def cnn_test(inputs, class_num=10):

    # x = tf.reshape(inputs, [-1, 32, 32, 3])
    x = inputs
    net = tf.layers.conv2d(x, 32, [5, 5], activation=tf.nn.relu, padding='SAME', name='conv1')
    net = tf.layers.max_pooling2d(net, [2, 2], strides=2, padding='VALID', name='pool1')

    net = tf.layers.conv2d(net, 64, [5, 5], activation=tf.nn.relu, padding='SAME', name='conv2')
    net = tf.layers.max_pooling2d(net, [2, 2], strides=2, padding='VALID', name='pool2')

    net = tf.layers.flatten(net, name='flatten3')

    net = tf.layers.dense(net, 1024, activation=tf.nn.relu, name='fc3')
    net = tf.layers.dropout(net, training=True, name='dropout3')
    outputs = tf.layers.dense(net, class_num, name='fco')
    return outputs

########################################


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _batch_normalization_layer(inputs, is_training=True, name='bn'):
    tmp = tf.layers.batch_normalization(inputs=inputs,
                                        training=is_training,
                                        name=name)
    return tmp


def _conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=False, strides=1, padding="SAME"):
    x = tf.layers.conv2d(
        inputs=inputs, filters=filters_num,
        kernel_size=kernel_size, strides=strides, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4),
        padding=padding,  # ('SAME' if strides == 1 else 'VALID'),
        use_bias=use_bias, name=name)
    return x


def _conv_1x1_bn(inputs, filters_num, name, use_bias=True, is_training=True):
    kernel_size = 1
    strides = 1
    x = _conv2d_layer(inputs, filters_num, kernel_size, name=name + "/conv_1p1", use_bias=use_bias, strides=strides)
    x = _batch_normalization_layer(x, is_training=is_training, name=name + '/conv_1p1_bn')
    return x


def _conv_bn_relu(inputs, filters_num, kernel_size, name, use_bias=True, strides=1, is_training=True, activation=tf.nn.relu6):
    x = _conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=use_bias, strides=strides)
    x = _batch_normalization_layer(x, is_training=is_training, name=name + '/cbr_bn')
    x = activation(x)
    return x


def _dwise_conv(inputs, k_h=3, k_w=3, depth_multiplier=1, strides=(1, 1),
                padding='SAME', name='dwise_conv', use_bias=False,
                ):
    kernel_size = (k_w, k_h)
    in_channel = inputs.get_shape().as_list()[-1]
    filters = int(in_channel*depth_multiplier)
    return tf.layers.separable_conv2d(inputs, filters, kernel_size,
                                      strides=strides, padding=padding,
                                      data_format='channels_last', dilation_rate=(1, 1),
                                      depth_multiplier=depth_multiplier, activation=None,
                                      use_bias=use_bias, name=name)


def relu6(x, name='relu6'):
    return tf.nn.relu6(x, name)


def hard_swish(x, name='hard_swish'):
    with tf.variable_scope(name):
        h_swish = x * tf.nn.relu6(x + 3) / 6
    return h_swish


def hard_sigmoid(x, name='hard_sigmoid'):
    with tf.variable_scope(name):
        h_sigmoid = tf.nn.relu6(x + 3) / 6
    return h_sigmoid


def _fully_connected_layer(inputs, units, name="fc", activation=None, use_bias=True, is_training=True):
    tmp = tf.layers.dense(inputs, units, activation=activation, use_bias=use_bias,
                          name=name, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4))
    # tmp = tf.layers.dropout(tmp, rate=0.2, training=is_training)
    return tmp


def _global_avg(inputs, pool_size, strides, padding='same', name='global_avg'):
    return tf.layers.average_pooling2d(inputs, pool_size, strides,
                                       padding=padding, data_format='channels_last', name=name)


def _squeeze_excitation_layer(inputs, out_dim, ratio, layer_name, is_training=True):
    with tf.variable_scope(layer_name):
        squeeze = _global_avg(inputs, pool_size=inputs.get_shape()[1:-1], strides=1)

        excitation = _fully_connected_layer(squeeze, units=out_dim / ratio, name='excitation1',
                                            is_training=is_training)
        excitation = relu6(excitation)
        # excitation = _conv_bn_relu(inputs=squeeze, filters_num=out_dim//ratio, kernel_size=1, name=layer_name+'/sel_cbn_1',
        #                            is_training=is_training)
        excitation = _fully_connected_layer(excitation, units=out_dim, name='excitation2',
                                            is_training=is_training)
        # excitation = _conv_bn_relu(inputs=excitation, filters_num=out_dim, kernel_size=1, name=layer_name+'/sel_cbn_2',
        #                            activation=hard_sigmoid, is_training=is_training)

        excitation = hard_sigmoid(excitation)
        # excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = inputs * excitation
        return scale


def mobilenet_v3_block(inputs, k_s, expansion_ratio, output_dim, stride, name, is_training=True,
                       use_bias=True, shortcut=True, activatation="RE", ratio=16, se=False):
    bottleneck_dim = expansion_ratio

    with tf.variable_scope(name):
        # pw mobilenetV2
        net = _conv_1x1_bn(inputs, bottleneck_dim, name="pw", use_bias=use_bias)

        if activatation == "HS":
            net = hard_swish(net)
        elif activatation == "RE":
            net = relu6(net)
        else:
            raise NotImplementedError

        # dw
        net = _dwise_conv(net, k_w=k_s, k_h=k_s, strides=[stride, stride], name='dw',
                          use_bias=use_bias)

        net = _batch_normalization_layer(net, is_training=is_training, name='dw_bn')

        if activatation == "HS":
            net = hard_swish(net)
        elif activatation == "RE":
            net = relu6(net)
        else:
            raise NotImplementedError
        # pw & linear
        net = _conv_1x1_bn(net, output_dim, name="pw_linear", use_bias=use_bias)
        # squeeze and excitation
        if se:
            channel = net.get_shape().as_list()[-1]
            net = _squeeze_excitation_layer(net, out_dim=channel, ratio=ratio, layer_name='se_block')

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            net += inputs
            net = tf.identity(net, name='block_output')
    return net


def mobilenet_v3_small(inputs, classes_num, multiplier=1.0, is_training=True, reuse=None):
    end_points = {}
    # layers = [
    #     [16, 16, 3, 2, "RE", True, 16],
    #     [16, 24, 3, 2, "RE", False, 72],
    #     [24, 24, 3, 1, "RE", False, 88],
    #     [24, 40, 5, 2, "RE", True, 96],
    #     [40, 40, 5, 1, "RE", True, 240],
    #     [40, 40, 5, 1, "RE", True, 240],
    #     [40, 48, 5, 1, "HS", True, 120],
    #     [48, 48, 5, 1, "HS", True, 144],
    #     [48, 96, 5, 2, "HS", True, 288],
    #     [96, 96, 5, 1, "HS", True, 576],
    #     [96, 96, 5, 1, "HS", True, 576],
    # ]
    layers = [
        [16, 16, 3, 1, "RE", True, 16],
        [16, 24, 3, 2, "RE", False, 72],
        [24, 24, 3, 1, "RE", False, 88],
        [24, 40, 3, 1, "RE", True, 96],
        [40, 40, 3, 1, "RE", True, 240],
        [40, 40, 3, 1, "RE", True, 240],
        [40, 48, 3, 2, "HS", True, 120],
        [48, 48, 3, 1, "HS", True, 144],
        [48, 96, 5, 2, "HS", True, 288],
    ]
    # input_size = inputs.get_shape().as_list()[1:-1]
    # assert ((input_size[0] % 32 == 0) and (input_size[1] % 32 == 0))

    reduction_ratio = 4
    with tf.variable_scope("MobilenetV3_small", reuse=reuse):
        init_conv_out = _make_divisible(16 * multiplier)
        x = _conv_bn_relu(inputs, filters_num=init_conv_out, kernel_size=3, name='init_convolution', use_bias=False,
                          strides=1,
                          is_training=is_training, activation=hard_swish)

        for idx, (in_channels, out_channels, kernel_size, stride, activatation, se, exp_size) in enumerate(layers):
            in_channels = _make_divisible(in_channels * multiplier)
            out_channels = _make_divisible(out_channels * multiplier)
            exp_size = _make_divisible(exp_size * multiplier)
            x = mobilenet_v3_block(x, kernel_size, exp_size, out_channels, stride,
                                   "bneck{}".format(idx), is_training=is_training, use_bias=True,
                                   shortcut=(in_channels == out_channels), activatation=activatation,
                                   ratio=reduction_ratio, se=se)
            end_points["bneck{}".format(idx)] = x

        conv1_in = _make_divisible(96 * multiplier)
        conv1_out = _make_divisible(128 * multiplier)
        x = _conv_bn_relu(x, filters_num=conv1_out, kernel_size=1, name="conv1_out",
                          use_bias=True, strides=1, is_training=is_training, activation=hard_swish)

        x = _squeeze_excitation_layer(x, out_dim=conv1_out, ratio=reduction_ratio, layer_name="se1_out",
                                     is_training=is_training)
        print(x.get_shape(), 'hey, I am here.')
        x = _global_avg(x, pool_size=x.get_shape()[1:-1], padding='valid', strides=1)
        # x = _global_avg(x, pool_size=7, strides=1)
        end_points["global_pool"] = x

    with tf.variable_scope('Logits_out', reuse=reuse):
        conv2_in = _make_divisible(576 * multiplier)
        conv2_out = _make_divisible(256 * multiplier)
        x = _conv_bn_relu(x, filters_num=conv2_out, kernel_size=1, name="conv2_out", use_bias=True, strides=1,
                          is_training=is_training, activation=hard_swish)
        end_points["conv2_out_1x1"] = x
        x = tf.layers.dropout(x, rate=0.9)
        x = _conv2d_layer(x, filters_num=classes_num, kernel_size=1, name="conv3_out", use_bias=True, strides=1,
                          )
        print(x.get_shape(), 'hey, I am here.')
        logits = tf.layers.flatten(x)
        logits = tf.identity(logits, name='output')
        end_points["Logits_out"] = logits

    return logits, end_points


def mobilenet_v3_large(inputs, classes_num, multiplier=1.0, is_training=True, reuse=None):
    end_points = {}
    layers = [
        [16, 16, 3, 1, "RE", False, 16],
        [16, 24, 3, 2, "RE", False, 64],
        [24, 24, 3, 1, "RE", False, 72],
        [24, 40, 5, 2, "RE", True, 72],
        [40, 40, 5, 1, "RE", True, 120],

        [40, 40, 5, 1, "RE", True, 120],
        [40, 80, 3, 2, "HS", False, 240],
        [80, 80, 3, 1, "HS", False, 200],
        [80, 80, 3, 1, "HS", False, 184],
        [80, 80, 3, 1, "HS", False, 184],

        [80, 112, 3, 1, "HS", True, 480],
        [112, 112, 3, 1, "HS", True, 672],
        [112, 160, 5, 1, "HS", True, 672],
        [160, 160, 5, 2, "HS", True, 672],
        [160, 160, 5, 1, "HS", True, 960],
    ]

    input_size = inputs.get_shape().as_list()[1:-1]
    assert ((input_size[0] % 32 == 0) and (input_size[1] % 32 == 0))

    reduction_ratio = 4
    with tf.variable_scope('init', reuse=reuse):
        init_conv_out = _make_divisible(16 * multiplier)
        x = _conv_bn_relu(inputs, filters_num=init_conv_out, kernel_size=3, name='init',
                          use_bias=False, strides=2, is_training=is_training, activation=hard_swish)

    with tf.variable_scope("MobilenetV3_large", reuse=reuse):
        for idx, (in_channels, out_channels, kernel_size, stride, activatation, se, exp_size) in enumerate(layers):
            in_channels = _make_divisible(in_channels * multiplier)
            out_channels = _make_divisible(out_channels * multiplier)
            exp_size = _make_divisible(exp_size * multiplier)
            x = mobilenet_v3_block(x, kernel_size, exp_size, out_channels, stride,
                                   "bneck{}".format(idx), is_training=is_training, use_bias=True,
                                   shortcut=(in_channels==out_channels), activatation=activatation,
                                   ratio=reduction_ratio, se=se)
            end_points["bneck{}".format(idx)] = x

        conv1_in = _make_divisible(160 * multiplier)
        conv1_out = _make_divisible(960 * multiplier)
        x = _conv_bn_relu(x, filters_num=conv1_out, kernel_size=1, name="conv1_out",
                          use_bias=True, strides=1, is_training=is_training, activation=hard_swish)
        end_points["conv1_out_1x1"] = x

        x = _global_avg(x, pool_size=x.get_shape()[1:-1], strides=1)
        end_points["global_pool"] = x

    with tf.variable_scope('Logits_out', reuse=reuse):
        conv2_in = _make_divisible(960 * multiplier)
        conv2_out = _make_divisible(1280 * multiplier)
        x = _conv2d_layer(x, filters_num=conv2_out, kernel_size=1, name="conv2", use_bias=True, strides=1)
        x = hard_swish(x)
        end_points["conv2_out_1x1"] = x

        x = _conv2d_layer(x, filters_num=classes_num, kernel_size=1, name="conv3", use_bias=True, strides=1)
        logits = tf.layers.flatten(x)
        logits = tf.identity(logits, name='output')
        end_points["Logits_out"] = logits

    return logits, end_points
