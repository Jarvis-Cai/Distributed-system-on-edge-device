import tensorflow as tf


def _resnet_block_v1(inputs, filters, stride, projection, stage, blockname, TRAINING):
	# defining name basis
	conv_name_base = 'res' + str(stage) + blockname + '_branch'
	bn_name_base = 'bn' + str(stage) + blockname + '_branch'

	with tf.name_scope("conv_block_stage" + str(stage)):
		if projection:
			shortcut = tf.layers.conv2d(inputs, filters, (1, 1),
										strides=(stride, stride),
										name=conv_name_base + '1',
										kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
										reuse=tf.AUTO_REUSE, padding='same', use_bias=False)
			shortcut = tf.layers.batch_normalization(shortcut, name=bn_name_base + '1',
													 training=TRAINING, reuse=tf.AUTO_REUSE)
		else:
			shortcut = inputs

		outputs = tf.layers.conv2d(inputs, filters,
								   kernel_size=(3, 3),
								   strides=(stride, stride),
								   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
								   name=conv_name_base + '2a', reuse=tf.AUTO_REUSE, padding='same',
								   use_bias=False)
		outputs = tf.layers.batch_normalization(outputs, name=bn_name_base + '2a',
												training=TRAINING, reuse=tf.AUTO_REUSE)
		outputs = tf.nn.relu(outputs)

		outputs = tf.layers.conv2d(outputs, filters,
								   kernel_size=(3, 3),
								   strides=(1, 1),
								   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
								   name=conv_name_base + '2b', reuse=tf.AUTO_REUSE, padding='same',
								   use_bias=False)

		outputs = tf.layers.batch_normalization(outputs, name=bn_name_base + '2b',
												training=TRAINING, reuse=tf.AUTO_REUSE)
		outputs = tf.add(shortcut, outputs)
		outputs = tf.nn.relu(outputs)
	return outputs


def _resnet_block_v2(inputs, filters, stride, projection, stage, blockname, TRAINING):
	# defining name basis
	conv_name_base = 'res' + str(stage) + blockname + '_branch'
	bn_name_base = 'bn' + str(stage) + blockname + '_branch'

	with tf.name_scope("conv_block_stage" + str(stage)):
		shortcut = inputs
		outputs = tf.layers.batch_normalization(inputs, name=bn_name_base + '2a',
												training=TRAINING, reuse=tf.AUTO_REUSE)
		outputs = tf.nn.relu(outputs)
		if projection:
			shortcut = tf.layers.conv2d(outputs, filters, (1, 1),
										strides=(stride, stride),
										name=conv_name_base + '1',
										kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
										reuse=tf.AUTO_REUSE, padding='same',
										data_format='channels_last',
										use_bias=False)
			shortcut = tf.layers.batch_normalization(shortcut, name=bn_name_base + '1',
													 training=TRAINING, reuse=tf.AUTO_REUSE)

		outputs = tf.layers.conv2d(outputs, filters,
								   kernel_size=(3, 3),
								   strides=(stride, stride),
								   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
								   name=conv_name_base + '2a', reuse=tf.AUTO_REUSE, padding='same',
								   data_format='channels_last',
								   use_bias=False)

		outputs = tf.layers.batch_normalization(outputs, name=bn_name_base + '2b',
												training=TRAINING, reuse=tf.AUTO_REUSE)
		outputs = tf.nn.relu(outputs)
		outputs = tf.layers.conv2d(outputs, filters,
								   kernel_size=(3, 3),
								   strides=(1, 1),
								   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
								   name=conv_name_base + '2b', reuse=tf.AUTO_REUSE, padding='same',
								   data_format='channels_last',
								   use_bias=False)

		outputs = tf.add(shortcut, outputs)
	return outputs


def inference(images, classes_num, training, filters, n, ver):
	"""Construct the resnet model
	Args:
	  images: [batch*channel*height*width]
	  training: boolean
	  filters: integer, the filters of the first resnet stage, the next stage will have filters*2
	  n: integer, how many resnet blocks in each stage, the total layers number is 6n+2
	  ver: integer, can be 1 or 2, for resnet v1 or v2
	Returns:
	  Tensor, model inference output
	"""
	# Layer1 is a 3*3 conv layer, input channels are 3, output channels are 16
	inputs = tf.layers.conv2d(images, filters=filters, kernel_size=(3, 3), strides=(1, 1),
							  name='conv1', reuse=tf.AUTO_REUSE, padding='same', data_format='channels_last',
							  use_bias=False)

	# no need to batch normal and activate for version 2 resnet.
	if ver == 1:
		inputs = tf.layers.batch_normalization(inputs, name='bn_conv1',
											   training=training, reuse=tf.AUTO_REUSE)
		inputs = tf.nn.relu(inputs)

	for stage in range(3):
		stage_filter = filters * (2 ** stage)
		for i in range(n):
			stride = 1
			projection = False
			if i == 0 and stage > 0:
				stride = 2
				projection = True
			if ver == 1:
				inputs = _resnet_block_v1(inputs, stage_filter, stride, projection,
										  stage, blockname=str(i), TRAINING=training)
			else:
				inputs = _resnet_block_v2(inputs, stage_filter, stride, projection,
										  stage, blockname=str(i), TRAINING=training)

	# only need for version 2 resnet.
	if ver == 2:
		inputs = tf.layers.batch_normalization(inputs, name='pre_activation_final_norm',
											   training=training, reuse=tf.AUTO_REUSE)
		inputs = tf.nn.relu(inputs)

	axes = [1, 2]
	inputs = tf.reduce_mean(inputs, axes)
	inputs = tf.identity(inputs, 'final_reduce_mean')

	# inputs = tf.reshape(inputs, [-1, filters * (2 ** 2)])

	inputs = tf.layers.dense(inputs=inputs, units=classes_num, name='dense1', reuse=tf.AUTO_REUSE)
	return inputs
