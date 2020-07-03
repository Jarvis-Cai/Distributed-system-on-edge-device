# author:caijiawei
# time:2020.3.9
# this file is to test every single device's computing capability. So we will run a simple neural network and return the
# time needed.

import tensorflow as tf
import cnn_model
import time
import numpy as np


def test_calculate_time():
	start_time = time.time()
	IMAGE_PIXELS = 32
	data = np.random.random([1, IMAGE_PIXELS, IMAGE_PIXELS, 3])
	label = [0]*10
	label[1] = 1
	label = [label]
	x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS, IMAGE_PIXELS, 3])
	y_ = tf.placeholder(tf.float32, [None, 10])

	# graph start
	# Predict
	y = cnn_model.cnn_test(x, 10)

	# graph end
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

	opt = tf.train.AdamOptimizer(0.001)

	train = opt.minimize(cost)

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	# local init_op created
	init_op = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init_op)
		sess.run([train, cost], feed_dict={x: data, y_: label})
	total_time_consumption = time.time() - start_time
	return round(total_time_consumption, 2)


if __name__ == '__main__':
	print('time consumption:', test_calculate_time(), 's')
