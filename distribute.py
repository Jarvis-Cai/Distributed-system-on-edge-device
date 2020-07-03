# -*-coding:GBK -*-
import time
import tensorflow as tf
import cnn_model
import cifar_10_data


class Distribute:
	IMAGE_PIXELS = 32

	def __init__(self, **kwargs):
		"""
		example:
		kwargs = {
		'job_name':'ps',
		'task_index':0,
		'batch_size':2048,
		'ps_hosts':['192.168.0.104:22221'],
		'worker_hosts':['192.168.0.100:22221','192.168.0.101:22221'],
		'training_epochs':5,
		'learning_rate':1e3,
		'train_steps':1200
		}
		:param kwargs:
		"""
		for k, v in kwargs.items():
			setattr(self, k, v)
		print(self.__class__.__name__, 'is starting!')

	def main(self):
		# num_labels = mnist_data.NUM_LABELS
		# train_total_data, train_size, validation_data, validation_labels, test_data, test_labels = \
		#     mnist_data.prepare_MNIST_data(False)
		# total_batch = int(train_size / FLAGS.batch_size)
		if self.job_name == "worker":
			trains, labels, validation_data, validation_labels = cifar_10_data.prepare_data()
			train_size = len(labels)
			total_batch = int(train_size / self.batch_size) + 1
			print(train_size, total_batch)

		if self.job_name is None or self.job_name == '':
			raise ValueError('Must specify an explicit job_name !')
		else:
			print('job_name : %s' % self.job_name)
		if self.task_index is None or self.task_index == '':
			raise ValueError('Must specify an explicit task_index!')
		else:
			print('task_index : %d' % self.task_index)

		ps_spec = self.ps_hosts  # list of char
		worker_spec = self.worker_hosts  # list of char

		# 创建集群
		num_worker = len(worker_spec)
		cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})  # define the whole network configuration
		server = tf.train.Server(cluster, job_name=self.job_name, task_index=self.task_index)

		is_chief = (self.task_index == 0)

		if self.job_name == 'ps':
			server.join()
		elif self.job_name == "worker":
			# worker_device = '/job:worker/task%d/cpu:0' % FLAGS.task_index
			with tf.device(tf.train.replica_device_setter(
					worker_device="/job:worker/task:%d" % self.task_index,
					cluster=cluster)):

				global_step = tf.Variable(0, name='global_step', trainable=False)  # 创建纪录全局训练步数变量
				# x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS*IMAGE_PIXELS])
				x = tf.placeholder(tf.float32, [None, self.IMAGE_PIXELS, self.IMAGE_PIXELS, 3])
				y_ = tf.placeholder(tf.float32, [None, 10])

				# graph start
				# Predict
				# y = cnn_model.CNN(x)
				y, _ = cnn_model.mobilenet_v3_small(x, 10)

				# graph end
				cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

				opt = tf.train.AdamOptimizer(self.learning_rate)

				train_step = opt.minimize(cost, global_step=global_step)

				correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
				# 生成本地的参数初始化操作init_op

				init_op = tf.global_variables_initializer()
				train_dir = '/home/distribute/log'
				saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
				sv = tf.train.Supervisor(is_chief=is_chief, logdir=train_dir, saver=saver, init_op=init_op,
										 recovery_wait_secs=1,
										 save_model_secs=60,
										 global_step=global_step)

				if is_chief:
					print('Worker %d: Initailizing session...' % self.task_index)
				else:
					print('Worker %d: Waiting for session to be initaialized...' % self.task_index)
				# sess = sv.prepare_or_wait_for_session(server.target)
				with sv.managed_session(master=server.target) as sess:
					print('Worker %d: Session initialization  complete.' % self.task_index)
					# writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

					time_begin = time.time()
					print('Traing begins @ %f' % time_begin)

					local_step = 0
					train_acc = 0
					for epoch in range(self.training_epochs):

						# Random shuffling
						# numpy.random.shuffle(train_total_data)
						# train_data_ = train_total_data[:, :-num_labels]
						# train_labels_ = train_total_data[:, -num_labels:]

						# Loop over all batches
						for i in range(total_batch):
							# Compute the offset of the current minibatch in the data.
							offset = (i * self.batch_size) % (train_size)
							batch_xs = trains[offset:offset + self.batch_size]
							batch_ys = labels[offset:offset + self.batch_size]
							train_feed = {x: batch_xs, y_: batch_ys}
							if sv.should_stop():
								break
							_, step, batch_acc = sess.run([train_step, global_step, accuracy], feed_dict=train_feed)
							train_acc += batch_acc
							local_step += 1

							now = time.time()
							print('%f: Worker %d: traing step %d done (global step:%d) epoch: %d--batch:%d'
								  % (now, self.task_index, local_step, step, epoch, i))
							if not step % 10:
								train_acc /= 10
								print('accuracy is:%f' % train_acc)
								train_acc = 0
								if not is_chief:
									saver.save(sess=sess, save_path=train_dir + '/model.ckpt', global_step=global_step)

					time_end = time.time()
					print('Training ends @ %f' % time_end)
					train_time = time_end - time_begin
					print('Training elapsed time:%f s' % train_time)

					val_feed = {x: validation_data, y_: validation_labels}
					val_xent, acc = sess.run([cost, accuracy], feed_dict=val_feed)
					print('After %d training step(s), validation cross entropy = %g, accuracy = %f' % (
					self.train_steps, val_xent, acc))


if __name__ == '__main__':
	dic = {
		'job_name': 'worker',
		'task_index': 0,
		'batch_size': 2048,
		'ps_hosts': ['192.168.0.100:22221'],
		'worker_hosts': ['192.168.0.104:22221', '192.168.0.101:22221'],
		'training_epochs': 5,
		'learning_rate': 1e-3,
		'train_steps': 1200
	}
	Distribute(**dic).main()
