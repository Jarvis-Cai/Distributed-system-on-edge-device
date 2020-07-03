# author:caijiawei
# time:2020.3.9
# this file is main file. And it's going to control everything.

from multiprocessing import Process
import auto_check
import socket
import json
from distribute import Distribute
from queue import Queue
from threading import Thread
# In the main control, we need to set a port to receive control message. set the port as 22223


class SystemCommandListener:
	def __init__(self, queue_size=10):
		self.q = Queue(maxsize=queue_size)  # set
		self.local_ip = self.get_local_ip_address()

	def listener(self):
		"""
		the func is used to listen port 22223
		:return: updated _key_ip_address
		"""
		sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		sock.bind(('0.0.0.0', 22223))
		while True:
			ret, addr = sock.recvfrom(1024)
			if ret:
				self.q.put(ret)

	def action(self):
		while True:
			command = self.q.get()
			if command == b'start':
				# start your own business
				message = self.q.get()
				message = json.loads(message)  # transform from bytes into dict format.
				message = self._reformat_ringkey_to_start(message)
				print('* - ' * 30)
				print(message)
				instance_3 = Distribute(**message)
				p3 = Process(target=instance_3.main, args=())
				p3.run()

			elif command == b'stop':
				p3.terminate()  # stop your own business

	@staticmethod
	def get_local_ip_address():
		"""
		check local ip address
		:return:
		"""
		try:
			s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			s.connect(('8.8.8.8', 80))
			ip = s.getsockname()[0]
		finally:
			s.close()
		return ip

	def _reformat_ringkey_to_start(self, message):
		"""
		ring_key = {'situation': {'flag_of_ok_device_sum': 0, 'system_is_running': False},
					'device': [['192.168.0.104', 0.0], ['192.168.0.100', 0.0], ['192.168.0.101', 0.0]],
					'server': []
					}
		:param message: is ring_key
		:return:dict
		kwargs = {
		'job_name':'ps',
		'task_index':0,
		'batch_size':2048,
		'ps_hosts':['192.168.0.104:22221'],
		'worker_hosts':['192.168.0.100:22221', '192.168.0.101:22221'],
		'training_epochs':5,
		'learning_rate':1e3,
		'train_steps':1200
		}
		"""
		template_dict = {'job_name': 'ps', 'task_index': 0, 'ps_hosts': ['192.168.0.104:22221'],
						 'worker_hosts':['192.168.0.100:22221', '192.168.0.101:22221'],
						 'batch_size': 2048, 'training_epochs':5, 'learning_rate':1e-3,
						 'train_steps':1200
						 }
		port = '22221'
		worker = message['device'][:-1]  # the last one is server.
		ps_raw = message['server']  # raw ip address
		worker_raw = [x[0] for x in worker]  # raw ip address
		if self.local_ip not in ps_raw:
			template_dict['job_name'] = 'worker'
			for i, c in enumerate(worker_raw):
				if c == self.local_ip:
					template_dict['task_index'] = i
		template_dict['ps_hosts'] = [x + ':' + port for x in ps_raw]
		template_dict['worker_hosts'] = [x + ':' + port for x in worker_raw]
		return template_dict

	def run(self):
		t1 = Thread(target=self.listener, args=())
		t2 = Thread(target=self.action, args=())
		t1.start()
		t2.start()
		print(self.__class__.__name__, 'is running!')


if __name__ == '__main__':
	ring_key = {'situation': dict(flag_of_ok_device_sum=0, system_is_running=False),
				'device': [['192.168.0.104', 0.0], ['192.168.0.103', 0.0]],
				'server': []
				}
	instance_1 = SystemCommandListener()
	p1 = Process(target=instance_1.run, args=())
	instance_2 = auto_check.AutoCheck(ring_key=ring_key, start_flag=True)
	p2 = Process(target=instance_2.run, args=())
	p1.run()
	p2.run()
	print('main_control start successfully!')
