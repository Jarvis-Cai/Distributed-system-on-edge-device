# author:caijiawei
# assistant:HuangBin
# time:2020.11.3
# this file is main file. And it's going to control everything.
# main control model

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
		self.q = Queue(maxsize=queue_size)  # create queue，queue max size is 10
		self.local_ip = self.get_local_ip_address()

	def listener(self):
		"""
		the func is used to listen port 22223
		:return: updated _key_ip_address
		"""
		sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)	# define socket
		sock.bind(('0.0.0.0', 22223))							# bind listen port-----‘0.0.0.0’ used to all open IPV4 addresses in this machine
		while True:
			ret, addr = sock.recvfrom(1024)						# receive UDP data about 1024 byte ，return（data,address）
			if ret:
				self.q.put(ret)									# the receive data is valid，put this data to queue

	def action(self):
		while True:
			command = self.q.get()								# Get data in queue insertion order
			if command == b'start':	
				# start your own business
				message = self.q.get()
				message = json.loads(message)  # transform from bytes into dict format.
				print("PreReformat message: \n",message)
				message = self._reformat_ringkey_to_start(message)
				print('* - ' * 30)
				print("Local device and distributed system initial information: \n" , message)

				instance_3 = Distribute(**message)
				p3 = Process(target=instance_3.main, args=())	# Create processes for distributed machines
				p3.run()										# start distributed

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
		# to adapt Token check function, 
		# Constantly update and maintain the latest distributed network structure information (ring_key)
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
		# ps device default key information
		template_dict = {'job_name': 'ps', 'task_index': 0, 'ps_hosts': ['192.168.0.104:22221'],
						 'worker_hosts':['192.168.0.100:22221', '192.168.0.101:22221'],
						 'batch_size': 2048, 'training_epochs':100, 'learning_rate':1e-3,
						 'train_steps':1200
						 }
		port = '22221'
		worker = message['device'][:-1]  # the last one is server.
		ps_raw = message['server']  # raw ip address
		worker_raw = [x[0] for x in worker]  # raw ip address
		if self.local_ip not in ps_raw:	# worker should reset key information
			template_dict['job_name'] = 'worker'
			for i, c in enumerate(worker_raw):
				if c == self.local_ip:
					template_dict['task_index'] = i	# get local device's task index
"""		if self.local_ip not in worker_raw: # ps reset key information
			template_dict['job_name'] = 'ps'
			for i,c in enumerate(ps_raw):
				if c == self.local_ip:
					template_dict['task_index'] = i # get local device's task index"""
		template_dict['ps_hosts'] = [x + ':' + port for x in ps_raw]
		template_dict['worker_hosts'] = [x + ':' + port for x in worker_raw]
		return template_dict

	def run(self):
		t1 = Thread(target=self.listener, args=())		# create listener Thread
		t2 = Thread(target=self.action, args=())		# create action Thread
		t1.start()
		t2.start()
		print(self.__class__.__name__, 'is running!')	# SystemCommandListener is running!


if __name__ == '__main__':
	ring_key = {'situation': dict(flag_of_ok_device_sum=0, system_is_running=False),
				'device': [['192.168.0.104', 0.0], ['192.168.0.103', 0.0]],
				'server': []
				}
	instance_1 = SystemCommandListener()										# main control Process
	p1 = Process(target=instance_1.run, args=())								
	instance_2 = auto_check.AutoCheck(ring_key=ring_key, start_flag=True)		# Token check Process
	p2 = Process(target=instance_2.run, args=())
	p1.run()
	p2.run()
	print('main_control start successfully!')
