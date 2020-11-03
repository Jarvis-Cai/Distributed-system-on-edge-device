# author: caijiawei
# assistant：HuangBin
# time: 2020.11.3
# officially!
# Fault self-detection module (token)

import socket
import time
import subprocess
import json
from threading import Thread
from multiprocessing import Queue
import calculate_speed_test as cst
import start_all_system as sas


class AutoCheck:
	"""
	example:
	ring_key = {'situation': {'flag_of_ok_device_sum': 0, 'system_is_running': False},
				'device': [['192.168.0.104', 0.0], ['192.168.0.100', 0.0], ['192.168.0.101', 0.0]],
				'server': []
				}
	Among all devices, at most one device's start_flag be true. And that means start from inside.
	"""
	def __init__(self, ring_key=None, start_flag=False, size_of_queue=10):
		self.PingLogFile = open('./ping_log.txt','w')
		self.q = Queue(maxsize=size_of_queue)	# create queue，queue max size is 10
		if start_flag:
			self.q.put(ring_key)
		self.local_ip = self.get_local_ip_address()

	@staticmethod
	def get_local_ip_address():
		"""
		check local ip address
		:return:
		"""
		# used udp to get local ip
		try:
			s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			s.connect(('8.8.8.8', 80))							
			ip = s.getsockname()[0]								
		finally:
			s.close()
		return ip

#	@staticmethod
	def judge_if_a_ip_can_reach(self,certain_ip):
		"""
		given a certain ip, judge whether it can be reached
		:param certain_ip:a ip address
		:return:Ture or False
		"""
		# judice certain_ip is ping or not 
		if subprocess.call(["ping", "-c", "2", certain_ip],stdout=self.PingLogFile) == 0 or \
				subprocess.call(["ping", certain_ip],stdout=self.PingLogFile) == 0:  # send two ECHO_REQUEST package
			print("==" * 5 ,certain_ip," is success connected!", "==" * 5)
			return True
		else:
			print("==" * 5 ,"Fail to connect",certain_ip, "==" * 5)
			return False

	def listener(self):
		"""
		the func is used to listen port 22222
		:return: updated _key_ip_address
		"""
		# udp listener
		sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)	
		sock.bind(('0.0.0.0', 22222))
		while True:
			ret, addr = sock.recvfrom(1024)	
			if ret:
				ret = json.loads(ret)							
				self.q.put(ret)									# if data is valid, put data(ret) in queue

	@staticmethod
	def send_a_message(a_message, next_one_ip):
		"""
		this function is used to send message to the next machine
		:param a_message: a updated ring_key
		:param next_one_ip: next machine in the chain
		:return: None
		"""
		time.sleep(3)  # take care of the delay time
		udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		dest_addr = (next_one_ip, 22222)						# ip of next device 
		json_string = json.dumps(a_message).encode()			# package the token information
		udp_socket.sendto(json_string, dest_addr)				# udp sent
		udp_socket.close()
		print('message sending finished!')


	def first_start(self):
		# System operating state transition function
		sendcommand_instance = sas.SendCommand()	# defince start_all_system.py->SendCommand()
		while True:
			_key_ring = self.q.get()				# take out one information in queue :ring_key (system run initial process will block in here)
			ip_of_devices = _key_ring['device']		# ip information of system's each device
			# 'device': [['192.168.0.100', 0.0], ['192.168.0.107', 0.0], ['192.168.0.105', 0.0]]
			Nums_of_devices = len(ip_of_devices)  # nums of devices at present
			situation_of_system = _key_ring['situation']	# Operating information at the moment
			# 'situation': {'flag_of_ok_device_sum': 0, 'system_is_running': False}
			flag_of_ok_device_sum = situation_of_system['flag_of_ok_device_sum']	# ready device number information
			system_is_running = situation_of_system['system_is_running']			# local device ready information
			# judge if 'flag_of_bit' is all ok

			if flag_of_ok_device_sum == Nums_of_devices and not system_is_running:
				# strat to the distribute system
				# launch branch
				_key_ring['situation']['system_is_running'] = True
				sendcommand_instance.start_system(_key_ring)  # start distribute system
				# sas.SendCommand.test_del()
				next_one_ip, _ = self._get_next_one_ip(ip_of_devices)
				self.send_a_message(_key_ring, next_one_ip)
			elif flag_of_ok_device_sum < Nums_of_devices:
				# Preparation Phase, to wake up next device 
				# calculate brach
				# find out yourself and get the capability
				next_one_ip, _ = self._get_next_one_ip(ip_of_devices, if_calculate=True)
				flag_of_ok_device_sum += 1
				_key_ring['situation']['flag_of_ok_device_sum'] = flag_of_ok_device_sum
				print(_key_ring, '\n prepare for starting!')  # test del
				self.send_a_message(_key_ring, next_one_ip)
			elif system_is_running:
				# Operating phase, token information is continuously updated
				# running branch
				next_one_ip, del_one_ip = self._get_next_one_ip(ip_of_devices)
				if del_one_ip and del_one_ip[0] == _key_ring['server'][0]:
					# while server Dropped, system should be restart
					# restart all system
					sendcommand_instance.stop_system()
					time.sleep(3)
					print('restart training system!!!', _key_ring)
					sendcommand_instance.start_system(_key_ring)
				else:
					# worker Dropped, del it in _key_ring
					print('following cluster system is running \n' , _key_ring)  # just for test
				self.send_a_message(_key_ring, next_one_ip)

	# search the next worker from token
	# return：next_one_ip
	#		  del_one_ip
	def _get_next_one_ip(self, ip_of_devices, if_calculate=False):
		Nums_of_devices = len(ip_of_devices)  	# nums of devices at present
		for i, c in enumerate(ip_of_devices): 	# i：Numbering，c：device's ip
			if c[0] == self.local_ip:  			# Find the number position of the IP number of the current device！
				if if_calculate:
					c[1] = cst.test_calculate_time()  # calculate time (Computing performance)----
				i += 1  # next one
				i %= Nums_of_devices
				print(ip_of_devices)  # del
				next_one_ip = ip_of_devices[i][0]		# sucess find next device ip
				break

		del_one_ip = None
		while not (self.judge_if_a_ip_can_reach(next_one_ip)):	# while can not connect next device, it should do something
			del_one_ip = ip_of_devices[i]
			print('remove ', ip_of_devices[i])
			del ip_of_devices[i]  # ++++++++ here add switch server function ++++++++ #
			if i == len(ip_of_devices):  # if i is the last of in the list.
				i = 0
			next_one_ip = ip_of_devices[i][0]

		return next_one_ip, del_one_ip

	def run(self):
		t1 = Thread(target=self.listener, args=())					
		t2 = Thread(target=self.first_start, args=())				
		t1.start()
		t2.start()
		print(self.__class__.__name__, 'start success!')			# AutoCheck start success!


if __name__ == '__main__':
	ring_key = {'situation': {'flag_of_ok_device_sum': 0, 'system_is_running': False},
				'device': [['192.168.0.104', 0.0], ['192.168.0.100', 0.0], ['192.168.0.101', 0.0]],
				'server': []
				}
	AutoCheck(ring_key=ring_key, start_flag=True).run()
