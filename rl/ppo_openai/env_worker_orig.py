

import zmq
import sys
from dm_control import suite
#from light import *
from light import utils


ADDR = sys.argv[1]
task = sys.argv[2], sys.argv[3]

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(ADDR)


#env = suite.load(*task)
env = utils.Pytorch_DMC_Env(*task)
env.reset()

while True:
	msg = socket.recv_pyobj()
	ret = None

	if msg[0] == 'exit':
		socket.send_pyobj(None)
		break

	elif msg[0] == 'action':
		ret = env.step(msg[1])

	elif msg[0] == 'reset':
		ret = env.reset()

	elif msg[0] == 'step':
		action = msg[1]
		ob, reward, done, _ = env.step(action)
		ret = (ob, reward, done)
		# if done:
		# 	# Reset observation when done
		# 	ret = (env.reset(), reward, done)

	else:
		raise RuntimeError("Invalid command: ", msg[0])

	socket.send_pyobj(ret)
