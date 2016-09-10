import numpy as np
import time, socket, pickle
''' RUN WITH PYTHON33 '''

def noize(mean = 100, std = 200, num_samples = 1, number_of_channels = 8, dtype = 'float64'):
	'''create array of random numbers'''
	whitenoise =   [np.random.normal(mean, std, size=num_samples) for x in range(number_of_channels)]
	wn =  np.array(whitenoise, dtype = dtype).flatten()
	wn = np.abs(wn)
	return wn

def emulate():
	''' create TCP socket and start feeding TouchDesigner random data'''
	
	# socket
	print ('creating socket for TouchDesigner...')
	TD_sock = socket.socket()
	TD_sock.bind(('localhost', 50783))
	TD_sock.listen(1)
	print ('Waiting for incoming connection...')
	conn, addr = TD_sock.accept()
	print ('...done \n')

	#data
	currentTime = time.time()*1000.0
	while 1:
		if time.time()*1000.0 - currentTime <= DELTA_T:
			pass
		else:
			spectrum = np.array([noize(), noize(), noize()])
			accl =  list(noize(number_of_channels = 3, mean = 200, std = 2000, dtype = 'float16'))
			currentTime = time.time()*1000.0
			data = pickle.dumps([spectrum, accl])
			# print [spectrum, accl]
			conn.send(data)


if __name__ == '__main__':

	DELTA_T = 50 # ms
	emulate()