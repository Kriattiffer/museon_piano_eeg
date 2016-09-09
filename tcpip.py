import socket, struct 
import numpy as np

def receive(sock):
	bytes = s.recv(6666)
	bytes = [bytes[i:i+4] for i in range(0, len(bytes), 4)]
	eeg = [int.from_bytes(bb, byteorder='little') for bb in bytes]
	# eeg = [struct.unpack("<", bb)[0] for bb in bytes] # python27
	print(len(eeg))
	eeg = eeg [10:]
	eeg = np.array(eeg)
	print (np.shape(eeg))
	eeg = eeg.reshape((np.shape(eeg)[0]/9,9))
	for a in eeg:
		print(a)
	return eeg

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 1234))
while 1:
	receive(s)
