import socket, struct 

def receive(sock):
	bytes = s.recv(66666)
	bytes = [bytes[i:i+4] for i in range(0, len(bytes), 4)]
	eeg = [int.from_bytes(bb, byteorder='little') for bb in bytes]
	# eeg = [struct.unpack("<", bb)[0] for bb in bytes]
	# print(len(eeg))
	return eeg


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 1234))
while 1:
	receive(s)
