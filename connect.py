import socket
from matplotlib import pyplot as plt
import pickle

##
fig,axes = plt.subplots(nrows =3, ncols = 1)
axes = axes.flatten()
plt.ion()
##backgrounds = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axes]
##lines = [ax.plot(x,y)[0] for ax in self.axes] )	
##

s= socket.socket()
s.connect(('192.168.0.2',50783))
print "connected"
while 1:
    data = s.recv(4096)
    data = pickle.loads(data)
    alpha = data[0][0][0]
    beta = data[0][1][0]
    gamma = data[0][2][0]

    axes[0].cla()
    axes[1].cla()
    axes[2].cla()
    
    axes[0].plot(alpha,'o')
    axes[1].plot(beta,'o')
    axes[2].plot(gamma,'o')
    plt.pause(0.05)
