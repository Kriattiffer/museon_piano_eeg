from pylsl import StreamInlet, resolve_stream
from scipy import signal
import numpy as np
import sys, os, time, socket, pickle
''' RUN WITH PYTHON27 '''

class EEG_STREAM(object):
	""" class for EEG\markers streaming, plotting and recording. """
	def __init__(self, sample_length = 1000, fft_interval = 10, fft_averaging_bin = 10, max_fft_freq = 60, StreamEeg = True, StreamAcceleration = True, TCP_socket = True):
		''' create objects for later use'''
		self.StreamEeg, self.StreamAcceleration, self.TCP_socket, self.fft_interval, self.sample_length, self.fft_averaging_bin = StreamEeg, StreamAcceleration, TCP_socket, fft_interval, sample_length, fft_averaging_bin
		
		self.EEG_ARRAY = self.create_array()
		self.ACCEL_ARRAY = self.create_array(top_exp_length = 1, number_of_channels = 4)
		self.ie, self.im =  self.create_streams()

		self.line_counter = 0
		self.line_counter_accel = 0
		
		self.fouriers = []
		T = 500.0				#sampling rate, Hz
		self.Bin_resolution = T/sample_length
		self.fourier_x = np.arange(0, max_fft_freq, self.Bin_resolution)
		self.alpha = np.logical_and(self.fourier_x <12, self.fourier_x >8)
		self.beta = np.logical_and(self.fourier_x <28, self.fourier_x >16)
		self.gamma = np.logical_and(self.fourier_x <40, self.fourier_x >30)

	
	def create_streams(self, stream_type_eeg = 'EEG', stream_name_accel = 'Accelerometer', recursion_meter = 0, max_recursion_depth = 3):
		''' Opens two LSL streams: one for EEG, another for Accelerometer, If error, tries to reconnect several times
			Also create TCP/IP socket to send data to Touch Designer
		'''
		if recursion_meter == 0:
			recursion_meter +=1
		elif 0<recursion_meter <max_recursion_depth:
			print ('Trying to reconnect for the %i time \n' % (recursion_meter+1))
			recursion_meter +=1
		else:
			print ('exiting')
			sys.exit()
		inlet_eeg = []; inlet_accel = []
		
		if self.StreamEeg == True:
			print ("Connecting to EEG stream...")
			if stream_type_eeg in [stream.type() for stream in resolve_stream()]:
				streams_eeg = resolve_stream('type', 'EEG')
				inlet_eeg = StreamInlet(streams_eeg[0])   
				try:
					inlet_eeg
					print ('...done \n')
				except NameError:
					print ("Error: Cannot conect to EEG stream\n")
					sys.exit()
			else:
				print ('Error: EEG stream is not available\n')
				sys.exit()
		else:
			inlet_eeg = []

		if self.StreamAcceleration == True:
			print ("Connecting to Accelerometer stream...")
			if stream_name_accel in [stream.type() for stream in resolve_stream()]:
				sterams_markers = resolve_stream('type', stream_name_accel)
				inlet_accel = StreamInlet(sterams_markers[0])   
				try:
					inlet_accel
					print ('...done \n')
				except NameError:
					print ("Error: Cannot conect to Accelerometer stream\n")
					plt.close()
					sys.exit()
			else:
				print ('Error: Accelerometer stream is not available\n')
				return self.create_streams(stream_type_eeg, stream_name_accel, recursion_meter)
		else:
			inlet_accel = []

		if self.TCP_socket == True:
			print ('creating socket for TouchDesigner...')
			self.TD_sock = socket.socket()
			self.TD_sock.bind(('localhost', 50783))
			self.TD_sock.listen(1)
			print ('Waiting for incoming connection...')
			self.conn, self.addr = self.TD_sock.accept()
			print ('...done \n')

		return inlet_eeg, inlet_accel
	
	def create_array(self, top_exp_length = 60, number_of_channels  = 9):
		'''Creates very long array of Nans, which will be filled by EEG. length is determined by maximum length of the experiment in minutes
			The array is mapped to disc for later use from classiffier process'''
		record_length = 500*60*top_exp_length*1.2
		array_shape = (record_length, number_of_channels)
		print 'Creating array with dimensions %s...' %str(array_shape) 
		a = np.zeros(array_shape, dtype = 'float')
		a[:,0:number_of_channels] = np.NAN
		print '... done'
		return a

	def fill_array(self, eeg_array, line_counter, data_chunk, timestamp_chunk, datatype = 'EEG'):
		'''Recieves preallocated array of NaNs, piece of data, piece of offsets and number of line, inserts everything into array. Works both with EEG and with markers '''
		if type(timestamp_chunk) == list:
			length = len(timestamp_chunk)
		else:
			length = 1
		eeg_array[line_counter:line_counter+length, 0] = timestamp_chunk
		eeg_array[line_counter:line_counter+length,1:] = data_chunk
	
	def run_streams(self):
		''' Main cycle for recording and plotting FFT. Pulls markers and eeg from lsl inlets, 
		fills preallocated arrays with data. After certain offset calculates FFT and updates plots. Records data on exit.'''
		CurrentTime = time.time()*1000
		while 1: #self.stop != True:	
			# pull chunks if Steam_eeg and stream_markess are True			
			try:
				EEG, timestamp_eeg = self.ie.pull_sample()
			except:
				EEG, timestamp_eeg = [], []
			try:
				acc, timestamp_accel = self.im.pull_sample()
			except :
				acc, timestamp_accel = [],[]
			

			if timestamp_accel:
				# self.line_counter_accel += len(timestamp_accel)
				self.line_counter_accel +=1
				self.fill_array(self.ACCEL_ARRAY, self.line_counter_accel, acc, timestamp_accel, datatype = 'ACCELERATION')				
				# print timestamp_accel, acc
			
			if timestamp_eeg:
				self.fill_array(self.EEG_ARRAY, self.line_counter, EEG, timestamp_eeg, datatype = 'EEG')
				# self.line_counter += len(timestamp_eeg)
				self.line_counter += 1


				if self.line_counter>self.sample_length and time.time()*1000 - CurrentTime >= self.fft_interval:
					FFT = self.compute_fft(self.EEG_ARRAY, self.line_counter, sample_length = self.sample_length)
					FFT = FFT[0:len(self.fourier_x),:]
					# smooth Fourier spectrum
					if type(self.fouriers) == list:
						if len(self.fouriers) < self.fft_averaging_bin:
							self.fouriers.append(FFT)
						else:
							self.fouriers = np.array(self.fouriers)
							self.sft = type(self.fouriers)
					else:
						self.fouriers[0:-1] = self.fouriers[1:]
						self.fouriers[-1] = FFT
						FFT = np.average(self.fouriers, axis = 0)	

					self.TD_PUSH(FFT, acc)
					CurrentTime = time.time()*1000
					# print (self.line_counter)

	def TD_PUSH(self, FFT, acc):
		FFT = np.abs(FFT)
		fft_bands = [np.average(FFT[band], axis = 0) for band in [self.alpha, self.beta, self.gamma]]
		fft_bands = np.array(fft_bands)#, dtype = 'float16')
		# fft_bands = [FFT[band]for band in [self.alpha, self.beta, self.gamma]]

		data = pickle.dumps([fft_bands, acc])
		print type(fft_bands[0][0])
		print [fft_bands, acc]
		# print fft_bands
		self.conn.send(data)

		# print fft_bands


	def butter_filt(self, data, cutoff_array, fs = 500, order=4):
		'''Butterworth filter'''
		nyq = 0.5 * fs
		normal_cutoff = [a /nyq for a in cutoff_array]
		b, a = signal.butter(order, normal_cutoff, btype = 'bandpass', analog=False)
		data = signal.filtfilt(b, a, data, axis = 0)
		return data

	def compute_fft(self, EEG_ARRAY,offset, sample_length = 1000):
		''' computes fourier transform from slice of EEG_ARRAY. slice is determined by current position and length of the sample to analyze.
		FT should be somehow normalized to fit into graph window - how?'''
		ARRAY_SLICE =  EEG_ARRAY[offset-sample_length:offset,1:]
		ARRAY_SLICE =self.butter_filt(ARRAY_SLICE, [3,40])
		fft = np.fft.rfft(ARRAY_SLICE, axis = 0)
		fft[0] = 0
		fft = fft
		return fft

if __name__ == '__main__':
	Stream = EEG_STREAM(sample_length = 500, fft_interval = 50, fft_averaging_bin = 5)
	Stream.run_streams()