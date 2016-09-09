from pylsl import StreamInlet, resolve_stream
# from scipy import signal
import numpy as np
import sys, os, time

class EEG_STREAM(object):
	""" class for EEG\markers streaming, plotting and recording. """
	def __init__(self,  mapnames = {'eeg':'eegdata.mmap', 'markers':'markers.mmap'}, sample_length = 1000):
		''' create objects for later use'''
		self.ie = self.create_streams()
		self.EEG_ARRAY = self.create_array(mmapname=mapnames['eeg'])
		self.MARKER_ARRAY = self.create_array(mmapname=mapnames['markers'], top_exp_length = 1, number_of_channels = 2)
		self.line_counter = 0
		self.line_counter_mark = 0
		self.sample_length = sample_length

	def create_streams(self, stream_type_eeg = 'EEG'):
		''' Opens two LSL streams: one for EEG, another for markers, If error, tries to reconnect several times'''

		print ("Connecting to NIC stream...")
		if stream_type_eeg in [stream.type() for stream in resolve_stream()]:
			streams_eeg = resolve_stream('type', 'EEG')
			inlet_eeg = StreamInlet(streams_eeg[0])   
		return inlet_eeg
	
	def create_array(self, mmapname, top_exp_length = 60, number_of_channels  = 9):
		'''Creates very long array of Nans, which will be filled by EEG. length is determined by maximum length of the experiment in minutes
			The array is mapped to disc for later use from classiffier process'''
		record_length = 500*60*top_exp_length*1.2
		array_shape = (record_length, number_of_channels)
		print ('Creating array with dimensions %s...' %str(array_shape) )
		# a = np.memmap(mmapname, dtype='float', mode='w+', shape=(array_shape))
		a = np.zeros(array_shape, dtype = 'float')
		a[:,0:9] = np.NAN
		print ('... done')
		return a

	def fill_array(self, eeg_array, line_counter, data_chunk, timestamp_chunk, datatype = 'EEG'):
		'''Recieves preallocated array of NaNs, piece of data, piece of offsets and number of line, inserts everything into array. Works both with EEG and with markers '''
		if type(timestamp_chunk) != float:
			length = len(timestamp_chunk)
		else:
			length = 1
		eeg_array[line_counter:line_counter+length, 0] = timestamp_chunk
		eeg_array[line_counter:line_counter+length,1:] = data_chunk
	
	def plot_and_record(self):
		''' Main cycle for recording and plotting FFT. Pulls markers and eeg from lsl inlets, 
		fills preallocated arrays with data. After certain offset calculates FFT and updates plots. Records data on exit.'''
	
		# while 1: #self.stop != True:	
			# pull chunks if Steam_eeg and stream_markess are True	
		try:
			EEG, timestamp_eeg = self.ie.pull_sample()
		except:
			EEG, timestamp_eeg = [], []

		if timestamp_eeg:
			self.fill_array(self.EEG_ARRAY, self.line_counter, EEG, timestamp_eeg, datatype = 'EEG')
			self.line_counter +=1
			# self.line_counter += len(timestamp_eeg)
			if self.line_counter>self.sample_length and self.line_counter % 10 == 0:
				# FFT = self.compute_fft(self.EEG_ARRAY, self.line_counter, sample_length = self.sample_length)
				print (timestamp_eeg,EEG)
				#return FFT

	def butter_filt(self, data, cutoff_array, fs = 500, order=4):
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
		fft = fft/10000
		return fft


#Stream = EEG_STREAM(sample_length = 2000)
#time.sleep(3)
## for a in range(50):
#while 1:
#	Stream.plot_and_record()