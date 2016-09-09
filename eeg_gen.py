import numpy as np

def noize(mean = 0, std = 10, num_samples = 1):
	whitenoise =   [np.random.normal(mean, std, size=num_samples),
					np.random.normal(mean, std, size=num_samples),
					np.random.normal(mean, std, size=num_samples),
					np.random.normal(mean, std, size=num_samples),
					np.random.normal(mean, std, size=num_samples),
					np.random.normal(mean, std, size=num_samples),
					np.random.normal(mean, std, size=num_samples),
					np.random.normal(mean, std, size=num_samples)]
	return np.array(whitenoise)

noize()